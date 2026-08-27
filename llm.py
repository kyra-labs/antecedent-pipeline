"""
Provider-agnostic LLM client.

Supports two wire protocols:

  openai     POST {base_url}/chat/completions
             Authorization: Bearer <key>
             system passed as a message with role="system"
             supports response_format={"type":"json_object"}

  anthropic  POST {base_url}/messages
             x-api-key: <key>, anthropic-version: 2023-06-01
             system passed as a top-level string
             max_tokens is REQUIRED
             no response_format -> JSON forced via assistant prefill

Any provider speaking either protocol works: DeepSeek, Xiaomi MiMo,
OpenRouter, Groq, a local vLLM server, etc.

Configuration lives entirely in .env so switching providers never
touches pipeline code. See ENV_EXAMPLE at the bottom of this file.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

# ---------------------------------------------------------------------------
# Known pricing, USD per 1M tokens. Used only for cost reporting.
# Verify against your provider's current pricing page; these drift.
# ---------------------------------------------------------------------------

PRICING: dict[str, tuple[float, float]] = {
    # model string            (input, output)
    "deepseek-chat":          (0.14,  0.28),
    "mimo-v2.5":              (0.105, 0.28),
    "mimo-v2.5-pro":          (0.348, 0.696),
}

DEFAULT_PRICING = (0.20, 0.50)   # conservative fallback for unknown models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str
    protocol: str          # "openai" | "anthropic"
    base_url: str
    api_key: str
    model: str

    def __post_init__(self) -> None:
        if self.protocol not in ("openai", "anthropic"):
            raise ValueError(
                f"{self.name}: protocol must be 'openai' or 'anthropic', "
                f"got {self.protocol!r}"
            )
        self.base_url = self.base_url.rstrip("/")

    @property
    def pricing(self) -> tuple[float, float]:
        key = self.model.lower()
        if key in PRICING:
            return PRICING[key]
        # Prefix match for dated variants, e.g. "mimo-v2.5-pro-20260422".
        # Longest first, so "-pro" is not swallowed by the base model name.
        for known in sorted(PRICING, key=len, reverse=True):
            if key.startswith(known):
                return PRICING[known]
        return DEFAULT_PRICING


def load_provider(role: str) -> ProviderConfig:
    """
    Load a provider by task role: "classify" or "brief".

    Reads LLM_<ROLE>_PROVIDER to pick a provider prefix, then reads that
    provider's settings. Falls back to LLM_DEFAULT_PROVIDER.

    Example .env:
        LLM_DEFAULT_PROVIDER=MIMO
        LLM_CLASSIFY_PROVIDER=MIMO
        LLM_BRIEF_PROVIDER=DEEPSEEK

        MIMO_PROTOCOL=openai
        MIMO_BASE_URL=https://api.mimo.mi.com/v1
        MIMO_API_KEY=...
        MIMO_MODEL=mimo-v2.5
    """
    prefix = (
        os.environ.get(f"LLM_{role.upper()}_PROVIDER")
        or os.environ.get("LLM_DEFAULT_PROVIDER")
    )
    if not prefix:
        raise RuntimeError(
            "No provider configured. Set LLM_DEFAULT_PROVIDER in .env "
            "(see ENV_EXAMPLE in llm.py)."
        )
    prefix = prefix.strip().upper()

    missing = [
        var for var in ("BASE_URL", "API_KEY", "MODEL")
        if not os.environ.get(f"{prefix}_{var}")
    ]
    if missing:
        raise RuntimeError(
            f"Provider {prefix} is missing: "
            + ", ".join(f"{prefix}_{m}" for m in missing)
        )

    return ProviderConfig(
        name=prefix,
        protocol=os.environ.get(f"{prefix}_PROTOCOL", "openai").strip().lower(),
        base_url=os.environ[f"{prefix}_BASE_URL"],
        api_key=os.environ[f"{prefix}_API_KEY"],
        model=os.environ[f"{prefix}_MODEL"],
    )


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    per_provider: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, provider: ProviderConfig, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok

        rate_in, rate_out = provider.pricing
        key = f"{provider.name}/{provider.model}"
        bucket = self.per_provider.setdefault(
            key, {"calls": 0, "in": 0, "out": 0, "usd": 0.0}
        )
        bucket["calls"] += 1
        bucket["in"] += in_tok
        bucket["out"] += out_tok
        bucket["usd"] += in_tok / 1e6 * rate_in + out_tok / 1e6 * rate_out

    @property
    def total_usd(self) -> float:
        return sum(b["usd"] for b in self.per_provider.values())

    def summary(self) -> str:
        if not self.per_provider:
            return "no LLM calls"
        lines = [f"total ${self.total_usd:.4f} across {self.calls} calls"]
        for key, b in sorted(self.per_provider.items()):
            lines.append(
                f"    {key}: {int(b['calls'])} calls, "
                f"in {int(b['in']):,} / out {int(b['out']):,}, "
                f"${b['usd']:.4f}"
            )
        if self.failures:
            lines.append(f"    failures: {self.failures}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    pass


class LLMClient:
    """One client per task role. Reuses a session for connection pooling."""

    def __init__(self, role: str, usage: Usage, timeout: int = 120):
        self.provider = load_provider(role)
        self.usage = usage
        self.timeout = timeout
        self.session = requests.Session()

    def describe(self) -> str:
        return (
            f"{self.provider.name} / {self.provider.model} "
            f"({self.provider.protocol} protocol)"
        )

    # -- public API ---------------------------------------------------------

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        retries: int = 3,
    ) -> dict | None:
        """
        Ask for a JSON object and return it parsed.
        Returns None if the call or the parse fails after all retries.
        """
        for attempt in range(retries):
            try:
                text = self._request(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    want_json=True,
                )
                return _parse_json(text)

            except LLMError as exc:
                if _is_rate_limit(exc):
                    time.sleep(5 * (attempt + 1))
                    continue
                if attempt == retries - 1:
                    self.usage.failures += 1
                    return None
                time.sleep(2 * (attempt + 1))

            except (json.JSONDecodeError, ValueError):
                # Model returned prose instead of JSON. Retry with a nudge.
                if attempt == retries - 1:
                    self.usage.failures += 1
                    return None
                user = user + "\n\nRespond with ONLY a JSON object."
                time.sleep(1)

        self.usage.failures += 1
        return None

    def complete_text(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        retries: int = 3,
    ) -> str | None:
        for attempt in range(retries):
            try:
                return self._request(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    want_json=False,
                )
            except LLMError as exc:
                if attempt == retries - 1:
                    self.usage.failures += 1
                    return None
                time.sleep((5 if _is_rate_limit(exc) else 2) * (attempt + 1))
        return None

    # -- protocol adapters -------------------------------------------------

    def _request(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        want_json: bool,
    ) -> str:
        if self.provider.protocol == "anthropic":
            return self._request_anthropic(
                system, user, max_tokens, temperature, want_json
            )
        return self._request_openai(
            system, user, max_tokens, temperature, want_json
        )

    def _request_openai(
        self, system: str, user: str,
        max_tokens: int, temperature: float, want_json: bool,
    ) -> str:
        payload: dict = {
            "model": self.provider.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if want_json:
            payload["response_format"] = {"type": "json_object"}

        resp = self.session.post(
            f"{self.provider.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.provider.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            # Some gateways reject response_format. Retry once without it.
            if want_json and resp.status_code == 400:
                payload.pop("response_format", None)
                payload["messages"][0]["content"] = (
                    system + "\n\nRespond with a single JSON object and nothing else."
                )
                resp = self.session.post(
                    f"{self.provider.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.provider.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        usage = data.get("usage", {}) or {}
        self.usage.record(
            self.provider,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

        choice = data["choices"][0]["message"]
        # Reasoning models may put the answer in content and thinking elsewhere
        return (choice.get("content") or "").strip()

    def _request_anthropic(
        self, system: str, user: str,
        max_tokens: int, temperature: float, want_json: bool,
    ) -> str:
        messages: list[dict] = [{"role": "user", "content": user}]

        # No response_format on this protocol. Prefill an opening brace so
        # the model has no room to start with prose.
        if want_json:
            messages.append({"role": "assistant", "content": "{"})

        payload = {
            "model": self.provider.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,       # required by this protocol
            "temperature": temperature,
        }

        resp = self.session.post(
            f"{self.provider.base_url}/messages",
            headers={
                "x-api-key": self.provider.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        usage = data.get("usage", {}) or {}
        self.usage.record(
            self.provider,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        text = "".join(parts).strip()

        # Re-attach the prefilled brace we sent
        if want_json and not text.startswith("{"):
            text = "{" + text

        return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rate_limit(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "rate" in message.lower()


def _parse_json(text: str) -> dict:
    """Tolerate fences, leading prose, and trailing commentary."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced object
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:index + 1])

    raise ValueError("unbalanced JSON object in response")


def selftest() -> None:
    """python llm.py -- verifies config and makes one cheap real call."""
    from dotenv import load_dotenv
    load_dotenv()

    usage = Usage()
    for role in ("classify", "brief"):
        try:
            client = LLMClient(role, usage)
        except Exception as exc:
            print(f"  {role:9s} NOT CONFIGURED  ({exc})")
            continue

        print(f"  {role:9s} {client.describe()}")
        result = client.complete_json(
            system=(
                'You are a test harness. Respond with exactly '
                '{"ok": true, "model_says": "<your model name>"} and nothing else.'
            ),
            user="Reply with the JSON object.",
            max_tokens=100,
        )
        print(f"             response: {result}")

    print("\n  " + usage.summary().replace("\n", "\n  "))


ENV_EXAMPLE = """
# ---- pick which provider handles which task ----
LLM_DEFAULT_PROVIDER=MIMO
LLM_CLASSIFY_PROVIDER=MIMO       # high volume, cheap model
LLM_BRIEF_PROVIDER=DEEPSEEK      # low volume, quality matters

# ---- Xiaomi MiMo, OpenAI-compatible endpoint ----
MIMO_PROTOCOL=openai
MIMO_BASE_URL=https://api.mimo.mi.com/v1
MIMO_API_KEY=your-token-here
MIMO_MODEL=mimo-v2.5

# ---- same provider via its Anthropic-compatible endpoint ----
# MIMO_PROTOCOL=anthropic
# MIMO_BASE_URL=https://api.mimo.mi.com/anthropic
# MIMO_MODEL=mimo-v2.5

# ---- DeepSeek ----
DEEPSEEK_PROTOCOL=openai
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_MODEL=deepseek-chat
"""


if __name__ == "__main__":
    print("LLM provider self-test\n")
    selftest()