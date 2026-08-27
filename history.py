"""
The stages that make this a context engine rather than a news reader.

  history.py  Stage 8  - find what already happened, from our own database
              Stage 9  - write a briefing grounded in that history
              Stage 10 - verify the briefing did not invent anything
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from llm import LLMClient

# Entities appearing in more than this share of the corpus carry almost no
# retrieval signal. "Google" matches hundreds of unrelated stories; feeding
# those in as history produces confused backstories.
GENERIC_ENTITY_THRESHOLD = 0.02

HISTORY_LOOKBACK_MONTHS = 24
MAX_HISTORY_EVENTS = 5


# ---------------------------------------------------------------------------
# Stage 8 - retrieval
# ---------------------------------------------------------------------------

def load_generic_entity_ids(supabase) -> set[str]:
    """
    Entity ids too common to be useful for retrieval. Computed once per run.
    """
    total = (
        supabase.table("events")
        .select("id", count="exact")
        .limit(1)
        .execute()
        .count
        or 0
    )
    if total == 0:
        return set()

    ceiling = max(int(total * GENERIC_ENTITY_THRESHOLD), 5)

    counts: dict[str, int] = {}
    offset = 0
    page = 1000
    while True:
        rows = (
            supabase.table("event_entities")
            .select("entity_id")
            .range(offset, offset + page - 1)
            .execute()
            .data
        )
        if not rows:
            break
        for row in rows:
            counts[row["entity_id"]] = counts.get(row["entity_id"], 0) + 1
        if len(rows) < page:
            break
        offset += page

    return {eid for eid, n in counts.items() if n > ceiling}


def resolve_entity_ids(supabase, entities: list[dict[str, str]]) -> list[str]:
    """Map classified entity names onto stored entity ids."""
    ids: list[str] = []
    for entity in entities:
        rows = (
            supabase.table("entities")
            .select("id")
            .eq("canonical_name", entity["name"])
            .eq("entity_type", entity.get("type", "technology"))
            .execute()
            .data
        )
        if rows:
            ids.append(rows[0]["id"])
    return ids


def retrieve_history(
    supabase,
    entity_ids: list[str],
    generic_ids: set[str],
    exclude_event_ids: set[str],
    limit: int = MAX_HISTORY_EVENTS,
) -> list[dict[str, Any]]:
    """
    Find prior events sharing a SPECIFIC entity with this story.

    Scoring favours events that share more entities, then recency, then
    importance. Generic entities are excluded entirely rather than
    down-weighted, because one "Google" match would otherwise dominate.
    """
    specific = [eid for eid in entity_ids if eid not in generic_ids]
    if not specific:
        return []

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=30 * HISTORY_LOOKBACK_MONTHS)
    ).isoformat()

    links = (
        supabase.table("event_entities")
        .select("event_id, entity_id")
        .in_("entity_id", specific)
        .execute()
        .data
    )
    if not links:
        return []

    overlap: dict[str, int] = {}
    for link in links:
        event_id = link["event_id"]
        if event_id in exclude_event_ids:
            continue
        overlap[event_id] = overlap.get(event_id, 0) + 1

    if not overlap:
        return []

    candidate_ids = sorted(overlap, key=overlap.get, reverse=True)[: limit * 6]

    events = (
        supabase.table("events")
        .select(
            "id, title, what_happened, category, event_type, "
            "importance_score, first_seen_at, origin"
        )
        .in_("id", candidate_ids)
        .gte("first_seen_at", cutoff)
        .execute()
        .data
    )

    for event in events:
        event["_shared"] = overlap.get(event["id"], 0)

    events.sort(
        key=lambda e: (
            e["_shared"],
            e["first_seen_at"],
            e["importance_score"],
        ),
        reverse=True,
    )

    # Return oldest first so the model reads them as a narrative.
    chosen = events[:limit]
    chosen.sort(key=lambda e: e["first_seen_at"])
    return chosen


# ---------------------------------------------------------------------------
# Stage 9 - briefing
# ---------------------------------------------------------------------------

BRIEF_SYSTEM_PROMPT = """You write briefings for a personal technology news
digest. Its purpose is to explain not just what happened, but how the reader
got here. Respond with a single JSON object and nothing else.

Schema:
{
  "headline": string, plain and factual, under 90 characters,
  "what_happened": string, 2-4 sentences on today's news only,
  "background": string, 2-5 sentences placing it in context. Use ONLY the
                supplied prior events. If none were supplied, set this to
                an empty string.
  "what_changed": [string], 2-4 concrete changes from this news,
  "why_it_matters": [string], 1-3 points aimed at a working developer,
  "timeline": [{"date": "YYYY-MM-DD", "text": string}],
  "uncertainties": [string], 0-3 things not yet known or confirmed
}

HARD RULES - violating these makes the briefing useless:
- Every fact must come from the supplied article text or the supplied prior
  events. You have no other knowledge of this story. Do not draw on anything
  you may remember about these companies or products.
- Every date in "timeline" must be a date that appears in the supplied
  material. Never estimate, infer, or approximate a date.
- Timeline entries must be in chronological order, oldest first, and should
  include the prior events plus today's news as the final entry.
- If the prior events are unrelated to today's news, ignore them and leave
  "background" empty. A missing background is far better than a wrong one.
- "uncertainties" should name real open questions, not hedging boilerplate.
  An empty list is fine.
- No speculation about what a company "may" or "is likely to" do next.
"""


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no prior events found in the archive)"
    parts = []
    for event in history:
        date = event["first_seen_at"][:10]
        parts.append(f"- [{date}] {event['title']}\n  {event['what_happened']}")
    return "\n".join(parts)


def generate_briefing(
    client: LLMClient,
    primary_title: str,
    primary_text: str,
    primary_date: str,
    secondary: list[tuple[str, str]],
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """One LLM call per event. This is the expensive stage; call it only for
    events that will plausibly reach the digest."""

    secondary_block = ""
    if secondary:
        secondary_block = "\n\nAdditional coverage:\n" + "\n\n".join(
            f"[{name}] {text[:1500]}" for name, text in secondary[:2]
        )

    user = (
        f"TODAY'S NEWS (published {primary_date})\n"
        f"Title: {primary_title}\n\n"
        f"{primary_text[:7000]}"
        f"{secondary_block}\n\n"
        f"PRIOR EVENTS FROM THE ARCHIVE (use only these for background):\n"
        f"{_format_history(history)}\n"
    )

    return client.complete_json(
        system=BRIEF_SYSTEM_PROMPT,
        user=user,
        max_tokens=1400,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Stage 10 - validation
# ---------------------------------------------------------------------------

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def validate_briefing(
    brief: dict[str, Any],
    primary_date: str,
    history: list[dict[str, Any]],
) -> list[str]:
    """
    Deterministic checks. Returns a list of problems; empty means it passed.

    The date check is the important one. A model that invents a plausible
    date produces a briefing that reads perfectly and is quietly wrong,
    which is the worst possible failure for this product.
    """
    problems: list[str] = []

    allowed_dates = {primary_date[:10]}
    allowed_dates.update(event["first_seen_at"][:10] for event in history)

    headline = (brief.get("headline") or "").strip()
    if not headline:
        problems.append("missing headline")
    elif len(headline) > 140:
        problems.append(f"headline too long ({len(headline)} chars)")

    if not (brief.get("what_happened") or "").strip():
        problems.append("missing what_happened")

    changed = brief.get("what_changed") or []
    if not isinstance(changed, list) or not changed:
        problems.append("what_changed is empty")

    timeline = brief.get("timeline") or []
    if not isinstance(timeline, list):
        problems.append("timeline is not a list")
        timeline = []

    seen_dates: list[str] = []
    for entry in timeline:
        if not isinstance(entry, dict):
            problems.append("malformed timeline entry")
            continue
        date = str(entry.get("date", "")).strip()
        if not DATE_RE.fullmatch(date):
            problems.append(f"bad date format: {date!r}")
            continue
        if date not in allowed_dates:
            problems.append(f"invented date not in evidence: {date}")
        seen_dates.append(date)

    if seen_dates != sorted(seen_dates):
        problems.append("timeline is not chronological")

    background = (brief.get("background") or "").strip()
    if background and not history:
        problems.append("background written with no prior events supplied")

    return problems


def repair_briefing(brief: dict[str, Any], problems: list[str],
                    primary_date: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Drop the offending parts rather than discarding the whole briefing.
    A briefing without a timeline is still useful; one with a fabricated
    timeline is not.
    """
    if any("date" in p or "chronological" in p for p in problems):
        allowed = {primary_date[:10]}
        allowed.update(event["first_seen_at"][:10] for event in history)
        brief["timeline"] = [
            entry for entry in (brief.get("timeline") or [])
            if isinstance(entry, dict)
            and DATE_RE.fullmatch(str(entry.get("date", "")))
            and str(entry["date"]) in allowed
        ]
        brief["timeline"].sort(key=lambda e: e["date"])

    if any("background" in p for p in problems):
        brief["background"] = ""

    return brief
