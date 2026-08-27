"""
Shared pipeline logic used by both backfill.py (one-time historical seed)
and main.py (daily digest run).

Stages 1-6 are identical in both: fetch, canonicalise, filter, extract,
classify, cluster. Only the time window and what happens afterwards differ.
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import feedparser
import requests
import trafilatura
import yaml

from llm import LLMClient

USER_AGENT = (
    "tech-context-digest/0.1 (personal research project; "
    "+https://github.com/YOURNAME/tech-context-digest)"
)

REQUEST_TIMEOUT = 20
POLITE_DELAY = 1.0        # seconds between requests to the same domain
MIN_BODY_CHARS = 400      # below this, treat extraction as failed
MIN_IMPORTANCE = 45       # backfill keeps a higher bar than daily runs

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source",
    "__twitter_impression", "guccounter", "amp",
}

JUNK_TITLE_PATTERNS = [
    # commerce and admin
    r"\bsponsored\b", r"\bdeal[s]?\b", r"\bcoupon\b", r"\bdiscount\b",
    r"\bgiveaway\b", r"\bwe'?re hiring\b", r"\bjob[s]? at\b",
    r"\bblack friday\b", r"\bcyber monday\b", r"\bbest .* deals\b",
    r"\bnewsletter\b", r"\bweekly roundup\b", r"\bwhat we'?re reading\b",

    # first person - a reporter's personal experience is not an event.
    # Deterministic here rather than left to the model, which ignored the
    # same rule when it was only stated in the prompt.
    r"^i\s+(finally|just|tried|tested|used|spent|switched|swapped|wore|ran)\b",
    r"^i'?ve\b", r"^i'?m\b",
    r"\bi (tried|tested|finally|spent \d+|switched to)\b",
    r"\bmy (pixel|iphone|galaxy|laptop|desk|setup|favou?rite)\b",

    # reviews and hands-on
    r"\breview\b", r"\bhands.?on\b", r"\bfirst impressions\b",
    r"\bi've been (using|testing|wearing)\b",
    r"\bis (a|the) (great|good|best|worst|perfect)\b.*\bbut\b",
    r"\bshould you (buy|upgrade|switch)\b",
    r"\bworth (it|buying|the upgrade)\b",

    # guides and listicles
    r"^how to\b", r"^here'?s how\b", r"\bhere'?s how (to|you)\b",
    r"^\d+ [\w\s]{0,20}\b(things|ways|tips|tricks|features|reasons|apps|settings)\b"
    r".*\byou (should|can|need|must|might want)\b",
    r"\byou (should|can|really need to) (enable|try|check|turn on|install|stop)\b",
    r"\bhidden [\w\s]{0,15}(toggles?|features?|settings?|gems?|menus?)\b",
    r"\btips? and tricks?\b", r"\bstep.by.step\b",
    r"\bevery(thing)? you need to know\b",
    r"\bguide to\b", r"\bexplained\b",

    # opinion framing
    r"^on paper\b", r"^why i\b", r"^the case (for|against)\b",
    r"\bis a lie\b", r"\bi'?m dying to\b",
]

JUNK_URL_PATTERNS = [
    r"/deals?/", r"/sponsored/", r"/jobs?/", r"/careers?/",
    r"/advertis", r"/newsletter", r"/tag/", r"/author/", r"/page/",
]


@dataclass
class Source:
    name: str
    domain: str
    category: str
    source_type: str
    feed_url: str
    archive_strategy: str = "none"
    sitemap_url: str | None = None
    reliability_score: int = 50
    db_id: str | None = None


@dataclass
class RawItem:
    """Everything we know about an article before extraction."""
    source: Source
    title: str
    url: str
    canonical_url: str = ""
    published_at: datetime | None = None
    excerpt: str = ""
    body: str = ""
    extraction_status: str = "pending"
    content_hash: str = ""


@dataclass
class Classification:
    content_type: str
    category: str
    event_type: str
    entities: list[dict[str, str]]
    importance: int
    what_happened: str
    reason: str = ""


@dataclass
class Stats:
    fetched: int = 0
    after_dedup: int = 0
    after_junk_filter: int = 0
    extracted: int = 0
    extraction_failed: int = 0
    classified: int = 0
    rejected_irrelevant: int = 0
    rejected_low_importance: int = 0
    events_created: int = 0
    events_merged: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Stage 1 - Archive discovery and fetching
# ----------------------------------------------------------------------------

def http_get(url: str) -> requests.Response | None:
    """GET with a polite user agent. Returns None on any failure."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp
        return None
    except requests.RequestException:
        return None


def parse_feed_entries(source: Source, feed_text: str) -> list[RawItem]:
    """Turn raw feed XML into RawItems. Handles RSS and Atom identically."""
    parsed = feedparser.parse(feed_text)
    items: list[RawItem] = []

    for entry in parsed.entries:
        link = entry.get("link")
        title = entry.get("title")
        if not link or not title:
            continue

        published = None
        for key in ("published_parsed", "updated_parsed"):
            if entry.get(key):
                published = datetime(*entry[key][:6], tzinfo=timezone.utc)
                break

        excerpt = entry.get("summary", "") or ""
        # Some feeds carry the full body in content[0].value
        if entry.get("content"):
            excerpt = entry["content"][0].get("value", excerpt)

        items.append(RawItem(
            source=source,
            title=html.unescape(title).strip(),
            url=link,
            published_at=published,
            excerpt=strip_html(excerpt)[:5000],
        ))

    return items


def fetch_archive(source: Source, cutoff: datetime, max_pages: int = 20) -> list[RawItem]:
    """
    Reach back through a feed's history using the configured strategy.
    Stops when it runs out of pages or crosses the cutoff date.
    """
    items: list[RawItem] = []
    seen_urls: set[str] = set()

    def add(new_items: list[RawItem]) -> tuple[int, bool]:
        """Returns (added_count, hit_cutoff)."""
        added = 0
        hit_cutoff = False
        for item in new_items:
            if item.url in seen_urls:
                continue
            if item.published_at and item.published_at < cutoff:
                hit_cutoff = True
                continue
            seen_urls.add(item.url)
            items.append(item)
            added += 1
        return added, hit_cutoff

    # Page 1 is always the live feed
    resp = http_get(source.feed_url)
    if resp is None:
        print(f"    ! could not reach {source.feed_url}")
        return []
    add(parse_feed_entries(source, resp.text))

    strategy = source.archive_strategy

    if strategy == "wordpress":
        for page in range(2, max_pages + 1):
            time.sleep(POLITE_DELAY)
            url = f"{source.feed_url}?paged={page}"
            resp = http_get(url)
            if resp is None:
                break
            batch = parse_feed_entries(source, resp.text)
            if not batch:
                break
            added, hit_cutoff = add(batch)
            if added == 0 or hit_cutoff:
                break

    elif strategy == "blogger":
        page_size = 25
        for page in range(1, max_pages + 1):
            time.sleep(POLITE_DELAY)
            start = 1 + page * page_size
            url = f"{source.feed_url}?start-index={start}&max-results={page_size}"
            resp = http_get(url)
            if resp is None:
                break
            batch = parse_feed_entries(source, resp.text)
            if not batch:
                break
            added, hit_cutoff = add(batch)
            if added == 0 or hit_cutoff:
                break

    elif strategy == "sitemap":
        sitemap_url = source.sitemap_url or f"https://{source.domain}/sitemap.xml"
        items.extend(fetch_from_sitemap(source, sitemap_url, cutoff, seen_urls))

    return items


def fetch_from_sitemap(
    source: Source,
    sitemap_url: str,
    cutoff: datetime,
    seen_urls: set[str],
) -> list[RawItem]:
    """Very small sitemap reader. Handles one level of sitemap index."""
    items: list[RawItem] = []
    resp = http_get(sitemap_url)
    if resp is None:
        return items

    text = resp.text

    # Sitemap index -> follow child sitemaps (capped)
    child_maps = re.findall(r"<sitemap>.*?<loc>(.*?)</loc>", text, re.S)
    targets = child_maps[:6] if child_maps else [sitemap_url]

    for target in targets:
        if child_maps:
            time.sleep(POLITE_DELAY)
            child = http_get(target)
            if child is None:
                continue
            text = child.text

        blocks = re.findall(r"<url>(.*?)</url>", text, re.S)
        for block in blocks:
            loc = re.search(r"<loc>(.*?)</loc>", block)
            mod = re.search(r"<lastmod>(.*?)</lastmod>", block)
            if not loc:
                continue
            url = loc.group(1).strip()
            if url in seen_urls:
                continue

            published = None
            if mod:
                try:
                    published = datetime.fromisoformat(
                        mod.group(1).strip().replace("Z", "+00:00")
                    )
                except ValueError:
                    published = None
            if published and published < cutoff:
                continue

            seen_urls.add(url)
            items.append(RawItem(
                source=source,
                title=url.rstrip("/").split("/")[-1].replace("-", " ").title(),
                url=url,
                published_at=published,
            ))

    return items


# ----------------------------------------------------------------------------
# Stage 2 - Canonicalisation and cheap filtering
# ----------------------------------------------------------------------------

def strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    # html.unescape handles named AND numeric entities (&#8217; &nbsp; &amp;)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalise(url: str) -> str:
    """Strip tracking params, normalise host, drop fragments."""
    try:
        parts = urlparse(url)
    except ValueError:
        return url

    query = [
        (k, v) for k, v in parse_qsl(parts.query)
        if k.lower() not in TRACKING_PARAMS
    ]

    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    path = parts.path
    if path.endswith("/") and len(path) > 1:
        path = path[:-1]

    return urlunparse((
        parts.scheme or "https",
        host,
        path,
        "",
        urlencode(sorted(query)),
        "",
    ))


def looks_like_junk(item: RawItem) -> str | None:
    """Return a reason string if this should be dropped, else None."""
    title_lower = item.title.lower()
    for pattern in JUNK_TITLE_PATTERNS:
        if re.search(pattern, title_lower):
            return f"junk title ({pattern})"

    url_lower = item.canonical_url.lower()
    for pattern in JUNK_URL_PATTERNS:
        if re.search(pattern, url_lower):
            return f"junk url ({pattern})"

    return None


# ----------------------------------------------------------------------------
# Stage 3 - Text extraction
# ----------------------------------------------------------------------------

def extract_body(item: RawItem) -> None:
    """
    Fallback chain:
      1. feed excerpt already long enough
      2. trafilatura on fetched HTML
      3. feed excerpt as-is
      4. metadata only
    Mutates item in place.
    """
    if len(item.excerpt) >= 2000:
        item.body = item.excerpt
        item.extraction_status = "full"
        return

    time.sleep(POLITE_DELAY)
    resp = http_get(item.url)
    if resp is not None:
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if text and len(text) >= MIN_BODY_CHARS:
            item.body = text
            item.extraction_status = "full"
            return

    if len(item.excerpt) >= MIN_BODY_CHARS:
        item.body = item.excerpt
        item.extraction_status = "partial"
        return

    item.body = item.excerpt or item.title
    item.extraction_status = "metadata_only"


# ----------------------------------------------------------------------------
# Stage 4 - LLM classification
# ----------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """You classify technology articles for a personal
news-context database. You will be given one article. Respond with a single JSON
object and nothing else. No markdown fences, no commentary.

Schema:
{
  "content_type": one of ["news_event","analysis","review","guide","opinion",
                          "listicle","interview","announcement_marketing"],
  "category": one of ["android","kotlin","artificial-intelligence",
                      "developer-tools","open-source","cybersecurity",
                      "hardware","technology","other"],
  "event_type": one of ["product_release","acquisition","funding","outage",
                        "vulnerability","research","policy","legal",
                        "deprecation","partnership","other"],
  "entities": [{"name": string, "type": "company"|"product"|"person"|"technology"}],
  "importance": integer 0-100,
  "what_happened": string, 2-3 sentences, factual, no speculation,
  "reason": string, one short sentence explaining the importance score
}

CONTENT_TYPE - decide this first, and be literal about it:
  news_event   Reports a specific thing that happened on a specific date.
               A release, arrest, breach, lawsuit, acquisition, outage,
               published research, policy change.
  analysis     Explains or contextualises a recent event without being the
               first report of it.
  review       Evaluates a product from hands-on use.
  guide        Teaches the reader to do something. Any "how to", tutorial,
               best-practices or lessons-learned post.
  opinion      Argues a position. First-person experience pieces.
  listicle     Organised as a numbered set of items, tips or settings.
  interview    Primarily someone's quotes.
  announcement_marketing
               A vendor promoting its own product without a dated event.

  Test: could a neutral observer write "On [date], X happened" from this?
  If not, it is not news_event. A guide about a feature is not the release
  of that feature. A reporter's personal experience is not an event.

ENTITIES - specificity is critical. These are the retrieval index, and a
parent company name matches hundreds of unrelated stories.
- Name the most specific thing: a product, version, codename, incident,
  standard or law.
  GOOD: "Android 16 QPR2", "Kimwolf botnet", "Gemini 3", "Kotlin 2.3",
        "Model Context Protocol", "SynthID"
  BAD:  "Google", "AI", "software", "smartphone"
- A bare company name ONLY when the company itself is the subject:
  acquisition, layoffs, earnings, a lawsuit against it.
- Include version numbers for releases. 2 to 5 entities.

IMPORTANCE is lasting significance to a working software developer, not
click appeal. Routine point release 20-40. Major platform release or
serious vulnerability 70-90. Consumer pricing and executive moves rarely
exceed 50.

what_happened must only contain facts present in the supplied text.
"""


# Only these reach a digest. Everything else is filtered before the
# ranking stage. Kept in code rather than the prompt so the rule is
# auditable and cannot be argued away by the model.
NEWS_CONTENT_TYPES = {"news_event", "analysis"}

VALID_CATEGORIES = {
    "android", "kotlin", "artificial-intelligence", "developer-tools",
    "open-source", "cybersecurity", "hardware", "technology", "other",
}


def classify(item: RawItem, client: LLMClient,
             errors: list[str] | None = None) -> Classification | None:
    """One LLM call via whichever provider is configured for 'classify'."""
    published = item.published_at.date().isoformat() if item.published_at else "unknown"

    user_content = (
        f"Publisher: {item.source.name}\n"
        f"Published: {published}\n"
        f"Title: {item.title}\n\n"
        f"Article text:\n{item.body[:6000]}"
    )

    parsed = client.complete_json(
        system=CLASSIFY_SYSTEM_PROMPT,
        user=user_content,
        max_tokens=700,
        temperature=0.1,
    )

    if parsed is None:
        if errors is not None:
            errors.append(f"classify failed for {item.url}")
        return None

    try:
        category = str(parsed.get("category", "other")).lower()
        if category not in VALID_CATEGORIES:
            category = "other"

        # Models sometimes repeat an entity. Dedupe on (name, type) or the
        # event_entities insert will violate its primary key.
        entities: list[dict[str, str]] = []
        seen_entities: set[tuple[str, str]] = set()
        for raw in parsed.get("entities", []):
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            name = str(raw["name"]).strip()
            etype = str(raw.get("type", "technology")).strip().lower()
            if not name:
                continue
            key = (name.lower(), etype)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            entities.append({"name": name, "type": etype})
            if len(entities) == 6:
                break

        importance = int(parsed.get("importance", 0))
        importance = max(0, min(100, importance))

        return Classification(
            content_type=str(parsed.get("content_type", "other")).strip().lower(),
            category=category,
            event_type=str(parsed.get("event_type", "other")).lower(),
            entities=entities,
            importance=importance,
            what_happened=str(parsed.get("what_happened", "")).strip(),
            reason=str(parsed.get("reason", "")).strip(),
        )
    except (TypeError, ValueError, KeyError) as exc:
        if errors is not None:
            errors.append(f"malformed classification for {item.url}: {exc}")
        return None


# ----------------------------------------------------------------------------
# Stage 5 - Clustering
# ----------------------------------------------------------------------------

def normalise_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    stop = {"the", "a", "an", "is", "are", "to", "for", "of", "in", "on",
            "and", "its", "it", "with", "now", "new"}
    words = [w for w in title.split() if w and w not in stop]
    return " ".join(words)


def title_similarity(a: str, b: str) -> float:
    """Jaccard overlap on normalised word sets. Cheap and adequate."""
    set_a = set(normalise_title(a).split())
    set_b = set(normalise_title(b).split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def find_matching_event(
    item: RawItem,
    cls: Classification,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Match against events already built in this run.

    Category is deliberately NOT required. The same story gets filed
    differently by different outlets (Android Police says 'android',
    The Verge says 'technology'), and requiring a category match was
    suppressing almost all real cross-outlet merges.

    Gate is: shared entity + 72h window + title overlap, with the
    similarity bar lowered when two or more entities match.

    Known limitation: headlines sharing an entity but no vocabulary
    ("Critical OpenSSH vulnerability" vs "regreSSHion flaw lets
    attackers run code as root") will not merge. That needs embeddings.
    """
    if not item.published_at:
        return None

    item_entities = {e["name"].lower() for e in cls.entities}
    if not item_entities:
        return None

    best: dict[str, Any] | None = None
    best_score = 0.0

    for event in events:
        gap = abs((item.published_at - event["first_seen_at"]).total_seconds())
        if gap > 72 * 3600:
            continue

        overlap = item_entities & event["entity_names"]
        if not overlap:
            continue

        score = title_similarity(item.title, event["title"])

        # Two shared entities is strong evidence; accept weaker titles.
        threshold = 0.20 if len(overlap) >= 2 else 0.25
        if score >= threshold and score > best_score:
            best = event
            best_score = score

    return best


# ----------------------------------------------------------------------------
# Stage 6 - Persistence
# ----------------------------------------------------------------------------


def upsert_sources(supabase, sources: list[Source]) -> None:
    """Ensure every configured source exists in the DB; store its id."""
    existing = supabase.table("sources").select("id,name,feed_url").execute().data
    by_feed = {row["feed_url"]: row["id"] for row in existing if row.get("feed_url")}

    for source in sources:
        if source.feed_url in by_feed:
            source.db_id = by_feed[source.feed_url]
            continue
        inserted = supabase.table("sources").insert({
            "name": source.name,
            "domain": source.domain,
            "feed_url": source.feed_url,
            "category": source.category,
            "source_type": source.source_type,
            "reliability_score": source.reliability_score,
        }).execute()
        source.db_id = inserted.data[0]["id"]


def load_known_urls(supabase) -> set[str]:
    """Every canonical_url already stored, so re-runs stay cheap."""
    known: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        rows = (
            supabase.table("articles")
            .select("canonical_url")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        if not rows:
            break
        known.update(row["canonical_url"] for row in rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return known


def upsert_entity(supabase, cache: dict[tuple[str, str], str],
                  name: str, entity_type: str) -> str:
    key = (name.lower(), entity_type)
    if key in cache:
        return cache[key]

    found = (
        supabase.table("entities")
        .select("id")
        .eq("canonical_name", name)
        .eq("entity_type", entity_type)
        .execute()
        .data
    )
    if found:
        cache[key] = found[0]["id"]
        return cache[key]

    inserted = supabase.table("entities").insert({
        "canonical_name": name,
        "entity_type": entity_type,
    }).execute()
    cache[key] = inserted.data[0]["id"]
    return cache[key]



def load_sources(path: str) -> list[Source]:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    return [Source(**entry) for entry in config["sources"]]



def fetch_recent(source: Source, cutoff: datetime) -> list[RawItem]:
    """Live feed only. The daily run never needs archive pagination."""
    resp = http_get(source.feed_url)
    if resp is None:
        return []
    return [
        item for item in parse_feed_entries(source, resp.text)
        if item.published_at is None or item.published_at >= cutoff
    ]


def prepare_items(items: list[RawItem], known_urls: set[str]) -> list[RawItem]:
    """Stages 2-3: canonicalise, dedupe within batch and against DB, drop junk."""
    seen: set[str] = set()
    survivors: list[RawItem] = []
    for item in items:
        item.canonical_url = canonicalise(item.url)
        if item.canonical_url in seen or item.canonical_url in known_urls:
            continue
        seen.add(item.canonical_url)
        if looks_like_junk(item):
            continue
        survivors.append(item)
    return survivors


def extract_all(items: list[RawItem], progress_every: int = 25) -> tuple[list[RawItem], int]:
    """Stage 4. Returns (usable items, failure count)."""
    usable: list[RawItem] = []
    failed = 0
    for index, item in enumerate(items, 1):
        extract_body(item)
        item.content_hash = hashlib.sha256(item.body.encode()).hexdigest()
        if item.extraction_status == "metadata_only":
            failed += 1
            continue
        usable.append(item)
        if progress_every and index % progress_every == 0:
            print(f"    {index}/{len(items)}")
    return usable, failed


def is_publishable(cls: Classification, min_importance: int = 40) -> bool:
    """Stage 5 gate: real news, and important enough to bother with."""
    return (
        cls.content_type in NEWS_CONTENT_TYPES
        and cls.importance >= min_importance
    )