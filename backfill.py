#!/usr/bin/env python3
"""
One-time historical backfill.

Reaches back ~12 months across the configured feeds, extracts article text,
classifies each article with an LLM, groups them into events, and stores the
result in Supabase so that day-one digests have real historical context.

Backfilled events are stored with origin='backfill' and first_seen_at set to
the ORIGINAL publication date, not today.

Usage:
    python backfill.py --dry-run                 # no LLM calls, no writes
    python backfill.py --dry-run --limit 20      # tiny sample
    python backfill.py --months 12 --limit 200   # real run, capped
    python backfill.py --months 12               # full run
"""

import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import feedparser
import requests
import trafilatura
import yaml
from dotenv import load_dotenv
from supabase import create_client

from llm import LLMClient, Usage

load_dotenv()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

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
    r"\bsponsored\b", r"\bdeal[s]?\b", r"\bcoupon\b", r"\bdiscount\b",
    r"\bgiveaway\b", r"\bwe'?re hiring\b", r"\bjob[s]? at\b",
    r"\bblack friday\b", r"\bcyber monday\b", r"\bbest .* deals\b",
    r"\bnewsletter\b", r"\bweekly roundup\b", r"\bwhat we'?re reading\b",
]

JUNK_URL_PATTERNS = [
    r"/deals?/", r"/sponsored/", r"/jobs?/", r"/careers?/",
    r"/advertis", r"/newsletter", r"/tag/", r"/author/", r"/page/",
]


# ----------------------------------------------------------------------------
# Data shapes
# ----------------------------------------------------------------------------

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
    is_relevant: bool
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
            title=title.strip(),
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
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;?", "&", text)
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

CLASSIFY_SYSTEM_PROMPT = """You classify technology news articles for a personal
news-context database. You will be given one article. Respond with a single JSON
object and nothing else. No markdown fences, no commentary.

Schema:
{
  "is_relevant": boolean,
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

Rules:
- is_relevant is false for opinion columns, listicles, deals, tutorials,
  job posts, and anything not reporting a concrete event.
- importance reflects lasting significance to a working software developer,
  not click appeal. A routine point release is 20-40. A major platform
  release or serious vulnerability is 70-90.
- entities: at most 6, use canonical names ("Google" not "google inc").
- what_happened must only contain facts present in the supplied text.
"""


VALID_CATEGORIES = {
    "android", "kotlin", "artificial-intelligence", "developer-tools",
    "open-source", "cybersecurity", "hardware", "technology", "other",
}


def classify(item: RawItem, client: LLMClient, stats: Stats) -> Classification | None:
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
        stats.errors.append(f"classify failed for {item.url}")
        return None

    try:
        category = str(parsed.get("category", "other")).lower()
        if category not in VALID_CATEGORIES:
            category = "other"

        entities = [
            {"name": str(e["name"]).strip(),
             "type": str(e.get("type", "technology")).lower()}
            for e in parsed.get("entities", [])
            if isinstance(e, dict) and e.get("name")
        ][:6]

        importance = int(parsed.get("importance", 0))
        importance = max(0, min(100, importance))

        return Classification(
            is_relevant=bool(parsed.get("is_relevant", False)),
            category=category,
            event_type=str(parsed.get("event_type", "other")).lower(),
            entities=entities,
            importance=importance,
            what_happened=str(parsed.get("what_happened", "")).strip(),
            reason=str(parsed.get("reason", "")).strip(),
        )
    except (TypeError, ValueError, KeyError) as exc:
        stats.errors.append(f"malformed classification for {item.url}: {exc}")
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
    Requires shared entity + close publication date + title overlap.
    """
    if not item.published_at:
        return None

    item_entities = {e["name"].lower() for e in cls.entities}

    for event in events:
        if event["category"] != cls.category:
            continue
        gap = abs((item.published_at - event["first_seen_at"]).total_seconds())
        if gap > 72 * 3600:
            continue
        if not (item_entities & event["entity_names"]):
            continue
        # 0.28 is deliberately loose: entity match + category + 72h window
        # already do most of the filtering. Measured separation on real
        # headlines was ~0.33-0.40 for true matches, 0.00 for unrelated.
        if title_similarity(item.title, event["title"]) >= 0.28:
            return event

    return None


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


def persist_events(supabase, events: list[dict[str, Any]], stats: Stats,
                   model_label: str) -> None:
    entity_cache: dict[tuple[str, str], str] = {}

    for event in events:
        try:
            article_ids: list[str] = []
            for item in event["items"]:
                inserted = supabase.table("articles").insert({
                    "source_id": item.source.db_id,
                    "canonical_url": item.canonical_url,
                    "title": item.title[:500],
                    "description": item.excerpt[:2000] or None,
                    "extracted_text": item.body[:50000],
                    "content_hash": item.content_hash,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "extraction_status": item.extraction_status,
                    "raw_metadata": {"archive_strategy": item.source.archive_strategy},
                }).execute()
                article_ids.append(inserted.data[0]["id"])

            event_row = supabase.table("events").insert({
                "title": event["title"][:500],
                "category": event["category"],
                "event_type": event["event_type"],
                "what_happened": event["what_happened"],
                "background": None,
                "what_changed": [],
                "why_it_matters": [],
                "uncertainties": [],
                "timeline": [],
                "importance_score": event["importance"],
                "origin": "backfill",
                "first_seen_at": event["first_seen_at"].isoformat(),
                "generation_metadata": {
                    "model": model_label,
                    "backfilled_at": datetime.now(timezone.utc).isoformat(),
                    "article_count": len(event["items"]),
                },
            }).execute()
            event_id = event_row.data[0]["id"]

            for index, article_id in enumerate(article_ids):
                supabase.table("event_articles").insert({
                    "event_id": event_id,
                    "article_id": article_id,
                    "source_role": "primary" if index == 0 else "secondary",
                }).execute()

            for entity in event["entities"]:
                entity_id = upsert_entity(
                    supabase, entity_cache,
                    entity["name"], entity.get("type", "technology"),
                )
                supabase.table("event_entities").insert({
                    "event_id": event_id,
                    "entity_id": entity_id,
                }).execute()

            stats.events_created += 1

        except Exception as exc:
            stats.errors.append(f"persist failed for '{event['title'][:60]}': {exc}")


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def load_sources(path: str) -> list[Source]:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    return [Source(**entry) for entry in config["sources"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical backfill")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0,
                        help="max articles to classify (0 = no cap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and extract only; no LLM calls, no writes")
    parser.add_argument("--config", default="config/sources.yaml")
    args = parser.parse_args()

    stats = Stats()
    usage = Usage()
    classify_client: LLMClient | None = None
    cutoff = datetime.now(timezone.utc) - timedelta(days=30 * args.months)
    sources = load_sources(args.config)

    print(f"\nBackfill: {len(sources)} sources, cutoff {cutoff.date()}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    supabase = None
    known_urls: set[str] = set()

    if not args.dry_run:
        for var in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
            if not os.environ.get(var):
                print(f"Missing environment variable: {var}")
                return 1

        try:
            classify_client = LLMClient("classify", usage)
        except Exception as exc:
            print(f"LLM provider not configured: {exc}")
            return 1

        supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
        upsert_sources(supabase, sources)
        known_urls = load_known_urls(supabase)
        print(f"Already stored: {len(known_urls)} articles\n")

    # --- Stage 1: fetch ---
    all_items: list[RawItem] = []
    for source in sources:
        print(f"  {source.name} ...", end=" ", flush=True)
        items = fetch_archive(source, cutoff)
        print(f"{len(items)} items")
        all_items.extend(items)
        stats.fetched += len(items)

    print(f"\nFetched {stats.fetched} raw items")

    # --- Stage 2: canonicalise, dedupe, junk filter ---
    seen: set[str] = set()
    survivors: list[RawItem] = []
    for item in all_items:
        item.canonical_url = canonicalise(item.url)
        if item.canonical_url in seen or item.canonical_url in known_urls:
            continue
        seen.add(item.canonical_url)
        stats.after_dedup += 1

        reason = looks_like_junk(item)
        if reason:
            continue
        survivors.append(item)

    stats.after_junk_filter = len(survivors)
    print(f"After dedup: {stats.after_dedup}")
    print(f"After junk filter: {stats.after_junk_filter}")

    survivors.sort(key=lambda i: i.published_at or datetime.min.replace(tzinfo=timezone.utc))
    if args.limit:
        survivors = survivors[:args.limit]
        print(f"Capped to {len(survivors)} by --limit")

    # --- Stage 3: extract ---
    print(f"\nExtracting {len(survivors)} articles ...")
    extracted: list[RawItem] = []
    for index, item in enumerate(survivors, 1):
        extract_body(item)
        item.content_hash = hashlib.sha256(item.body.encode()).hexdigest()
        if item.extraction_status == "metadata_only":
            stats.extraction_failed += 1
            continue
        extracted.append(item)
        stats.extracted += 1
        if index % 25 == 0:
            print(f"  {index}/{len(survivors)}")

    print(f"Extracted {stats.extracted}, failed {stats.extraction_failed}")

    if args.dry_run:
        print("\n--- DRY RUN: sample of what would be classified ---")
        for item in extracted[:5]:
            date = item.published_at.date() if item.published_at else "?"
            print(f"\n  [{date}] {item.source.name}")
            print(f"  {item.title}")
            print(f"  {item.canonical_url}")
            print(f"  {item.extraction_status}, {len(item.body)} chars")
            print(f"  {item.body[:200]}...")
        est_in = stats.extracted * 1800
        est_out = stats.extracted * 200
        cost = est_in / 1e6 * 0.14 + est_out / 1e6 * 0.28
        print(f"\nEstimated classification cost for {stats.extracted} articles: ${cost:.2f}")
        print("No LLM calls made. No data written.")
        return 0

    # --- Stage 4: classify ---
    print(f"\nClassifying {len(extracted)} articles ...")
    print(f"  provider: {classify_client.describe()}")
    keepers: list[tuple[RawItem, Classification]] = []

    for index, item in enumerate(extracted, 1):
        cls = classify(item, classify_client, stats)
        if cls is None:
            continue
        stats.classified += 1

        if not cls.is_relevant:
            stats.rejected_irrelevant += 1
            continue
        if cls.importance < MIN_IMPORTANCE:
            stats.rejected_low_importance += 1
            continue

        keepers.append((item, cls))

        if index % 20 == 0:
            print(f"  {index}/{len(extracted)}  kept {len(keepers)}  "
                  f"${usage.total_usd:.4f}")

    print(f"Kept {len(keepers)} relevant articles")

    # --- Stage 5: cluster ---
    events: list[dict[str, Any]] = []
    for item, cls in keepers:
        match = find_matching_event(item, cls, events)
        if match:
            match["items"].append(item)
            match["entity_names"].update(e["name"].lower() for e in cls.entities)
            match["importance"] = max(match["importance"], cls.importance)
            stats.events_merged += 1
            continue

        events.append({
            "title": item.title,
            "category": cls.category,
            "event_type": cls.event_type,
            "what_happened": cls.what_happened,
            "importance": cls.importance,
            "entities": cls.entities,
            "entity_names": {e["name"].lower() for e in cls.entities},
            "first_seen_at": item.published_at or datetime.now(timezone.utc),
            "items": [item],
        })

    print(f"Clustered into {len(events)} events ({stats.events_merged} merges)")

    # --- Stage 6: persist ---
    print(f"\nWriting to Supabase ...")
    persist_events(supabase, events, stats, classify_client.describe())

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Fetched:              {stats.fetched}")
    print(f"  After dedup:          {stats.after_dedup}")
    print(f"  Extracted:            {stats.extracted}")
    print(f"  Extraction failures:  {stats.extraction_failed}")
    print(f"  Classified:           {stats.classified}")
    print(f"  Rejected irrelevant:  {stats.rejected_irrelevant}")
    print(f"  Rejected low score:   {stats.rejected_low_importance}")
    print(f"  Events created:       {stats.events_created}")
    print(f"  Articles merged:      {stats.events_merged}")
    print(f"  LLM usage:")
    print("    " + usage.summary().replace("\n", "\n    "))

    if stats.errors:
        print(f"\n  Errors ({len(stats.errors)}):")
        for error in stats.errors[:15]:
            print(f"    - {error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())