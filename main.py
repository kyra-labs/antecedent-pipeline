#!/usr/bin/env python3
"""
Daily digest pipeline. Runs once each morning in GitHub Actions.

    fetch -> filter -> extract -> classify -> cluster
          -> retrieve history -> brief -> validate -> rank
          -> assemble digest -> store -> notify

Usage:
    python main.py --dry-run          # no writes, no briefings, cost estimate
    python main.py --dry-run --verbose
    python main.py                    # real run
    python main.py --markdown out.md  # also write a local Markdown copy
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

import yaml
from dotenv import load_dotenv
from supabase import create_client

import common
import history as hist
from llm import LLMClient, Usage

load_dotenv()

FETCH_WINDOW_HOURS = 26      # slight overlap so nothing falls between runs
BRIEF_HEADROOM = 3           # brief a few more events than the digest needs


# ---------------------------------------------------------------------------
# Stage 11 - ranking
# ---------------------------------------------------------------------------

def score_event(
    event: dict[str, Any],
    config: dict[str, Any],
    history_count: int,
) -> float:
    """
    Transparent weighted formula. Deliberately not an LLM judgement call:
    you can look at any score and understand why a story ranked where it did.
    """
    interests = config.get("interests", {})

    importance = event["importance"] / 100
    interest = interests.get(event["category"], 0.3)
    source_quality = event["source_quality"] / 100
    novelty = 1.0 if history_count == 0 else max(0.3, 1.0 - history_count * 0.12)
    context_value = min(history_count / 4, 1.0)

    score = (
        importance * 0.35
        + interest * 0.30
        + source_quality * 0.15
        + novelty * 0.10
        + context_value * 0.10
    ) * 100

    if event["event_type"] in config.get("always_surface", []):
        score += 12
    if event["event_type"] in config.get("deprioritise", []):
        score -= 15

    return round(score, 1)


# ---------------------------------------------------------------------------
# Stage 12 - digest introduction
# ---------------------------------------------------------------------------

INTRO_SYSTEM_PROMPT = """You write the opening of a personal technology news
digest. You are given the briefings that will appear in it. Respond with a
single JSON object and nothing else.

{
  "title": string, under 60 characters, naming the day's dominant theme,
  "introduction": string, exactly two sentences summarising the day,
  "reading_minutes": integer
}

Do not restate every story. Do not use "In today's digest" or similar
throwaway openings. Name the actual subject matter. If the day has no
coherent theme, say so plainly rather than inventing one.
"""


def generate_intro(client: LLMClient, briefs: list[dict[str, Any]]) -> dict[str, Any]:
    lines = [
        f"- {b['brief']['headline']} ({b['category']}, score {b['final_score']})"
        for b in briefs
    ]
    result = client.complete_json(
        system=INTRO_SYSTEM_PROMPT,
        user="Stories in today's digest:\n" + "\n".join(lines),
        max_tokens=400,
        temperature=0.4,
    )
    if not result:
        return {
            "title": f"Tech context for {date.today():%d %B}",
            "introduction": f"{len(briefs)} stories from the last 24 hours.",
            "reading_minutes": max(3, len(briefs) * 2),
        }
    return result


# ---------------------------------------------------------------------------
# Stage 13 - storage
# ---------------------------------------------------------------------------

def store_event(
    supabase,
    event: dict[str, Any],
    entity_cache: dict[tuple[str, str], str],
    errors: list[str],
) -> str | None:
    """
    Write one event and its articles, sources and entity links.

    If the event was briefed, the full briefing is stored. If it was only
    classified, the short classification summary is stored instead. Both
    are archived, because an event that is unremarkable today may be
    essential context six months from now.
    """
    brief = event.get("brief")

    try:
        article_ids: list[str] = []
        for raw in event["items"]:
            inserted = (
                supabase.table("articles")
                .insert(
                    {
                        "source_id": raw.source.db_id,
                        "canonical_url": raw.canonical_url,
                        "title": raw.title[:500],
                        "description": raw.excerpt[:2000] or None,
                        "extracted_text": raw.body[:50000],
                        "content_hash": raw.content_hash,
                        "published_at": raw.published_at.isoformat()
                        if raw.published_at
                        else None,
                        "extraction_status": raw.extraction_status,
                        "raw_metadata": {},
                    }
                )
                .execute()
            )
            article_ids.append(inserted.data[0]["id"])

        row = {
            "title": (brief["headline"] if brief else event["title"])[:500],
            "category": event["category"],
            "event_type": event["event_type"],
            "what_happened": (
                brief.get("what_happened") if brief else event.get("what_happened", "")
            ),
            "background": (brief.get("background") or None) if brief else None,
            "what_changed": brief.get("what_changed", []) if brief else [],
            "why_it_matters": brief.get("why_it_matters", []) if brief else [],
            "uncertainties": brief.get("uncertainties", []) if brief else [],
            "timeline": brief.get("timeline", []) if brief else [],
            "importance_score": event["importance"],
            "origin": "observed",
            "first_seen_at": event["first_seen_at"].isoformat(),
            "generation_metadata": {
                "final_score": event.get("final_score"),
                "history_used": event.get("history_count", 0),
                "article_count": len(event["items"]),
                "briefed": bool(brief),
            },
        }
        event_id = supabase.table("events").insert(row).execute().data[0]["id"]

        for index, article_id in enumerate(article_ids):
            supabase.table("event_articles").insert(
                {
                    "event_id": event_id,
                    "article_id": article_id,
                    "source_role": "primary" if index == 0 else "secondary",
                }
            ).execute()

        linked: set[str] = set()
        for entity in event["entities"]:
            entity_id = common.upsert_entity(
                supabase, entity_cache, entity["name"],
                entity.get("type", "technology"),
            )
            if entity_id in linked:
                continue
            linked.add(entity_id)
            supabase.table("event_entities").insert(
                {"event_id": event_id, "entity_id": entity_id}
            ).execute()

        return event_id

    except Exception as exc:
        label = (brief or {}).get("headline") or event.get("title", "?")
        errors.append(f"store failed for {label[:50]}: {exc}")
        return None


def store_run(
    supabase,
    briefed: list[dict[str, Any]],
    archived: list[dict[str, Any]],
    intro: dict[str, Any],
    config: dict[str, Any],
    errors: list[str],
) -> tuple[str | None, int]:
    """
    Archive every relevant event from this run, then publish the top slice
    as today's digest.

    The digest is a VIEW over the archive, not the archive itself. Storage
    is deliberately more generous than publication.
    """
    today = date.today().isoformat()

    existing = (
        supabase.table("digests").select("id").eq("digest_date", today).execute().data
    )
    if existing:
        digest_id = existing[0]["id"]
        supabase.table("digests").update({"status": "processing"}).eq(
            "id", digest_id
        ).execute()
        supabase.table("digest_items").delete().eq("digest_id", digest_id).execute()
    else:
        digest_id = (
            supabase.table("digests")
            .insert(
                {
                    "digest_date": today,
                    "title": intro.get("title"),
                    "introduction": intro.get("introduction"),
                    "reading_minutes": intro.get("reading_minutes"),
                    "status": "processing",
                }
            )
            .execute()
            .data[0]["id"]
        )

    entity_cache: dict[tuple[str, str], str] = {}
    must_know = config.get("must_know", 3)
    published = 0

    # Published stories first, so a mid-run failure loses archive rows
    # rather than the digest itself.
    for rank, event in enumerate(briefed, 1):
        event_id = store_event(supabase, event, entity_cache, errors)
        if not event_id:
            continue
        supabase.table("digest_items").insert(
            {
                "digest_id": digest_id,
                "event_id": event_id,
                "rank": rank,
                "section": "must_know" if rank <= must_know else "also",
            }
        ).execute()
        published += 1

    archived_count = 0
    for event in archived:
        if store_event(supabase, event, entity_cache, errors):
            archived_count += 1

    if not published:
        supabase.table("digests").update({"status": "failed"}).eq(
            "id", digest_id
        ).execute()
        errors.append("no stories published; digest marked failed")
        return None, archived_count

    supabase.table("digests").update(
        {
            "status": "ready",
            "title": intro.get("title"),
            "introduction": intro.get("introduction"),
            "reading_minutes": intro.get("reading_minutes"),
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", digest_id).execute()

    return digest_id, archived_count


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_markdown(intro: dict[str, Any], briefs: list[dict[str, Any]],
                    config: dict[str, Any]) -> str:
    must_know = config.get("must_know", 3)
    out = [
        f"# {intro.get('title', 'Tech context')}",
        f"\n_{date.today():%A, %d %B %Y} · {intro.get('reading_minutes', '?')} min read_\n",
        intro.get("introduction", ""),
        "",
    ]

    for index, item in enumerate(briefs, 1):
        brief = item["brief"]
        if index == 1:
            out.append("\n## Must know\n")
        elif index == must_know + 1:
            out.append("\n## Also worth knowing\n")

        out.append(f"### {brief['headline']}")
        out.append(
            f"`{item['category']}` · score {item['final_score']} · "
            f"{item['history_count']} prior events\n"
        )
        out.append(brief.get("what_happened", ""))

        if brief.get("background"):
            out.append(f"\n**How we got here**\n\n{brief['background']}")

        if brief.get("what_changed"):
            out.append("\n**What changed**\n")
            out.extend(f"- {c}" for c in brief["what_changed"])

        if brief.get("why_it_matters"):
            out.append("\n**Why it matters**\n")
            out.extend(f"- {w}" for w in brief["why_it_matters"])

        if brief.get("timeline"):
            out.append("\n**Timeline**\n")
            out.extend(f"- `{e['date']}` {e['text']}" for e in brief["timeline"])

        if brief.get("uncertainties"):
            out.append("\n**Not yet known**\n")
            out.extend(f"- {u}" for u in brief["uncertainties"])

        out.append("\n**Sources**\n")
        out.extend(f"- [{i.source.name}]({i.canonical_url})" for i in item["items"])
        out.append("\n---\n")

    return "\n".join(out)


def send_notification(intro: dict[str, Any], count: int, errors: list[str]) -> None:
    """FCM push. Silently skipped when no credentials are configured."""
    creds = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    topic = os.environ.get("FCM_TOPIC", "digest")
    if not creds:
        return
    try:
        import json
        import firebase_admin
        from firebase_admin import credentials, messaging

        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                credentials.Certificate(json.loads(creds))
            )
        messaging.send(
            messaging.Message(
                notification=messaging.Notification(
                    title=intro.get("title", "Today's digest is ready"),
                    body=f"{count} stories · {intro.get('reading_minutes', '?')} min",
                ),
                topic=topic,
            )
        )
    except Exception as exc:
        errors.append(f"notification failed: {exc}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Daily digest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--markdown", metavar="PATH")
    parser.add_argument("--sources", default="config/sources.yaml")
    parser.add_argument("--categories", default="config/categories.yaml")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    errors: list[str] = []
    usage = Usage()

    with open(args.categories) as handle:
        config = yaml.safe_load(handle)
    sources = common.load_sources(args.sources)
    cutoff = started - timedelta(hours=FETCH_WINDOW_HOURS)

    print(f"\nDaily digest · {started:%Y-%m-%d %H:%M} UTC")
    print(f"Window: last {FETCH_WINDOW_HOURS}h · {len(sources)} sources")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    for var in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
        if not os.environ.get(var):
            print(f"Missing environment variable: {var}")
            return 1

    supabase = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )
    common.upsert_sources(supabase, sources)
    known_urls = common.load_known_urls(supabase)

    try:
        classify_client = LLMClient("classify", usage)
        brief_client = LLMClient("brief", usage)
    except Exception as exc:
        print(f"LLM provider not configured: {exc}")
        return 1

    # -- Stage 1: fetch ----------------------------------------------------
    raw: list[common.RawItem] = []
    for source in sources:
        items = common.fetch_recent(source, cutoff)
        raw.extend(items)
        if args.verbose:
            print(f"  {source.name}: {len(items)}")
    print(f"Fetched {len(raw)} items")

    # -- Stages 2-3: filter ------------------------------------------------
    survivors = common.prepare_items(raw, known_urls)
    print(f"After dedup and junk filter: {len(survivors)}")

    if not survivors:
        print("\nNothing new since the last run. No digest generated.")
        return 0

    # -- Stage 4: extract --------------------------------------------------
    extracted, failed = common.extract_all(survivors, progress_every=0)
    print(f"Extracted {len(extracted)} ({failed} failed)")

    # -- Stage 5: classify -------------------------------------------------
    keepers: list[tuple[common.RawItem, common.Classification]] = []
    dropped: dict[str, int] = {}
    for item in extracted:
        cls = common.classify(item, classify_client, errors)
        if cls is None:
            dropped["classify_failed"] = dropped.get("classify_failed", 0) + 1
            continue
        if common.is_publishable(cls):
            keepers.append((item, cls))
            continue
        reason = (
            cls.content_type
            if cls.content_type not in common.NEWS_CONTENT_TYPES
            else f"importance<{40}"
        )
        dropped[reason] = dropped.get(reason, 0) + 1
        if args.verbose:
            print(f"    drop [{reason}] {item.title[:58]}")

    print(f"News events: {len(keepers)}  (${usage.total_usd:.4f} so far)")
    if dropped:
        summary = ", ".join(f"{k} {v}" for k, v in sorted(dropped.items()))
        print(f"  dropped: {summary}")

    if not keepers:
        print("\nNothing relevant today. No digest generated.")
        return 0

    # -- Stage 6: cluster --------------------------------------------------
    events: list[dict[str, Any]] = []
    for item, cls in keepers:
        match = common.find_matching_event(item, cls, events)
        if match:
            match["items"].append(item)
            match["entity_names"].update(e["name"].lower() for e in cls.entities)
            match["importance"] = max(match["importance"], cls.importance)
            match["source_quality"] = max(
                match["source_quality"], item.source.reliability_score
            )
            continue
        events.append(
            {
                "title": item.title,
                "category": cls.category,
                "event_type": cls.event_type,
                "what_happened": cls.what_happened,
                "importance": cls.importance,
                "entities": cls.entities,
                "entity_names": {e["name"].lower() for e in cls.entities},
                "source_quality": item.source.reliability_score,
                "first_seen_at": item.published_at or started,
                "items": [item],
            }
        )
    print(f"Clustered into {len(events)} events")

    # -- Stage 7: order sources within each event --------------------------
    for event in events:
        event["items"].sort(key=lambda i: i.source.reliability_score, reverse=True)

    # -- Stage 8: retrieve history ----------------------------------------
    generic_ids = hist.load_generic_entity_ids(supabase)
    if args.verbose:
        print(f"  {len(generic_ids)} entities excluded as too generic")

    for event in events:
        entity_ids = hist.resolve_entity_ids(supabase, event["entities"])
        event["history"] = hist.retrieve_history(
            supabase, entity_ids, generic_ids, exclude_event_ids=set()
        )
        event["history_count"] = len(event["history"])

    with_history = sum(1 for e in events if e["history_count"])
    print(f"History found for {with_history}/{len(events)} events")

    # -- Stage 11 (early): rank, so we only brief what we will publish -----
    for event in events:
        event["final_score"] = score_event(event, config, event["history_count"])

    events.sort(key=lambda e: e["final_score"], reverse=True)
    minimum = config.get("minimum_score", 38)
    shortlist = [e for e in events if e["final_score"] >= minimum]

    capacity = config.get("must_know", 3) + config.get("also_worth_knowing", 5)
    to_brief = shortlist[: capacity + BRIEF_HEADROOM]

    print(f"Above threshold: {len(shortlist)} · briefing {len(to_brief)}")

    if args.dry_run:
        print("\n--- would brief ---")
        for event in to_brief:
            print(
                f"  {event['final_score']:5.1f}  [{event['category']:24s}] "
                f"{event['history_count']} hist  {event['title'][:60]}"
            )
        est = len(to_brief) * (8000 * 0.105 + 800 * 0.28) / 1e6
        print(f"\nSpent so far: ${usage.total_usd:.4f}")
        print(f"Briefings would add roughly: ${est:.4f}")
        print("No briefings generated. Nothing written.")
        return 0

    # -- Stage 9-10: brief and validate -----------------------------------
    briefed: list[dict[str, Any]] = []
    for event in to_brief:
        primary = event["items"][0]
        primary_date = (
            primary.published_at.date().isoformat()
            if primary.published_at
            else started.date().isoformat()
        )
        secondary = [(i.source.name, i.body) for i in event["items"][1:3]]

        brief = hist.generate_briefing(
            brief_client,
            primary.title,
            primary.body,
            primary_date,
            secondary,
            event["history"],
        )
        if brief is None:
            errors.append(f"briefing failed: {event['title'][:50]}")
            continue

        problems = hist.validate_briefing(brief, primary_date, event["history"])
        if problems:
            if args.verbose:
                print(f"  repaired: {event['title'][:45]} -> {problems}")
            brief = hist.repair_briefing(brief, problems, primary_date, event["history"])
            errors.extend(f"validation: {p} ({event['title'][:40]})" for p in problems)

        if not brief.get("headline"):
            brief["headline"] = event["title"]

        event["brief"] = brief
        briefed.append(event)

        if len(briefed) >= capacity:
            break

    if not briefed:
        print("\nNo briefings survived validation. No digest generated.")
        return 1

    print(f"Briefed {len(briefed)} stories (${usage.total_usd:.4f})")

    # -- Stage 12: intro ---------------------------------------------------
    intro = generate_intro(brief_client, briefed)

    # -- Stage 13: store ---------------------------------------------------
    briefed_ids = {id(e) for e in briefed}
    archived = [e for e in events if id(e) not in briefed_ids]

    digest_id, archived_count = store_run(
        supabase, briefed, archived, intro, config, errors
    )
    if digest_id:
        print(f"Published {len(briefed)} · archived {archived_count} more")
    else:
        print("Digest FAILED to store")

    if args.markdown:
        with open(args.markdown, "w") as handle:
            handle.write(render_markdown(intro, briefed, config))
        print(f"Markdown written to {args.markdown}")

    # -- Stage 14: notify --------------------------------------------------
    if digest_id:
        send_notification(intro, len(briefed), errors)

    # -- Summary -----------------------------------------------------------
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print("\n" + "=" * 58)
    print(f"  {intro.get('title', 'Digest')}")
    print("=" * 58)
    print(f"  Stories:   {len(briefed)}")
    print(f"  Runtime:   {elapsed / 60:.1f} min")
    print(f"  LLM usage:")
    print("    " + usage.summary().replace("\n", "\n    "))

    if errors:
        print(f"\n  Warnings ({len(errors)}):")
        for error in errors[:12]:
            print(f"    - {error}")

    return 0 if digest_id else 1


if __name__ == "__main__":
    sys.exit(main())