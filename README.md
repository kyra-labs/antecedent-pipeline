<h3 align="center">Antecedent Pipeline</h3>
<p align="center">
  The daily engine behind <a href="https://antecedent.kyralabs.dev/"><strong>Antecedent</strong></a> — it doesn't just collect tech news, it figures out how each story got here.
</p>

<p align="center">
  <a href="https://antecedent.kyralabs.dev/">See it in action →</a>
  ·
  <a href="https://github.com/kyra-labs/antecedent-web-app">Web app repo</a>
  ·
  <a href="https://github.com/kyra-labs">Kyra Labs</a>
</p>

---

## What this is

Every morning, this pipeline runs as a scheduled GitHub Action, pulls the last day's tech news, and — instead of just summarizing what happened — traces each story back through everything it has previously archived about the same entities (companies, people, technologies). The output is a **digest**: a ranked set of stories, each with its own historical background, timeline, and "why it matters," written by an LLM but grounded entirely in structured data the pipeline itself has been accumulating.

The [Antecedent web app](https://github.com/kyra-labs/antecedent-web-app) reads what this pipeline writes to Supabase. This repo has no UI — it's the fetch → understand → contextualize → store engine underneath it.

---

## Pipeline flow

One run works through twelve stages, defined in `common.py`, `history.py`, and `main.py`:

```
fetch → canonicalise → filter → extract → classify → cluster
      → retrieve history → brief → validate
      → rank → assemble digest → store → notify
```

### Stages 1–6 — ingest and understand (`common.py`, shared by every run)

| Stage | What happens |
|---|---|
| **1. Fetch** | Pulls ~20–30 RSS feeds (`feedparser`) plus the Hacker News API (top 100 stories, score ≥ 100) |
| **2. Canonicalise** | Strips tracking params and normalizes URLs so the same article from two different links doesn't get treated as two stories |
| **3. Filter** | Two-layer relevance filtering — ~40 deterministic regex patterns catch obvious junk (first-person posts, reviews, guides, listicles) *before* spending a single LLM token, then an LLM pass classifies the survivors with a `content_type` label rather than a blunt yes/no |
| **4. Extract** | Fetches each article's own HTML and pulls the readable body text with Trafilatura, stripping nav/ads/boilerplate |
| **5. Classify** | LLM assigns category, importance score, and source-quality score to each surviving article |
| **6. Cluster** | Deduplicates near-identical stories across outlets by matching on shared specific entities weighted against title similarity — not by requiring the same category, which used to miss real cross-outlet duplicates |

### Stages 7–9 — add the history (`history.py`)

| Stage | What happens |
|---|---|
| **7. Retrieve history** | For every surviving event, runs a structured, entity-based SQL query against the archive (GraphRAG-style — no vector embeddings) to pull everything previously stored about the same entities |
| **8. Brief** | An LLM writes the actual digest content for the top-ranked events: a summary, a "how we got here" background section, a dated timeline, "what changed" / "why it matters" bullets, and open questions when they exist |
| **9. Validate** | Checks the generated briefing against the retrieved timeline data before it's allowed to ship, protecting date integrity in the historical thread |

### Stages 10–12 — rank, assemble, and ship (`main.py`)

| Stage | What happens |
|---|---|
| **10. Store (archive)** | Every relevant event is archived (`store_event`) regardless of rank — so a story that ranks 9th on a busy news day is never lost, even though it doesn't make that day's published digest |
| **11. Rank** | A transparent, weighted formula scores each event — importance (35%), category interest (30%), source quality (15%), novelty (10%), and historical context value (10%) — plus fixed bonuses/penalties for categories marked `always_surface` (e.g. vulnerabilities) or `deprioritise` (e.g. funding news). It's deliberately not an LLM judgment call: any score is explainable after the fact |
| **12. Assemble & publish** | The top-ranked slice is written as that day's digest (`store_run`), with an LLM-written intro paragraph tying the day's stories together |
| **Notify** | Push notification via Firebase Cloud Messaging *(planned, not yet live)* |

---

## The cold-start problem — `backfill.py`

RSS feeds only ever show the last 10–50 posts, and the HN API only surfaces what's currently trending — neither gives a pipeline any *history* to reference on day one. `backfill.py` solves this with a one-time 12-month historical seed, reusing the same stages 1–6 as the daily run against a much wider time window. It's already been run successfully: roughly 745 events archived for about $0.21 in total LLM cost.

---

## Why DB-only, structured retrieval instead of vector search

This is a deliberate architecture choice, not a limitation to route around:

- **Date integrity** — a vector similarity match can surface something *topically* related but chronologically wrong; a structured SQL query filtered by entity and date can't
- **Cost** — no embedding model, no vector DB, no re-indexing
- **A self-compounding archive** — every day's run makes tomorrow's historical context slightly richer, since retrieval is just a SQL query over everything stored so far

History retrieval is currently running at roughly 71% of events finding real prior context — comfortably above the ~60% considered a "healthy archive." If that ever drops for good reason (rather than just an early archive), the fallback plan is a small Hetzner VPS running live web search augmentation on top of the DB, not a switch to embeddings.

---

## Automation — GitHub Actions

`.github/workflows/daily-digest.yml` runs the pipeline on a cron schedule:

- **`37 0 * * *`** (00:37 UTC / 06:07 IST) — a deliberately off-the-hour minute, since round times queue behind the entire shared runner fleet
- **`workflow_dispatch`** with a `dry_run` toggle — lets a run be triggered manually from the Actions tab with no writes and no LLM briefing calls, just a cost/count estimate
- A `digest.md` artifact is uploaded and kept for 30 days, so output quality can be reviewed without opening the database
- Runs on a 45-minute timeout with `concurrency` set to prevent two runs overlapping

**Operational gotcha worth knowing:** GitHub automatically disables scheduled workflows on public repos after 60 days with no repository activity, and only new *commits* reset that clock — a schedule alone doesn't count.

### CLI

```bash
python main.py --dry-run              # no writes, no LLM briefings, cost estimate only
python main.py --dry-run --verbose
python main.py                        # real run
python main.py --markdown digest.md   # real run + a local readable copy
```

---

## Cost discipline

The entire pipeline runs on a hard ceiling of **under ₹120/month**, currently averaging around **₹32/month**:

| Component | Choice | Why |
|---|---|---|
| Compute | GitHub Actions (free tier, public repo) | No server to maintain, minutes are effectively free at this run frequency |
| LLM | MiMo V2.5, pay-as-you-go OpenAI-compatible endpoint | Cheapest reliable option found; DeepSeek is kept as a fallback `base_url` in `llm.py` |
| Storage | Supabase Postgres, free tier | Structured, relational — fits the entity/SQL retrieval model directly |
| Ingestion | RSS + Hacker News API | Both free, no keys required |

---

## Tech stack

- **Python 3.12**
- `feedparser` — RSS parsing
- `requests` — HTTP fetching (article HTML, HN API)
- `trafilatura` — full-text article extraction
- `supabase-py` — Postgres client
- `PyYAML` — source list and category config
- `python-dotenv` — local env var loading
- GitHub Actions — scheduler + runner

---

## Project structure

```
.
├── main.py                  # daily orchestrator — stages 10-12, wires everything together
├── backfill.py                # one-time 12-month historical seed (stages 1-6, wider window)
├── common.py                  # shared stages 1-6: fetch, canonicalise, filter, extract, classify, cluster
├── history.py                  # stages 7-9: retrieve history, brief, validate
├── llm.py                      # provider-agnostic LLM client (MiMo primary, DeepSeek fallback)
├── config/
│   ├── sources.yaml              # RSS feed list
│   └── categories.yaml            # interests, always_surface, deprioritise weighting
├── .github/
│   └── workflows/
│       └── daily-digest.yml        # cron + manual trigger
├── requirements.txt
└── .env.example
```

---

## Running locally

```bash
git clone https://github.com/kyra-labs/antecedent-pipeline.git
cd antecedent-pipeline
pip install -r requirements.txt
cp .env.example .env   # fill in Supabase + LLM credentials
```

```bash
python main.py --dry-run --verbose   # safe first run — no writes, no LLM spend on briefings
python main.py --markdown digest.md  # a real run, with a local copy to read
```

Required environment variables (also set as GitHub Actions secrets/variables for the scheduled run): `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, LLM provider config (`MIMO_*` / `DEEPSEEK_*`), and — once notifications ship — `FIREBASE_SERVICE_ACCOUNT_JSON`.

---

## Roadmap

- **Push notifications** via Firebase Cloud Messaging
- **Hetzner CX23 VPS + web search augmentation** — a fallback layer for events where the DB-only archive doesn't yet have enough history, kept separate from the core structured-retrieval model rather than replacing it

---

## Related

- [**antecedent-web-app**](https://github.com/kyra-labs/antecedent-web-app) — the React frontend that reads this pipeline's output
- [**Kyra Labs**](https://github.com/kyra-labs) — the org this project lives under

## License

MIT — see [LICENSE](./LICENSE) for details.
