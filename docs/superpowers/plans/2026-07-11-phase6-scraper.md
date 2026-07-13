# Phase 6: Data Auto-Refresh Implementation Plan

> **PIVOT (2026-07-11):** ufcstats.com fronts all pages with a JavaScript proof-of-work
> anti-bot challenge (verified live — two polite requests, then stopped). Bypassing
> bot-detection is out of bounds, and a scraper against a site actively refusing
> automated clients would be adversarial and fragile anyway. Per user decision, Phase 6
> is now a **weekly Kaggle auto-refresh**: pull the latest version of the same
> maintained ufcstats-mirror dataset via the sanctioned Kaggle API, rebuild all
> artifacts, commit if changed. Tasks 1–3 and 6 below (scraper parsers/orchestrator)
> are RETIRED — superseded by the two tasks in the Addendum at the bottom. The
> original text is retained for the record.

# (Retired) Phase 6: ufcstats Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An incremental, polite ufcstats.com scraper that appends new events to the existing raw CSVs in the exact schema the pipeline already consumes, plus a weekly GitHub Action that refreshes data, rebuilds artifacts, and commits — making the repo and app self-updating.

**Architecture:** `src/mma/scrape_parse.py` (pure HTML→dict parsers, fixture-tested, no network) + `src/mma/scrape_ufcstats.py` (fetch/cache/rate-limit + incremental orchestration writing `data/raw/UFC.csv` and `data/raw/fighter_details.csv` rows) + `scripts/scrape_ufcstats.py` (CLI) + `.github/workflows/refresh-data.yml`.

**Branch:** `phase-6-scraper`. Env notes as before (OMP_NUM_THREADS=1, background long runs, index.lock retries, classifier-outage retries).

**Design decisions (locked):**
- **ID compatibility is the contract.** ufcstats URLs carry the same 16-hex ids the Kaggle dataset uses (`fighter-details/<id>`, `fight-details/<id>`, `event-details/<id>`). The scraper extracts ids from URLs — everything downstream joins natively.
- **Output = the raw-CSV schema, not the processed tables.** The scraper appends rows to `data/raw/UFC.csv` (only the columns `make_dataset.py` consumes: fight_id, event_id, event_name, date, division, title_fight, method, finish_round, match_time_sec [final-round clock — matches the duration derivation], total_rounds, referee, r_id/b_id, r_name/b_name, winner, winner_id, and the per-corner stat columns from `_STAT_COLUMNS`: kd, sig_str_landed/atmpted, total_str_landed/atmpted, td_landed/atmpted, sub_att, ctrl) and upserts `data/raw/fighter_details.csv` (id, name, height/reach in **cm**, weight kg, stance, dob "Mon DD, YYYY" — matching Kaggle's units). Nothing downstream changes.
- **Method-string normalization**: ufcstats abbreviations map to the Kaggle vocabulary before writing: U-DEC→"Decision - Unanimous", S-DEC→"Decision - Split", M-DEC→"Decision - Majority", SUB→"Submission", KO/TKO stays, "TKO - Doctor's Stoppage" stays, DQ stays, CNC→"Could Not Continue", Overturned stays, anything else→"Other" (log it).
- **Politeness:** ≥1.1s between requests, identifying User-Agent, raw HTML cached to `data/raw/html_cache/` (gitignored) keyed by URL id — re-runs never re-fetch cached pages except the event-list index.
- **Incremental:** scrape the completed-events index, keep only events dated strictly after the max date in the existing UFC.csv, oldest-first; skip fight ids already present (idempotent). New fighter bios fetched only for ids not already in fighter_details.csv.
- **Draws/NC:** rows where neither corner is marked W get winner/winner_id empty (NaN on read) — `_winner_code` then applies its draw/nc logic off the method string, same as Kaggle rows.
- **Testing:** parsers are tested ONLY against small committed HTML fixtures (`tests/fixtures/ufcstats/*.html`, trimmed real pages, a few KB each). No network in pytest. Live scraping happens in implementation verification and the Action.
- **CI:** weekly Action (Mon 12:00 UTC): scrape → make_dataset → build_ratings → build_features → train_xgb → train_torch → build_display_priors → pytest → commit & push if anything changed. All CPU, minutes-scale. Concurrency-guarded, `contents: write` permission.

**HTML-structure honesty:** exact ufcstats selectors can't be dictated blind. Task 1 begins by fetching real sample pages and committing trimmed fixtures; parser code adapts to the real markup while conforming to the OUTPUT contracts specified below (which are exact and non-negotiable). Deviations from the illustrative selector code are expected; deviations from output contracts are not.

---

### Task 1: Fixtures + pure parsers, part 1 (event list + event page)

**Files:** Create `tests/fixtures/ufcstats/` (events_index.html, event_page.html — trimmed), `src/mma/scrape_parse.py`, `tests/test_scrape_parse.py`.

- [ ] **Step 1 (discovery):** politely fetch `http://ufcstats.com/statistics/events/completed?page=all` and one recent event page; save trimmed copies (keep enough rows to test: ≥3 events / ≥3 fights; strip scripts/styles/footers; keep real ids) as fixtures. Note the real markup structure in a short comment atop each fixture.
- [ ] **Step 2 (contracts + TDD):** write failing tests, then implement:

```python
def parse_events_index(html: str) -> list[dict]:
    """[{event_id, name, date (pd.Timestamp), location}], newest first as served."""

def parse_event_page(html: str) -> list[dict]:
    """[{fight_id}] for every fight row on an event page."""
```

Tests assert: ≥3 events parsed from the fixture with 16-hex event_ids, parseable dates, non-empty names; ≥3 fight ids parsed, all 16-hex, unique. Edge test: both functions return [] on an empty/`<html></html>` document rather than raising.

- [ ] **Step 3:** Full suite green (report count). Commit: `"Add ufcstats index/event parsers with fixtures"` (plain, no attribution).

### Task 2: Pure parsers, part 2 (fight details + fighter bio)

**Files:** Add fixtures fight_page.html (pick a fight with full stats; ALSO fight_page_nc.html for a no-contest or draw if findable, else document), fighter_page.html; extend `src/mma/scrape_parse.py`, `tests/test_scrape_parse.py`.

- [ ] Contracts (exact):

```python
def parse_fight_page(html: str) -> dict:
    """{
      r_id, b_id, r_name, b_name,            # from fighter links + W/L badges
      winner_id,                              # id of the W corner; None for draw/NC
      method,                                 # NORMALIZED to Kaggle vocabulary
      finish_round: int, match_time_sec: int, # 'Time:' MM:SS via parse_mmss_seconds
      total_rounds: int | None,               # 'Format:' via parse_scheduled_rounds
      division: str, title_fight: int,        # from the bout-type line
      referee: str | None,
      r_kd, b_kd, r_sig_str_landed, r_sig_str_atmpted, ... (all _STAT_COLUMNS
      pairs, parsed from the TOTALS row via parse_landed_attempted /
      parse_mmss_seconds -> ctrl in seconds), all None-safe
    }"""

def parse_fighter_page(html: str) -> dict:
    """{id, name, height_cm, reach_cm, weight_kg, stance, dob_str} —
    imperial->metric conversion (inches*2.54, lbs*0.453592, rounded 2dp);
    missing fields None; dob_str kept as ufcstats' 'Mon DD, YYYY' string."""
```

MUST reuse `mma.parsing` functions (this is what they were built for) and add a `normalize_method(raw: str) -> str` implementing the locked mapping table (unit-tested against all vocabulary entries + unknown→"Other"). division extraction reuses nothing new — store the bout-type string's weight-class portion as lowercase text (e.g. "lightweight", "women's bantamweight") to match Kaggle's `division` column; title_fight = 1 if "Title" in bout text.

- [ ] Tests: fixture-driven value assertions (hand-read the fixture and assert exact numbers: e.g. the real kd/sig counts, the real method, real finish_round/time); metric conversions hand-computed; W/L→winner_id correctness; method normalization table fully covered.
- [ ] Full suite green. Commit: `"Add fight and fighter page parsers"`.

### Task 3: Fetch/cache layer + incremental orchestrator + CLI

**Files:** `src/mma/scrape_ufcstats.py`, `scripts/scrape_ufcstats.py`, `tests/test_scrape_orchestrator.py`.

- [ ] `Fetcher` class: session with UA `"mma-prediction research scraper (github.com/dylanmryan/mma-prediction)"`; `get(url, cache_key)` → checks `data/raw/html_cache/{cache_key}.html` first; sleeps ≥1.1s between real fetches; retries once on 5xx/timeout; raises after that.
- [ ] `refresh(raw_dir, fetcher, limit_events=None) -> dict` orchestrator: reads existing CSVs → max date → parse index (index page itself NEVER cached) → filter strictly-newer events (oldest first, cap at limit_events) → per event: parse fights → per fight not already in UFC.csv: parse fight page → assemble the UFC.csv row (event fields + fight fields; leave any UFC.csv columns we don't scrape as empty) → collect unseen fighter ids → fetch/parse bios → append/upsert both CSVs atomically (write temp, rename) → return summary {events_added, fights_added, fighters_added, skipped_existing}.
- [ ] Orchestrator tests use a `FakeFetcher` serving the committed fixtures (no network): verify incremental filtering by date, idempotency (second run adds 0), column alignment with the real UFC.csv header (load the actual file header and assert the assembled row's keys ⊆ columns), fighter upsert-no-duplicate.
- [ ] CLI: `--limit-events N` and `--dry-run` flags; prints the summary.
- [ ] **Live verification** (report verbatim): `--limit-events 2` real run; then `make_dataset.py` must pass all integrity checks with the enlarged data; report new fight/fighter counts and the two event names. THEN run the full remaining refresh (no limit — expect the ~10 months since 2025-09-06, roughly 30-45 events; ~15-25 min at 1.1s/request; background it).
- [ ] Full suite green. Commit code only (raw CSVs are gitignored — the refreshed data lands in processed parquet next task): `"Add incremental ufcstats scraper"`.

### Task 4: Full pipeline refresh on scraped data

- [ ] Run in order, backgrounded, capturing tails: make_dataset → build_ratings → build_features → train_xgb → train_torch → build_display_priors. Gates: integrity checks pass; fights count > 8,337 (grew); winner val metrics move only marginally (train/val years unchanged — training data only grows if new fights predate 2021, which they don't, so **metrics should be IDENTICAL**; if they differ, investigate — the only legitimate diffs are from new fighters' snapshots). Actually: features.parquet gains rows (new fights are 2025-2026 → outside train AND val → metrics identical is the expectation; verify and report).
- [ ] `pytest` — all green (report count; test_processed_* now run against bigger data).
- [ ] Update README data-source line (Kaggle bootstrap + ufcstats incremental refresh; data current through <max date>). Commit everything incl. refreshed parquet + model jsons if changed: `"Refresh dataset from ufcstats through <date>"`.

### Task 5: Weekly GitHub Action

**Files:** `.github/workflows/refresh-data.yml`.

- [ ] Workflow: `schedule: cron '0 12 * * 1'` + `workflow_dispatch`; concurrency group; `permissions: contents: write`; ubuntu-latest; setup-python 3.11; `pip install -e ".[dev]"` (CPU torch via extra-index-url in the pip step); run scraper (no limit) → full rebuild chain → `pytest -q` → if `git status --porcelain` non-empty: commit "Weekly data refresh" as github-actions[bot] and push. Add a ~25-minute timeout and `continue-on-error: false` throughout.
- [ ] Validate YAML (`python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/refresh-data.yml').read_text())"` — add pyyaml to dev deps if absent, or use actionlint if available). NOTE: the Action can't be executed locally; flag in the report that its first real run needs watching (repo Settings → Actions must allow workflows).
- [ ] README: one line about the weekly auto-refresh. Commit: `"Add weekly data-refresh workflow"`.

### Task 6: Final review + merge

- [ ] Opus final review (adversarial): politeness/ToS posture (rate limit, UA, cache), parser robustness against fixture drift, orchestrator idempotency, no-network-in-tests guarantee, Action correctness (secrets? none needed; push auth via GITHUB_TOKEN), metrics-unchanged verification, README accuracy. Then merge to main per established preference, push, delete branch.

---

## Addendum — active tasks after the pivot

### Task A: Refresh script

**Files:** Create `scripts/refresh_data.py`, `tests/test_refresh.py`.

- `refresh_needed(raw_dir, processed_dir) -> tuple[bool, str]`: compares the freshly
  downloaded raw `UFC.csv`'s max date + row count against `data/processed/fights.parquet`;
  returns (True, reason) when raw is newer/larger, (False, reason) otherwise. Pure
  function over two DataFrames — unit-tested with tiny synthetic frames (no network).
- `main()`: run the existing `scripts/download_data.py` logic (import its `main` or
  refactor its download into a callable), then `refresh_needed`; print the verdict;
  exit 0 always, but write the verdict to stdout as `REFRESH_NEEDED=true|false` for CI
  consumption. `--force` flag skips the check.
- README: replace the scraper mention in the data-flow description with the Kaggle
  auto-refresh story; note parsing.py is retained for potential future raw-string sources.
- pyproject: drop `beautifulsoup4` (unused after pivot); keep `requests`.
- Commit: `"Add data refresh check script"`.

### Task B: Weekly GitHub Action

**Files:** `.github/workflows/refresh-data.yml`.

- Weekly cron (Mon 12:00 UTC) + workflow_dispatch; concurrency group; permissions
  contents: write; ubuntu-latest, python 3.11; install with CPU-torch extra-index;
  step 1 `python scripts/refresh_data.py` — parse `REFRESH_NEEDED` from output into a
  step output; subsequent steps gated on it: make_dataset → build_ratings →
  build_features → train_xgb → train_torch → build_display_priors → `pytest -q` →
  commit "Weekly data refresh" as github-actions[bot] + push if `git status --porcelain`
  non-empty. 30-min timeout. Note in README (one line). Kaggle access: kagglehub works
  anonymously for public datasets; if CI hits auth errors, the user adds KAGGLE_USERNAME /
  KAGGLE_KEY secrets and the workflow exports them (include the env lines commented-out
  with a pointer).
- Validate YAML parses; flag that the first live run must be watched by the user.
- Commit: `"Add weekly data-refresh workflow"`.

### Task C: Final review + merge (unchanged in spirit)

- Opus adversarial review of the branch (script correctness, Action correctness,
  README accuracy, no leftover scraper stubs), then merge to main, push.

## Done criteria (Phase 6, post-pivot)

- Suite green; refresh_needed unit-tested without network.
- Action YAML valid; refresh path documented in README; first live run flagged for user monitoring.
- No bot-detection circumvention anywhere in the repo.
