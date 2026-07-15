# MMA Fight Prediction

![Weekly data refresh](https://github.com/dylanmryan/mma-prediction/actions/workflows/refresh-data.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**[▶ Try the live app](https://mma-prediction-lxqvmgheqvzccdpyord3q9.streamlit.app/)** — pick any two UFC fighters, get calibrated win probabilities with uncertainty.

Predicting UFC fight winners, method of victory, and finish round —
an Elo rating system, gradient boosting, and a calibrated multi-task
neural-network ensemble, each evaluated honestly on strict time splits,
plus an interactive Streamlit matchup explorer.

**Highlights**

- **Point-in-time discipline, machine-verified**: every feature is built only
  from data available before each fight; a truncation-invariance test proves
  no feature can see the future.
- **Baseline ladder**: coin flip → Elo (0.576 acc) → XGBoost (0.595) →
  5-seed calibrated neural ensemble (0.606, best log-loss) on 1,507
  never-tuned-on validation fights.
- **Uncertainty done properly**: deep-ensemble spread + MC dropout, per-seed
  temperature scaling, display probabilities recalibrated to historical base rates.
- **Self-updating**: a weekly GitHub Action refreshes the dataset and rebuilds
  every artifact; the entire pipeline reproduces byte-for-byte.
- **Prospective evaluation**: real upcoming UFC events get predicted and
  committed to git *before* they happen, then auto-graded afterward — see
  [Prospective track record](#prospective-track-record) below.

Design doc: `docs/superpowers/specs/2026-07-10-mma-prediction-design.md` ·
Phase plans: `docs/superpowers/plans/`

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py   # Kaggle UFC dataset -> data/raw/
.venv/bin/python scripts/make_dataset.py    # clean parquet -> data/processed/
.venv/bin/python scripts/build_ratings.py   # tune + build Elo ratings
.venv/bin/pytest
```

Interactive app: `streamlit run app.py` (pick any two fighters, get win
probability with uncertainty, method and finish-round tendencies, and a
"Why this prediction?" breakdown of the top factors behind the call).

Data bootstraps from the [Kaggle UFC dataset](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
via `kagglehub` and auto-refreshes weekly from the same maintained mirror
(`scripts/refresh_data.py`). A direct ufcstats.com scraper was planned but
dropped — the site gates automated clients behind an anti-bot challenge;
`src/mma/parsing.py` is retained in case a future raw-string data source
needs its parsing helpers. The `.github/workflows/refresh-data.yml` Action
runs this refresh weekly and commits any rebuilt artifacts automatically.

## Results so far

All models are evaluated on a strict time split: trained on pre-2021 fights,
reported on 2021–2023 validation fights. **Test years (2024+) are held out
until the final model comparison.**

**Winner prediction** (1,507 validation fights):

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Coin flip | 0.500 | 0.693 | 0.250 |
| Higher-Elo-wins dummy | 0.573 | — | — |
| Elo baseline | 0.576 | 0.678 | 0.242 |
| XGBoost (46 features) | 0.595 | 0.658 | 0.233 |
| **Neural net** (5-seed ensemble, calibrated) | **0.606** | **0.654** | **0.231** |

**Method of victory** (3 classes): XGBoost 0.489 accuracy vs 0.485 majority-class
baseline, macro-F1 0.281. The neural net's class-weighted method head makes a
different trade: 0.407 accuracy but **macro-F1 0.364** — it actually identifies
submissions and KOs instead of defaulting to "decision". **Finish round**
(R1/R2/R3/R4-5, finishes only): XGBoost 0.477 accuracy vs 0.474 majority
baseline, macro-F1 0.167. Predicting *how* fights end is genuinely hard; these
numbers are reported honestly rather than hidden.

## Final held-out test results (2024+)

These numbers were computed exactly once, by `scripts/final_test_eval.py`,
after all development was frozen — the 2024+ fights were never read by any
training, tuning, or calibration code at any point in the project (the
per-seed temperatures were fit on 2021–2023 validation and reused as-is).

**Winner prediction** (873 test fights):

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Coin flip | 0.472 | 0.693 | 0.250 |
| Higher-Elo-wins dummy | 0.558 | — | — |
| Elo baseline | 0.553 | 0.670 | 0.239 |
| XGBoost | 0.642 | 0.635 | 0.222 |
| **Neural net** (5-seed ensemble, calibrated) | **0.645** | **0.633** | **0.221** |

(The coin flip's 0.472 is just the test split's class balance: p = 0.5
ties break toward "A wins", and corner A won 47.2% of test rows.)

**Method of victory** (871 test fights): XGBoost 0.535 accuracy /
macro-F1 0.298; the neural ensemble 0.434 accuracy / **macro-F1 0.380**;
majority-class ("decision") baseline 0.540 accuracy / macro-F1 0.234 —
the same trade-off each model made on validation. **Finish round**
(401 test finishes): XGBoost 0.494 accuracy / macro-F1 0.169 vs a
majority ("round 1") baseline of 0.499 / 0.166 — still barely
distinguishable from the base rate.

The validation results held up on test: the learned models actually
improved out of sample (XGBoost 0.595 → 0.642 accuracy, neural net
0.606 → 0.645, both with better log-loss) while the Elo baseline slipped
slightly (0.576 → 0.553), and the ladder's ordering was preserved.

The neural net is a multi-task network (shared trunk; winner, method, and
finish-round heads) trained as a deterministic 5-seed ensemble with per-seed
temperature scaling — fitted temperatures all land near 1.0, i.e. the raw
model was already well calibrated. Uncertainty comes from ensemble spread
(mean 0.087) and MC dropout. Per the Phase 3 ablation, the era-proxy
`*_missing` flags are excluded from its inputs.

What predicts the winner? Reach and age differentials, Elo differential, and
opponent-quality-adjusted activity rates top the feature importances. A caveat
worth knowing: reach-*missingness* indicators rank highly, which likely proxies
for era and fighter obscurity rather than physiology.

The Elo system: fighters start at 1500; K = 64 for a fighter's first 5 UFC
fights then 48 (tuned by grid search on pre-2020 log-loss); KO/sub wins get a
1.4× update bonus; separate striking and grappling Elos update in proportion
to how striking-dominated each fight was.

All-time peak Elo (through 2025-09):

| # | Fighter | Peak Elo |
|---|---|---|
| 1 | Jon Jones | 1985 |
| 2 | Islam Makhachev | 1938 |
| 3 | Georges St-Pierre | 1916 |
| 4 | Anderson Silva | 1912 |
| 5 | Kamaru Usman | 1907 |
| 6 | Charles Oliveira | 1877 |
| 7 | Max Holloway | 1872 |
| 8 | Francis Ngannou | 1870 |
| 9 | Khabib Nurmagomedov | 1867 |
| 10 | Tony Ferguson | 1867 |

## Prospective track record

Every result above is a backtest: the model already knows the outcome of
every fight it's scored on, and it's possible (even with honest time
splits) to unconsciously tune a project until it looks good in hindsight.
So this project also makes **public, timestamped predictions for real
upcoming UFC events, committed to git before those events happen, and
graded automatically afterward.**

This is the strongest evaluation in the repo, for one reason: a prediction
committed to git history with a timestamp *cannot be edited after the fact*
to look better than it was. Every Monday, `.github/workflows/refresh-data.yml`
runs `scripts/predict_upcoming.py`, which:

1. Fetches the "List of UFC events" page from Wikipedia (MediaWiki API,
   polite rate limit) and keeps events in the next 30 days.
2. Fetches each event's fight card and matches fighter names against
   `fighters.parquet` by exact, unicode-normalized string match. A name
   that matches zero or multiple fighters is **skipped with a logged
   reason** rather than guessed.
3. Predicts every matched fight with the exact committed ensemble
   (`mma.inference.predict_symmetrized`) and writes one JSON record per
   event to `predictions/`, stamped with the prediction time and the git
   sha of the model that made it.
4. Writing is idempotent: re-running never overwrites an existing
   prediction, even if the fighters' stats or the model itself have since
   changed. Fights announced later are appended with their own timestamp.

A second step, `scripts/grade_predictions.py`, runs every week too: once an
event's date has passed, it looks up the actual result in
`data/processed/fights.parquet` (matched on the fighter pair + date within
3 days) and appends grading fields — never touching the original
prediction — to the same file. Two comparison baselines are graded
alongside the model on every fight: a coin flip and a "higher Elo wins"
dummy (using the Elo ratings recorded at prediction time, so the dummy
stays gradeable even as ratings keep moving). Aggregate stats land in
[`predictions/track_record.json`](predictions/track_record.json).

**Current status** (honest — this just started):

| Model version | Fights predicted | Fights graded | Accuracy | Log-loss |
|---|---|---|---|---|
| `79135ef` (current) | 13 | 0 | — | — |

All 13 predictions are pending: they cover 4 events between 2026-07-18 and
2026-08-08, none of which have happened yet. Grading itself can lag a
finished event by days to weeks, because it depends on the Kaggle mirror
picking up the result — the same lag documented for the weekly data
refresh above. Nothing here is cherry-picked: every prediction this
pipeline ever makes gets a row, win or lose.

A walk-forward retraining hook (`scripts/roll_window.py`) watches this
track record: once 150 graded prospective fights have accumulated since
the current model's data cutoff, it reports a pre-registered promotion
protocol (retrain on a pushed-forward cutoff, validate on the newest 2
years, promote only if the new model beats the incumbent re-evaluated on
that *same* slice by more than 0.002 log-loss). Promotion is deliberately
a manual, human-reviewed step (`--execute`, run by hand via
`workflow_dispatch`) — the weekly Action only ever runs it in `--dry-run`
and prints the report.

## Interactive app

`app.py` is a Streamlit front end over the committed ensemble: pick two
fighters, a weight class, round count, and title-fight flag, and it renders
the win probability, ensemble spread, MC-dropout uncertainty histogram,
method-of-victory / finish-round breakdown, both fighters' Elo
trajectories, and a "Why this prediction?" panel breaking down the top
factors driving the call — all from the artifacts already checked into
`models/torch/` and `models/xgb_winner.json`, no training required.
Predictions are symmetrized across both fighter orderings
(`mma.inference.predict_symmetrized`) so the reported probability is always
self-consistent.

Run it locally:

```bash
.venv/bin/pip install -e ".[dev,app]"
.venv/bin/streamlit run app.py
```

Deploy to Streamlit Community Cloud: push this repo to a public GitHub
remote, go to [share.streamlit.io](https://share.streamlit.io), click
**New app**, and point it at this repo/branch with `app.py` as the entry
point.
