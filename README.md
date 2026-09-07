# MMA Fight Prediction

![Weekly data refresh](https://github.com/dylanmryan/mma-prediction/actions/workflows/refresh-data.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**[▶ Try the live app](https://mma-prediction-lxqvmgheqvzccdpyord3q9.streamlit.app/)** — pick any two UFC fighters, get calibrated win probabilities with uncertainty.

Predicting UFC fight winners, method of victory, and finish round —
an Elo rating system, gradient boosting, and a calibrated multi-task
neural-network ensemble, each evaluated honestly by expanding-window
walk-forward over 2018–2026, plus an interactive Streamlit matchup explorer.

**Highlights**

- **Point-in-time discipline, machine-verified**: every feature is built only
  from data available before each fight; a truncation-invariance test proves
  no feature can see the future.
- **Baseline ladder**: coin flip → Elo (0.553 acc, 0.683 log-loss) →
  XGBoost (0.613, 0.654) → 5-seed calibrated neural ensemble (0.610, 0.651)
  on 4,804 walk-forward fights, each scored by a model that had never seen
  its year.
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
(`scripts/refresh_data.py`). The upstream dataset was rebuilt in August 2026
with a new file layout — `master.csv` (one row per fight, including fight
totals), `fighter.csv`, `round.csv` (per-round stats, including head/body/leg
and distance/clinch/ground strike splits), and `fighter_bonus.csv`, plus
referees. Ingestion switched to this layout in September 2026;
`scripts/reconcile_sources.py` audits the swap and confirmed the new source
is a strict superset of the old one — all 8,337 previously ingested fights
are present with identical labels (one 2025 result was relabelled draw vs.
no-contest by the source itself), and the new source adds 3,104 fights,
concentrated in 1997–2011 (hundreds per year in 2004–2011, mostly fights the
earlier scrape had dropped) plus 40–80 per year from 2017 onward and the
September 2025 – August 2026 refresh. Every artifact in this repo was
retrained on the resulting 11,441 fights. A direct ufcstats.com scraper was
planned but dropped — the site gates automated clients behind an anti-bot
challenge; `src/mma/parsing.py` now supplies the string parsers the new
layout needs. The `.github/workflows/refresh-data.yml` Action runs this
refresh weekly and commits any rebuilt artifacts automatically.

## Results so far

Every model is evaluated by **expanding-window walk-forward** over
2018–2026 (`scripts/run_walkforward.py`, reports in `models/walkforward/`),
and the deployed models are then **refit on every decisive fight through the
latest event** — there is no longer a historical year held out from the model
the app serves. The [prospective track record](#prospective-track-record) —
predictions committed to git before the fights happen — is the only true
holdout. The [Final held-out test results](#final-held-out-test-results-2024)
and the market benchmark are one-time artifacts from July 2026 and are left
as recorded.

### Walk-forward evaluation (2018–2026)

Protocol: for each fold year *Y* from 2018 to 2025, train on every fight
dated before *Y−1*, early-stop (and, for the neural net, fit per-seed
temperatures) on *Y−1*, and score *Y*; the last fold also absorbs the
2026 fights. Each fight is scored exactly once, by a model that never saw
its year, and the eight folds pool to **4,804 fights**. Elo carries no
tuning step: its ratings are recomputed point-in-time and scored directly.

**Winner prediction** (4,804 pooled walk-forward fights, 2018–2026):

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Elo baseline | 0.553 | 0.683 | 0.245 |
| XGBoost (46 features) | 0.613 | 0.654 | 0.231 |
| **Neural net** (5-seed ensemble, calibrated) | **0.610** | **0.651** | **0.230** |

Pooled method macro-F1: XGBoost 0.330, neural net 0.391; finish-round
macro-F1 (finishes only, 2,446 fights): XGBoost 0.183, neural net 0.305 —
the same trade-off as before: the class-weighted neural heads identify
submissions and early finishes instead of defaulting to the majority class.

**Noise floor and pre-registered bar.** Re-running the neural walk-forward
with three disjoint 5-seed sets gives pooled log-losses of 0.6510 / 0.6516 /
0.6510, i.e. **σ_seed ≈ 0.00035** (an n=3 estimate; 95% CI roughly
0.00018–0.0022, `models/walkforward/noise_floor.json`). The pre-registered
bar for a challenger to *replace* the incumbent is **0.003 pooled log-loss**
— roughly 9σ, so a change has to be far larger than seed-to-seed wobble, and
must not regress any single fold, before it ships.

**Refit strategy.** The harness then asked whether early-stopping on a
held-out year (protocol A) is actually necessary, or whether a fixed budget
taken from those early-stopping runs, trained on *all* data through the
newest year (protocol B), does as well. On the same folds: XGBoost
0.6537 → 0.6524, neural net 0.6510 → 0.6512 — both within the noise floor
(`models/walkforward/refit_decision.json`). Since B is not worse and uses
every available fight, the deployed models are therefore trained on all
**11,238 decisive fights through 2026-08-08** with that fixed budget (neural
net: 14 epochs, temperature 1.1 on every seed; XGBoost: 82/80/76 trees for
the winner/method/round heads). `models/torch/metrics_val.json` and
`models/xgb_metrics_val.json` record the recipe and quote the harness
numbers as their evidence.

### Original validation window (2021–2023, for continuity)

The numbers below described models trained on pre-2021 fights only and
scored on 2021–2023 (1,709 fights); they no longer describe the deployed
models, which train through 2026, and are kept for continuity with earlier
write-ups.

**Winner prediction** (1,709 validation fights, pre-2021 models):

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Coin flip | 0.500 | 0.693 | 0.250 |
| Higher-Elo-wins dummy | 0.559 | — | — |
| Elo baseline | 0.559 | 0.680 | 0.243 |
| XGBoost (46 features) | 0.606 | 0.660 | 0.234 |
| **Neural net** (5-seed ensemble, calibrated) | **0.608** | **0.651** | **0.230** |

**Method of victory** (3 classes): XGBoost 0.489 accuracy vs 0.477 majority-class
baseline, macro-F1 0.303. The neural net's class-weighted method head makes a
different trade: 0.437 accuracy but **macro-F1 0.390** — it actually identifies
submissions and KOs instead of defaulting to "decision". **Finish round**
(R1/R2/R3/R4-5, finishes only, 891 validation fights): XGBoost 0.495 accuracy
vs 0.495 majority baseline, macro-F1 0.169. Predicting *how* fights end is
genuinely hard; these numbers are reported honestly rather than hidden.

## Final held-out test results (2024+)

*This artifact is frozen from the July 2026 model: computed once on
2026-07-13 against the pre-2021-trained models and 8,337-fight dataset of
that date, and never recomputed. The 2024+ years are now part of both the
walk-forward evaluation above and the deployed model's training data, so
this holdout cannot be repeated — `scripts/final_test_eval.py` refuses to
run against a refit model.*

These numbers were computed exactly once, by `scripts/final_test_eval.py`,
after all development was frozen — at that point the 2024+ fights had never
been read by any training, tuning, or calibration code (the per-seed
temperatures were fit on 2021–2023 validation and reused as-is).

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
temperature scaling — in the original split protocol the fitted temperatures
all landed near 1.0, i.e. the raw model was already well calibrated; the
deployed refit applies the harness-derived temperature 1.1 to every seed.
Uncertainty comes from ensemble spread (mean 0.087 on the original
validation window) and MC dropout. Per the Phase 3 ablation, the era-proxy
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
   `fighters.parquet` in two tiers: an exact unicode-normalized match
   first, then (only if the exact match finds nothing) an accent-folded
   match — Wikipedia writes "Rakić" where the dataset stores "Rakic".
   Either tier must find exactly **one** fighter; a name matching zero or
   multiple fighters is **skipped with a logged reason** rather than
   guessed, and every prediction records which tier matched it
   (`match_tier`). If the scheduled-events parse ever returns nothing at
   all — which can only mean Wikipedia's page structure changed, since the
   UFC always has future events booked — this step fails loudly instead of
   reporting an empty success. That failure only stops *new* predictions;
   the Action's grading, retrain-check, and commit steps are wired with
   `if: always()` so a broken Wikipedia parser can't stall grading of
   predictions already committed to git.
3. Predicts every matched fight with the exact committed ensemble
   (`mma.inference.predict_symmetrized`) and writes one JSON record per
   event to `predictions/`, stamped with the prediction time and the git
   sha of the model that made it.
4. Writing is idempotent: re-running never overwrites an existing
   prediction, even if the fighters' stats or the model itself have since
   changed. Fights announced later are appended with their own timestamp.
   One deliberate exception: a fight previously recorded as *skipped*
   contains no prediction, only a failure reason — so re-runs re-attempt
   it, and if it now matches (a debuting fighter entering the dataset, a
   matcher improvement), the skip stub is replaced by a real prediction
   with a fresh timestamp. That's still strictly pre-event, so the
   integrity guarantee holds; actual predictions remain immutable.

A second step, `scripts/grade_predictions.py`, runs every week too: once an
event's date has passed, it looks up the actual result in
`data/processed/fights.parquet` (matched on the fighter pair + date within
3 days) and appends grading fields — never touching the original
prediction — to the same file. Two comparison baselines are graded
alongside the model on every fight: a coin flip and a "higher Elo wins"
dummy (using the Elo ratings recorded at prediction time, so the dummy
stays gradeable even as ratings keep moving). Aggregate stats land in
[`predictions/track_record.json`](predictions/track_record.json).

**Current status**:

| Model version | Fights predicted | Fights graded | Accuracy | Log-loss |
|---|---|---|---|---|
| `40df77ec43c7` | 78 | 21 | 0.667 | 0.616 |

21 of the 78 predicted fights have been graded so far (events through
2026-08-08): 0.667 accuracy, 0.616 log-loss, 0.213 Brier — against a
coin-flip baseline of 0.476 accuracy on the same fights. 29 further
predictions are awaiting results, and the rest cover events that haven't
happened yet. Grading itself can lag a finished event by days to weeks,
because it depends on the Kaggle mirror picking up the result — the same
lag documented for the weekly data refresh above — but continues
automatically as results land. Nothing here is cherry-picked: every
prediction this pipeline ever makes gets a row, win or lose.

A walk-forward retraining hook (`scripts/roll_window.py`) watches this
track record: once 150 graded prospective fights have accumulated since
the current model's data cutoff, it reports a pre-registered promotion
protocol. The gate operates on the **full 5-seed torch ensemble — the exact
model the app serves**, not a proxy: it retrains the ensemble on a pushed-
forward cutoff into a temp dir, scores both that candidate and the committed
incumbent ensemble on the same newest-2-years held-forward slice (each as
its complete artifact, per-seed temperatures included), and promotes only if
the candidate beats the incumbent by more than 0.002 log-loss. On promotion
the candidate ensemble is *staged* into `models/torch` (the incumbent is
backed up on disk first) and nothing else happens — the script performs no
git writes. A human then re-runs the refit recipe (bare `scripts/train_xgb.py`,
`scripts/train_torch.py`, `scripts/build_display_priors.py` — a split-protocol
candidate never ships as-is), runs the suite, reviews the metrics diff, and
commits by hand; the new model's artifact hash (`mma.versioning.model_version`,
computed over the torch weights and preprocessing stats) becomes the new
`model_version` and starts a fresh `track_record.json` section automatically.
Promotion is deliberately a
manual, stage-only step (`--execute`, run by hand via `workflow_dispatch`)
and is never wired into CI auto-promotion — the weekly Action only ever runs
it in `--dry-run` and prints the report. One caveat since the refit recipe
shipped: the deployed incumbent trains through the latest event, so the
newest-2-years slice is *in-sample* for it and `--execute` now aborts
rather than run an invalid comparison. Until SP4 moves this gate onto the
walk-forward harness, recipe comparisons go through the harness
(`models/walkforward/`), not this slice.

## Model vs. the betting market

Every metric above compares the model to other models (or to itself).
The real question for any prediction system is whether it beats the
market — professional bookmaker lines are one of the sharpest, most
efficient forecasts that exist for a sporting event, built from liquid
money and constant correction, and closing lines in particular are
close to the ceiling of what's knowable pre-fight. So `scripts/build_odds_benchmark.py`
pulls historical UFC moneylines ([`jerzyszocik/ufc-betting-odds-daily-dataset`](https://www.kaggle.com/datasets/jerzyszocik/ufc-betting-odds-daily-dataset),
CC0, via kagglehub) and scores the committed neural ensemble against the
devigged market-implied probability on the same fights. **The odds are
used only as an evaluation comparator — they are never a model feature**;
nothing in this benchmark touches training, tuning, or inference.

Alignment uses the shared 16-hex ufcstats fight id embedded in both
datasets' URLs, with each fight's two bookmaker-quoted fighters matched to
our `fighter_a`/`fighter_b` convention by their ufcstats fighter ids
(present for every fight in this dataset, so the name-based accent-folding
fallback — reused from the prospective pipeline — never had to fire: 6,274
fights aligned by id, 0 by name, 238 skipped for having no usable odds
rows). Alignment is sanity-checked against five famous fights with known,
independently-verifiable betting favorites — including two where the
favorite *lost* (Holly Holm over Ronda Rousey, Chris Weidman over Anderson
Silva) and one rematch where the odds file lists the same two fighters in
reversed column order (proving this isn't a "column 1 is always the
favorite" bug) — before any aggregate numbers are trusted.

The **headline comparison** restricts to fights on or after 2021-01-01 —
the validation+test era of the July 2026 model this benchmark was computed
against, which that model never trained on — so the model isn't credited
for fights it already knows the answer to. (The currently deployed model is
refit through the latest event, so those years are in-sample for it and the
benchmark is left as the one-time July 2026 artifact rather than recomputed.)
Odds coverage isn't total, but on the matched, 2021+, odds-available set
(**1,956 fights**):

| | Accuracy | Log-loss | Brier |
|---|---|---|---|
| **Model** (neural ensemble) | 0.619 | 0.644 | 0.227 |
| **Market** (devigged consensus odds) | **0.671** | **0.606** | **0.209** |

The market wins on every metric, by a comfortable but not suspicious
margin (Δlog-loss = +0.038 in the market's favor). **This is the expected,
correct outcome, not a disappointing one** — a model built from
box-score-derived features losing to a market that also prices in
injuries, weight cuts, camp changes, and everything else the betting
public knows the morning of the fight is exactly what a well-behaved
evaluation should show. The honesty gate in `build_odds_benchmark.py`
would have stopped the pipeline and refused to report clean numbers if the
model had implausibly *beaten* the market instead (a >0.02 log-loss edge
in the model's favor almost always means a leakage or alignment bug, not a
real edge). A secondary cut over all 6,273 matched fights (including
pre-2021 fights the model trained on, so treat this as a looser sanity
check rather than an honest comparison) shows the same ordering: market
0.610 log-loss vs. model 0.641.

**Calibration**: both are well-behaved across probability deciles — mean
predicted probability tracks the empirical win rate bin-by-bin for both
the model and the market — but the market's predictions spread further
into the confident tails (more fights called at >70% or <20%), while the
model stays comparatively conservative in the middle of the range. That
extra confidence, where warranted, is a big part of where the market's
sharper log-loss comes from.

**Simulated ROI** (flat 1-unit stake, betting whenever the model's
probability exceeds the market's devigged implied probability by a
threshold, settled at that fighter's actual decimal odds) is negative at
every threshold tested, for both the favorite-edge and underdog-edge
variants — consistent with a sharp market and a model that doesn't beat
it. This is an **in-sample-of-the-market backtest, not a strategy
claim**: no bankroll management, no line-shopping or timing realism, no
transaction costs, and it's evaluated on the same historical lines used
for the log-loss comparison above.

Full numbers (n_fights, per-metric breakdowns, 10-bin calibration tables,
and the full ROI sweep at 0%/5%/10% thresholds) are in the committed
[`models/market_benchmark.json`](models/market_benchmark.json).

## Development notes

- **Local-disk virtualenv.** If the repo lives in an iCloud-synced folder,
  create the venv elsewhere (`python3 -m venv ~/.venvs/mma && ~/.venvs/mma/bin/pip install -e ".[dev,app]"`)
  — torch's shared libraries stall for minutes when paged in from iCloud.
  Better still, keep the repo itself outside iCloud (or exclude `.git` from
  sync): sync has corrupted `.git` metadata here more than once.
- `OMP_NUM_THREADS=1` for any script that imports both torch and xgboost.
- **Model identity and evidence.** Prospective predictions are stamped with
  `mma.versioning.model_version()`, a hash of the deployed torch weights and
  preprocessing statistics, so the track record splits by model, not by
  commit. Retraining (weekly refresh or walk-forward promotion) starts a new
  section automatically; retraining is deterministic, so an unchanged
  dataset yields an unchanged version. The evidence behind each deployed
  model lives in `models/walkforward/` (the harness reports, noise floor,
  and refit decision), and the metrics files quote it. After a weekly data
  refresh the refit reuses the committed budget (14 epochs / T 1.1;
  82/80/76 trees) on the newer data, but the harness reports are *not*
  re-run by the Action — re-run `scripts/run_walkforward.py`,
  `scripts/noise_floor.py`, and `scripts/refit_decision.py` by hand to
  refresh the evidence (the train scripts warn when the harness's data is
  older than the training cutoff). Automating that is SP4.

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
