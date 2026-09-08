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
  XGBoost (0.614, 0.651) → 5-seed calibrated neural ensemble (0.621, 0.648)
  on 4,804 walk-forward fights, each scored by a model that had never seen
  its year.
- **Features earned, not assumed**: eight feature blocks measured against a
  pre-registered bar, one shipped, seven documented as negative results — and
  a selection leak in per-corner missingness flags found and removed along the
  way ([Features](#features)).
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

## Features

The feature table is assembled from named **blocks** (`src/mma/feature_blocks.py`,
`scripts/build_features.py --blocks`), and a block only ships if it clears the
pre-registered walk-forward bar. The table on disk is
**11,238 fights × 54 columns**, built from `base,external`
(`data/processed/features_blocks.json` records which blocks produced it, and the
serving path reads the same sidecar so a trained column cannot go missing at
prediction time).

- **`base`** — the v1 feature set: career volume and win/finish rates,
  per-fight and per-minute striking, takedown and control rates, streak and
  layoff, last-5 form and strength of schedule, the three Elo ratings
  (overall / striking / grappling) and Elo-tracked experience, plus physical
  differentials (height, reach, age), per-corner age and fight count, and the
  stance / debut / missingness flags. All built inside the chronological
  accumulators in `src/mma/history.py`, so every value is strictly pre-fight.
- **`external`** — what each fighter brought *into* the UFC, from the
  `ehan03/jds-mma-data` snapshot joined by ufcstats fighter id and never by
  name: `pre_ufc_wins`, `pre_ufc_losses`, `pre_ufc_finish_rate`,
  `pre_ufc_finish_loss_rate`, `pre_ufc_avg_opp_wins` and
  `days_since_pro_debut` as A-minus-B differentials. This is the only signal
  in the table that is not a recombination of the fight history: it is the
  regional career the UFC record cannot see. Worth **−0.0034** pooled
  log-loss to the neural net, and **−0.0043** on fresh seeds.

### Block results

Eight blocks were measured; **one shipped**. Every rejected block's harness
report is committed under `models/walkforward/` — negative results are
deliverables here, not deleted branches. Pooled winner log-loss, torch
deciding (the deployed scorer), against that block's incumbent:

| Block | What it added | XGB | torch | Δ vs incumbent | Ships? |
|---|---|---|---|---|---|
| `external` | pre-UFC career and origin | 0.6506 | **0.6476** | **−0.0034** | **yes** |
| `recency` | training window + recency weights | 0.6525 | 0.6499 | −0.0011 | no |
| `notice` | short-notice replacement, missed weight | 0.6501 | 0.6472 | −0.0004 | no |
| `trajectory` | Glicko-2, Elo momentum, peak-minus-current | 0.6486 | 0.6472 | −0.0004 | no |
| `context` | referee rates, home country, bonus history | 0.6499 | 0.6474 | −0.0002 | no |
| `rankings` | official weekly divisional rank | 0.6499 | 0.6487 | +0.0011 | no |
| `in_fight` | per-round and strike-target profile | 0.6540 | 0.6528 | +0.0018 | no |
| `opponent_adjusted` | rates versus what opponents allowed | 0.6519 | 0.6532 | +0.0022 | no |

Seven of eight fail, and mostly in the same shape: the tree ensemble is
roughly neutral or slightly better while the neural net — the scorer that
decides — is worse. A tree can ignore a correlated column that is NaN on most
rows; the MLP has to impute a median for it on every one of those rows and
spend capacity on the result. `in_fight`, `opponent_adjusted`, `trajectory`
and `rankings` are all recombinations of information the base table already
carries. The box score, in other words, is close to tapped out.

Two blocks failed a *serving* gate rather than the accuracy bar, which is
worth recording separately: `context`'s referee columns are present on 97.7%
of training rows and **0%** of the rows the deployed model will ever score
(Wikipedia cards do not name a referee), and `notice`'s source stops at
2024-12-14, so "unknown" is the served state for every future fight by
construction.

### The leak in the `external` block

The most interesting finding in this pass is a bug we shipped past. As
specified, the block carried a **per-corner** missingness flag. Measured that
way it looked spectacular — torch 0.6367, Δ −0.0143, the debut slice −0.0846
— and it was wrong.

Membership of the snapshot's cross-source fighter mapping is a function of how
long a fighter's UFC career turned out to be: among fighters debuting
2013–2022, the mapped share is 0.241 for those with one UFC bout ever, 0.813
at two, 0.940 at three and 1.000 at eleven or more. So "this corner is missing
from the external table" means "this fighter did not go on to have a career."
Over the 1,445 fights where exactly one corner is unmapped, that corner loses
**76%** of the time — and 90% in the debut slice. Meanwhile the actual pre-UFC
differentials carry almost nothing on their own.

The walk-forward signature is what a leak looks like when the harness catches
it: folds 2018–2023 improved by −0.026 to −0.046 while **2025 cost +0.0487**,
because in 2025 missingness stops meaning "short career" and starts meaning
"debuted after the snapshot". `no_fold_regression` failed and the variant did
not ship.

The fix keeps the flags in the **table** and out of both **model matrices**
(`mma.tensors.DROPPED`, `mma.models.xgb.MODEL_EXCLUDED`): without
`external_missing` in the table there is no `external_missing` evaluation
slice, and the coverage decay below becomes unmeasurable. The differentials
are NaN whenever either corner is unmapped, so no per-corner channel survives
— and the property asserted is a data-level one, not a rule about column
names: over those 1,445 rows the block's eight columns take **exactly one
distinct value tuple**, so the selection effect is unreachable rather than
merely unmodelled (`tests/test_external.py`).

**Coverage caveat, and why the block is provisional.** The snapshot is static
(UFC coverage ends 2024-12-14). `external_missing` is 0.201 of all rows but
0.359 of 2025 and 0.534 of 2026, and the block's gain follows: −0.0047
row-weighted on 2018–2023 against −0.0006 on 2024–2025. It pays where the
snapshot has seen the fighters and does nothing where it has not, and the
share it has not seen only grows. Keeping this block paying needs a
refreshable source — an SP4 follow-up, not a solved problem.

### Were the rejected blocks capacity-limited? A pre-registered re-test

The table above has a suspicious pattern: on the blocks with the most new
columns, the tree ensemble improves while the neural net — the scorer that
decides — gets worse. That is what a *capacity* limit looks like, not
necessarily an absence of information. A gradient-boosted ensemble can pick
the better-conditioned member of a correlated pair per split; a fixed-width
MLP must spend capacity on all of them, and the deployed MLP's shape was tuned
when the table had 46 columns. So the rejections might have been measuring the
architecture rather than the features.

That was written down as a hypothesis and tested, with the design, the arms,
the search space and the decision rule **pre-registered before anything ran**
(`docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md`). The four
rejected blocks whose numbers fit the capacity story — `trajectory`, `notice`,
`context`, `opponent_adjusted` — were restored (each first verified to
reproduce its committed SP2 number exactly, so a "gain" could not be a
restoration bug) into a combined 87-column table, and a 25-configuration torch
architecture search was run over it.

**The control arm is the point.** The identical 25 configurations were also
searched over the *shipped* 54-column table. Without that arm, any improvement
would have been credited to the features when it might have been the
architecture all along.

| Arm | Features | Architecture | Pooled | Δ vs deployed 0.6476 | Ships? |
|---|---|---|---|---|---|
| A1 | combined | deployed | 0.6475 | −0.0001 | no |
| **C** (control) | **shipped** | searched, best of 25 | 0.6463 | −0.0013 | no |
| **T** (treatment) | **combined** | searched, best of 25 | 0.6464 | −0.0012 | no |

**Neither arm cleared the 0.003 bar, and T − C = +0.0001 is inside the seed
noise floor of 0.00035.** The pre-registered rule's fourth branch applied
verbatim — *the blocks are dead and the architecture is adequate; record and
revert* — so the blocks were unregistered again and **nothing was deployed**.
Neither hypothesis survived: the extra columns are not worth 0.003 even at
higher capacity, and a search across five architecture families and five
hyper-parameters could not move the control arm a third of the bar either. The
box score and the fixed-width MLP are both at their ceiling.

Two findings are worth keeping regardless of the verdict:

- **A result that looked like a ship, and was seed luck.** On the combined
  table XGBoost improved by −0.0039 with every evaluation slice better — over
  the bar, `ships: true`. The pre-registration required a fresh-seed re-score
  before that could count. Re-run at three fresh seeds it averaged **−0.0024**,
  under the bar. Had the confirmation step been optional, this project would
  have shipped a coin flip.
- **XGBoost's seed noise, measured here for the first time: σ ≈ 0.0009–0.0013,
  about 3× the torch ensemble's 0.00035.** A single boosted fit is a much
  noisier measurement than a 5-seed average, which is exactly why the −0.0039
  looked real. No earlier rejection rests on a single-seed XGB number alone —
  the screen never dropped a block without a torch run — but XGB screening
  should use ≥3 seeds from here on, and `--model-seed` plus
  `scripts/noise_floor.py --candidate xgb` now exist for that.

Everything measured is committed (`models/walkforward/search/`,
`models/walkforward/sp2_1_decision.json`), including the strongest lead the
experiment turned up and could not act on: an equal-weight **blend** of the
two scorers reads 0.6459 on the shipped table and 0.6436 on the combined one,
with every slice improving. A blend was not one of the five pre-registered
arms, so shipping it on these numbers would be the exact after-the-fact
search-widening the pre-registration forbids. It gets its own experiment.

### Data sources and licences

| Source | Used for | Licence |
|---|---|---|
| [Kaggle UFC dataset](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025) (`neelagiriaditya`) | the primary fights / fighters / round-stats tables | per the dataset page |
| [`ehan03/jds-mma-data`](https://github.com/ehan03/jds-mma-data) | pre-UFC career, nationality, notice and weigh-in tables (`data/external/`) | **MIT** (notice vendored as `data/external/LICENSE-jds-mma-data`) |
| [`jerzyszocik/ufc-rankings-history`](https://www.kaggle.com/datasets/jerzyszocik/ufc-rankings-history) | the weekly divisional rankings (`data/external/rankings.parquet`) | **CC0** |
| [`jerzyszocik/ufc-betting-odds-daily-dataset`](https://www.kaggle.com/datasets/jerzyszocik/ufc-betting-odds-daily-dataset) | the market benchmark only — never a model feature | **CC0** |
| [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats) | secondary daily gap-filler (`scripts/refresh_secondary.py`) | **GPL-3.0** — used as a *data* source only (its published CSVs are fetched over HTTPS; no GPL code is vendored), writes off by default, report-only in CI |
| Wikipedia event pages | upcoming fight cards, and the withdrawal / missed-weight parser | CC BY-SA 4.0 |

Only *derived*, compact tables are committed (a few MB); the raw snapshots
are never vendored and are regenerated by `scripts/build_external.py` and
`scripts/build_rankings.py`.

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

**Winner prediction** (4,804 pooled walk-forward fights, 2018–2026), on the
shipped `base,external` feature table:

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Elo baseline | 0.553 | 0.683 | 0.245 |
| XGBoost (46 modelled features) | 0.614 | 0.651 | 0.230 |
| **Neural net** (5-seed ensemble, calibrated) | **0.621** | **0.648** | **0.228** |

Pooled method macro-F1: XGBoost 0.333, neural net 0.387; finish-round
macro-F1 (finishes only, 2,446 fights): XGBoost 0.178, neural net 0.315 —
the same trade-off as before: the class-weighted neural heads identify
submissions and early finishes instead of defaulting to the majority class.
On the v1 46-column table the same protocol gave XGBoost 0.6537 and the
neural net 0.6510 (`models/walkforward/{xgb,torch}_v1.json`), so the
`external` block is worth −0.0034 to the scorer that ships.

**Noise floor and pre-registered bar.** Re-running the neural walk-forward
with three disjoint 5-seed sets gives pooled log-losses of 0.6510 / 0.6516 /
0.6510, i.e. **σ_seed ≈ 0.00035** (an n=3 estimate; 95% CI roughly
0.00018–0.0022, `models/walkforward/noise_floor.json`). The pre-registered
bar for a challenger to *replace* the incumbent is **0.003 pooled log-loss**
— roughly 9σ, so a change has to be far larger than seed-to-seed wobble, and
must not regress any single fold, before it ships.

**Fresh-seed re-score.** The shipped feature set was re-run on seeds 5–9
against a paired incumbent — the v1 recipe on the *same* 54-column table
with the eight external columns held out of the model, which reproduces
`torch_v1_seeds5.json` bit-exactly. Pooled **0.6516 → 0.6473, Δ −0.0043**,
worst fold +0.0006: the block clears the bar on seeds it was never chosen
on, by more than it did on seeds 0–4.

**Refit strategy.** The harness then asked whether early-stopping on a
held-out year (protocol A) is actually necessary, or whether a fixed budget
taken from those early-stopping runs, trained on *all* data through the
newest year (protocol B), does as well. On the same folds and the shipped
table: XGBoost 0.6506 → 0.6507, neural net 0.6476 → 0.6470 — both within the
noise floor (`models/walkforward/refit_decision_v3.json`). Since B is not
worse and uses every available fight, the deployed models are trained on all
**11,238 decisive fights through 2026-08-08** with that fixed budget (neural
net: 10 epochs, temperature 1.07 on every seed; XGBoost: 105/61/75 trees for
the winner/method/round heads). The budget belongs to a *feature table*, not
to the recipe: the v1 table's budget was 14 epochs at temperature 1.1 and
82/80/76 trees (`refit_decision.json`), and both files record which table
they were taken on. `models/torch/metrics_val.json` and
`models/xgb_metrics_val.json` record the recipe and quote the harness
numbers as their evidence. A fresh-seed re-score of the shipped recipe
(seeds 5–9 instead of 0–4) reproduced the same result — pooled
0.6473 → 0.6475, Δ +0.0002, still inside σ_seed — confirming the recipe
choice wasn't a seed-lucky fluke (`fresh_seed_rescore` in
`models/walkforward/refit_decision_v3.json`).


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
deployed refit applies the harness-derived temperature 1.07 to every seed.
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
coin-flip baseline of 0.476 accuracy on the same fights. Those 78 predictions
were made by model `40df77ec43c7`, the SP1 model trained on the 46-column
table; the SP2 feature set and re-derived budget produce a different scorer
(`b617b96dae45`), so the next weekly run opens a new section rather than
mixing the two models' predictions into one row. 29 further
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
  refresh the refit reuses the committed budget (10 epochs / T 1.07;
  105/61/75 trees) on the newer data, but the harness reports are *not*
  re-run by the Action — re-run `scripts/run_walkforward.py`,
  `scripts/noise_floor.py`, and `scripts/refit_decision.py --reports v3` by
  hand to
  refresh the evidence (the train scripts warn when the harness's data is
  older than the training cutoff). Automating that is SP4.

## Interactive app

`app.py` is a Streamlit front end over the committed ensemble: pick two
fighters, a weight class, round count, and title-fight flag, and it renders
the win probability, ensemble spread, MC-dropout uncertainty histogram,
method-of-victory / finish-round breakdown, both fighters' Elo
trajectories, and a "Why this prediction?" panel breaking down the top
factors driving the call — all from the artifacts already checked into
`models/torch/` and `models/xgb_<head>_seed*.json`, no training required.
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
