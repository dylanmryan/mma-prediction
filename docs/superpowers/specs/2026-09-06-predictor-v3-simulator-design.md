# Predictor v3: Simulated Fights, Expanded Data, Walk-Forward Evaluation — Design

**Date:** 2026-09-06
**Status:** Approved 2026-09-06; amended the same day after SP0 (recency, data leads).
SP0, SP1 and **SP2 done** (SP2 signed off 2026-09-08 — see its section below),
plus **SP2.1**, a pre-registered re-test of SP2's rejections that shipped
nothing, and **SP2.2**, a pre-registered blend experiment that **did** ship
and changed what "the deployed model" means (see both subsections); SP3 and
SP4 outstanding. **The deployed scorer is a blend of a 5-seed XGBoost
ensemble and the 5-seed torch ensemble, hash `5aa33460ef40`** — statements
elsewhere in this document that the torch ensemble is the deployed model
describe the state before 2026-09-08.
**Scope:** Predictions only. The value-betting / odds-analysis layer is a
later, separate spec (see "Out of scope").

## 1. Why this pass exists

The v1 models are done and honest: winner log-loss 0.633 on the (now spent)
2024+ test set, with the torch ensemble the best of a coin-flip → Elo →
XGBoost → neural ladder. Two later experiment sessions (Elo v1.1, model v2)
tuned and blended everything reachable from the current feature table and
gained at most 0.003 log-loss, against a measured ensemble noise floor of
about 0.0014. The devigged betting market beats the model by 0.038 on the
same fights. That gap is an order of magnitude larger than anything tuning
has produced, so this pass changes **what the model sees** and **what it
predicts**, not how hard it is tuned:

1. **New signal.** The Kaggle source was rebuilt in August 2026 with a new
   schema that carries per-round stats, strike-target and position splits,
   referees, and bonuses — none of which the feature table uses. A public,
   MIT-licensed snapshot of Sherdog, Tapology, and Fight Matrix data (through
   roughly January 2025) supplies pre-UFC careers, gyms, and nationality.
   The best leak-controlled public work (a 2025 Yale thesis that matched the
   Bovada closing line on ~4,000 walk-forward fights) attributes its gains
   to exactly these kinds of alternative data, not to architecture.
2. **A different prediction paradigm.** Replace the three loosely coupled
   classification heads with a **round-by-round fight simulator**: a
   per-round hazard model for each finish type plus a decision model, played
   out by Monte Carlo. Winner, method, round, and "goes the distance" all
   fall out of one coherent process. This directly targets the method and
   round heads, which today barely beat the majority class.
3. **A bigger, pre-registered evaluation.** 1,507 validation fights cannot
   resolve 0.003 effects, and the 2024+ test set is spent. An expanding-window
   walk-forward over 2018–2025 evaluates on ~3,900 fights and lets every
   experiment be judged against a measured noise floor.

Everything the project already does well — point-in-time discipline with the
truncation-invariance test, corner symmetrization, reproducible artifacts, the
weekly refresh Action, the prospective track record — is preserved and
extended, not replaced.

## 2. Findings that shape the design (verified this session)

| Finding | Consequence |
|---|---|
| Kaggle dataset `neelagiriaditya/ufc-datasets-1994-2025` was rebuilt 2026-08-11: new files `fighter.csv`, `event.csv`, `fight.csv` (11,441 rows, +referee, +bonuses), `round.csv` (25,131 per-round rows); same ufcstats hex ids. | Our download/refresh scripts are pinned to the old `UFC.csv` layout, so the weekly Action has found "no new data" for a year. Re-ingestion is the first task. The 11,441 vs 8,337 fight-row discrepancy must be reconciled before trusting the new file. |
| Origin holds 78 prospective predictions, 0 graded, because grading waits on refreshed data. | Once ingestion works, grading catches up automatically and the track record becomes real. |
| `scripts/predict_upcoming.py` stamps `model_version` with the HEAD commit sha at run time; weekly "predictions and grading" commits change HEAD, so the track record fragments into a new "version" every week with an unchanged model. | Version must derive from the model artifacts (hash of `models/torch/*` + `models/*.json`), not from HEAD. Existing records get re-keyed by a one-time migration. |
| Every deployed model trains only on fights before 2021-01-01: 2021–2023 was reserved for validation and 2024+ for the (spent) test. The model predicting today's cards has never seen the ~2,900 most recent fights. | Deployment must refit the selected configuration on all data through the latest event (SP1 decides the refit recipe on the harness; SP4 ships it). |
| The Kaggle mirror refreshes irregularly (ends 2026-08-08 today) while a daily-refreshed ufcstats scrape on GitHub carries the same fighter ids. | A secondary daily source can fill the gap between Kaggle versions so grading, retraining, and fighter form stay at most a day stale (SP2). |
| Verified 2026-09-06: the open `ehan03/jds-mma-data` snapshot (MIT, through 2024-08) includes `Bet MMA/late_replacements.csv` (fighter, bout, notice days; ~1,000 rows) and `missed_weights.csv` (~260 rows), plus Tapology weigh-in and rehydration weights, all joinable to ufcstats ids via its mapping tables. Wikipedia event pages carry the same facts in templated prose from ~2009 (withdrawals/replacements) and ~2013 (missed weight), so they can extend the snapshot forward. Official rankings history 2013→now exists as a weekly CC0 Kaggle dataset (names only); the UFC switched to an Elo-based ranking on 2026-06-20. | Short-notice and missed-weight features come from the snapshot first and a Wikipedia parser second; rankings need name matching and a regime flag after June 2026. |
| The raw `r_weight`/`b_weight` columns are per-fighter profile values (0 fighters have more than one distinct value across their fights), not weigh-in weights. | Not usable; as-of-scrape leak. |
| Head/body/leg, distance/clinch/ground, and referee are populated for 99.7% of fights in the current raw file; the new `round.csv` adds them per round. | Rich in-fight signal is available without any external source. |
| Tapology ToS forbids scraping; Sherdog ToS forbids aggregating content elsewhere; Fight Matrix data is now sold through an enterprise API. | Use the static open snapshot (`ehan03/jds-mma-data`, MIT) for external data; no scraping of those sites. |
| Repo lives on an iCloud-synced folder; torch runs have stalled for minutes on page-ins and `.git` has been corrupted twice; five byte-identical `* 2.*` sync duplicates are sitting untracked. | Training runs use a scratch venv on local disk. Moving the repo off iCloud is recommended (user action). Duplicates are deleted in SP0. |

## 3. Targets and outputs

The predictor produces, for one fight (A vs B, R scheduled rounds), a full
outcome distribution:

- **Joint outcome**: P(winner ∈ {A, B} × method ∈ {KO/TKO, submission,
  decision} × round ∈ {1, 2, 3, 4, 5}) with round only defined for finishes
  and rounds 4–5 only for five-round fights. Draws and no-contests are not
  predicted (as today).
- **Marginals** derived from the joint: P(A wins), P(method), P(finish
  round), P(goes the distance), P(A by KO in R1), etc.
- **Uncertainty**: for every reported probability, an interval from the
  spread across model members (see §6), separate from fight randomness.

Method label mapping is unchanged (`labels.py`): KO/TKO includes doctor
stoppages; all decision types collapse to "decision".

## 4. Sub-projects

The work decomposes into five sub-projects. Each gets its own implementation
plan under `docs/superpowers/plans/`, is built on its own branch (SP0 and SP1
sequentially; SP2 and SP3 may run in parallel worktrees once SP1 lands), and
is merged only when its acceptance criteria hold.

```
SP0 repair ─► SP1 eval harness ─┬─► SP2 data + features v3 ─┐
                                └─► SP3 fight simulator ─────┴─► SP4 ship + prospective
```

### SP0 — Repair and foundations

Goal: the pipeline ingests the new data and the prospective loop is trustworthy.

- **New-schema ingestion.** `download_data.py` / `refresh_data.py` fetch the
  new file set. `dataset.py` gains a schema adapter that builds the existing
  `fighters`, `fights`, `fight_stats` tables from the new files (same output
  columns and dtypes, so every downstream test keeps passing) plus a new
  `round_stats` table (one row per fighter per round: sig/total strikes by
  target and position, kd, td, sub attempts, control). Old-schema builders are
  removed once the new path reproduces the old tables on the overlapping
  fights; a reconciliation script reports fight-count, id, and label
  differences between old and new sources and must be reviewed before the
  switch is committed.
- **Truncation-invariance test** extended to `round_stats`-derived columns.
- **Model version fix.** `model_version` = short sha256 over the sorted bytes
  of the deployed model artifacts. One-time migration re-keys existing
  prediction files and `track_record.json`; the weekly Action then grades the
  78 pending fights.
- **Housekeeping.** Delete the five iCloud `* 2.*` duplicates; document the
  local-disk scratch-venv procedure in the README's development section;
  recommend (not perform) moving the repo off iCloud.
- **Acceptance:** full suite green; `refresh_data.py` reports newer data;
  rebuilt `fights.parquet` covers through 2026-08-08; `track_record.json`
  shows one version per real model and graded fights > 0 after the next
  Action run (or a local grading run).

### SP1 — Evaluation harness v2

Goal: one function that scores any candidate model honestly enough to detect
0.003-scale effects, with the acceptance bar written down before any
candidate is run.

- **Split.** Expanding-window walk-forward: for each fold year Y in
  2018…2025, train on all decisive fights dated before Y-01-01, evaluate on
  fights in Y. Hyperparameters and early stopping use the last training year
  as an inner validation slice (never the fold year). Pooled metrics over all
  fold years (~3,900 fights) are the headline; per-fold metrics are reported
  to catch regressions concentrated in one era. 2026 fights (partial year)
  are appended to the 2025 fold. The spent 2024+ test set is folded in; the
  **prospective track record is the only remaining true holdout**, and the
  README says so.
- **Metrics.** Winner: log-loss (primary), accuracy, Brier, ECE. Joint
  outcome: log-loss over the outcome classes defined in §3 (primary for the
  simulator), plus marginal method macro-F1 and finish-round macro-F1 for
  continuity with v1 tables. Slices reported for every candidate: debut
  involved, women's bouts, five-round fights, fights with any external-data
  feature missing (post-snapshot debutants), fold year.
- **Noise floor.** Measured once: the incumbent architecture retrained with
  three disjoint seed sets through the full harness; the standard deviation
  of pooled winner log-loss across them is `σ_seed`. Cached alongside the
  harness output.
- **Pre-registered bars** (fixed in this spec, applied mechanically):
  - A feature block or model change **ships for winner** only if pooled
    winner log-loss improves by more than `max(0.003, 2·σ_seed)` and no fold
    year worsens by more than 0.01.
  - The simulator **ships as the joint predictor** only if joint-outcome
    log-loss beats the composed v1 baseline (v1 winner × v1 method × v1 round
    probabilities) by more than 0.01 **and** its winner marginal is not worse
    than the incumbent winner model by more than `σ_seed`.
  - Because several blocks and configs are tried on the same fights, the
    final shipped configuration is re-scored once with fresh seeds; the
    fresh-seed number is the one reported.
- **Compute.** Measured after SP0: the full suite runs in ~5 s and a 5-seed
  torch retrain in ~8 s from the local-disk venv, so a full walk-forward of
  the torch ensemble takes about a minute. No feature caching is needed.
- **Recency support (harness capability; the experiments run in SP2).**
  Every candidate accepts an optional training-window start (`train_start`,
  e.g. drop pre-2005 fights from model training while Elo and career stats
  still use them) and an optional exponential recency sample weight
  (`half_life_years`, weight = 0.5^(age / half-life), age measured from the
  fold's evaluation start). XGBoost takes weights natively; the torch loss
  becomes a per-sample weighted mean.
- **Refit-strategy experiment (in SP1; decides the deployment recipe).** For
  each fold Y compare (A) the incumbent protocol — train on fights before
  Y−1, early-stop on year Y−1 — with (B) train on everything before Y with
  a fixed training budget (the median best iteration / best epoch from the
  earlier folds) and no early stopping. Both score on Y. (B) uses one more
  year of the freshest data; if it is not worse than (A) by more than
  σ_seed it becomes the deployment recipe: select on the harness, refit on
  everything through the latest event, stamp a new artifact hash. A
  fresh-seed re-score of the shipped recipe (`fresh_seed_rescore` in
  `models/walkforward/refit_decision.json`) confirmed the gate and also
  surfaced the temperature-drift follow-up recorded there, which is
  scheduled into SP2's recency block above rather than re-decided here.
- **Deliverables:** `src/mma/walkforward.py` (fold construction, scoring,
  slices, bar check), `src/mma/candidates.py` (one fit/predict protocol
  wrapping Elo, XGBoost, and the torch ensemble, with config, window, and
  weight options), `scripts/run_walkforward.py` (candidate spec in → JSON
  report out under `models/walkforward/`), the noise-floor artifact, the
  refit-strategy decision, `--refit-through` support in the train scripts so
  the deployed model uses all data through the latest event, and a README
  section replacing the spent-test story with the walk-forward story.
- **Acceptance:** incumbent XGBoost and torch models re-scored through the
  harness with numbers that reproduce the committed 2021–2023 validation
  metrics on those folds; `σ_seed` recorded; unit tests cover fold
  boundaries (no fold-year fight in any training set), slice masks, and the
  bar arithmetic.

### SP2 — Data expansion and features v3

Goal: add signal in evaluated blocks; keep only blocks that clear the SP1 bar.
Also: add the daily-refreshed ufcstats scrape (same fighter ids) as a secondary
source adapter that fills the gap between Kaggle versions, reconciled through
the same `reconcile_sources.py` guard, so processed data is at most a day stale.

All features stay point-in-time by construction (built inside the
chronological accumulators in `history.py`, or joined on dated rows with a
strict `date <` filter) and pass the truncation-invariance test. Every
external feature carries a missingness flag. Blocks are evaluated in this
order, each on top of whatever previously cleared the bar, with negative
results documented in the plan file as before.

1. **In-fight profile block** (from `round_stats` and the strike splits):
   per-fighter expanding rates of head/body/leg and distance/clinch/ground
   output and absorption; knockdowns absorbed per fight; round-1 output share
   and late-round fade (round-over-round change in sig strikes landed and
   absorbed); finish and been-finished rates by round; median time-to-finish;
   KO-loss count and recency ("chin" proxies).
2. **Opponent-adjusted block:** for the core rates (sig landed/absorbed per
   minute, TD landed/allowed, control share) an expanding "versus expectation"
   version: fighter's value minus what their opponents allowed on average
   before that fight; and average opponent quality of wins vs losses.
3. **Rating-trajectory block:** Glicko-2 rating and deviation as features
   (previously rejected as an Elo *replacement*; here tested as *additional*
   columns), average post-fight Elo change over the last 3 and 5 fights,
   peak-minus-current Elo, years since UFC debut, age², age × fight count.
4. **Context block:** referee's historical finish rate and decision rate
   (point-in-time), event location country vs fighter nationality (home
   advantage; nationality from the external snapshot), bonus history
   (performance-bonus count as a "fan-friendly finisher" proxy).
5. **Recency block** (harness capability from SP1): training-window start
   and exponential recency weights over a small grid (no cut / 2000 / 2005 /
   2010 start; half-life ∞ / 8 / 4 / 2 years), judged by the same bar. The
   sport's meta shifts fast enough that this is expected to matter more than
   any single feature. Also tests the fixed budget and temperature taken
   from the most recent k folds instead of the all-fold median (motivated
   by the temperature drift recorded in `models/walkforward/refit_decision.json`).
6. **Short-notice and weigh-in block**: days of notice for late
   replacements and pounds over the limit, from the snapshot's
   `late_replacements.csv` / `missed_weights.csv` (through 2024-08) extended
   forward by a parser over Wikipedia event pages, which the prospective
   pipeline already fetches (templated "withdrew ... was replaced by" and
   "weighed in at N pounds, M over" prose; withdrawals reliable from ~2009,
   missed weight from ~2013). Point-in-time by construction. Notice days
   are often stated only qualitatively on Wikipedia, so the feature is
   binned (≤ 7 / ≤ 30 / full camp / unknown).
7. **Rankings block**: official UFC rank at fight time from the weekly
   rankings-history dataset (2013→now, names only, so matched through the
   fighters table with the same never-guess policy as the prospective
   matcher), with a missingness flag, an unranked-vs-ranked matchup flag,
   and a regime flag for the June 2026 switch to Elo-based rankings.
8. **External snapshot block** (`ehan03/jds-mma-data`, vendored as a
   *derived* compact per-fighter parquet of a few MB, regenerated by a
   script from the downloaded snapshot, never the raw 80 MB): pre-UFC record
   at UFC debut (wins, losses, finishes, finish-loss count), days since pro
   debut, pre-UFC opponent quality (mean opponent record), regional
   promotion of origin; gym id → gym win-rate accumulator (only if the
   snapshot dates gym membership; a static current-gym column is as-of-scrape
   and is rejected); nationality. Entity matching uses the snapshot's
   ufcstats-id mapping; unmatched fighters are flagged, and the post-snapshot
   debutant slice from SP1 monitors degradation over time.

Feature selection is by block, not by column, to limit multiple-comparison
optimism; within a block, a column is dropped only if a documented ablation
shows it hurts.

- **Acceptance:** each block's harness report committed; the shipped feature
  table is the union of blocks that cleared the bar; `features.parquet`
  rebuilt; truncation test green over every column; README feature section
  updated.

**Status: SP2 DONE (2026-09-08), plan
`docs/superpowers/plans/2026-09-07-sp2-features-v3.md`.** All eight blocks
above were built and measured; **one shipped**. The feature table is
`base,external` (11,238 x 54), the deployed budget was re-derived on it
(`models/walkforward/refit_decision_v3.json`: XGB 105/61/75 trees, torch 10
epochs at temperature 1.07) and the models redeployed (hash `b617b96dae45`).
Fresh-seed re-score of the shipped set against a bit-exact paired incumbent:
pooled 0.6516 -> 0.6473, **delta -0.0043**, worst fold +0.0006, ships.

| block | torch pooled | Δ vs incumbent | ships |
|---|---|---|---|
| external | 0.6476 | **-0.0034** | **yes** |
| recency | 0.6499 | -0.0011 | no |
| notice | 0.6472 | -0.0004 | no |
| trajectory | 0.6472 | -0.0004 | no |
| context | 0.6474 | -0.0002 | no |
| rankings | 0.6487 | +0.0011 | no |
| in_fight | 0.6528 | +0.0018 | no |
| opponent_adjusted | 0.6532 | +0.0022 | no |

**What seven rejections imply for SP3.** The one block that paid is the one
whose information is *not* in the fight table: pre-UFC career, from an outside
source. Every block that recombines what the box score already carries --
per-round and strike-target profile, opponent-adjusted rates, Glicko-2 and
Elo dynamics, official rankings -- lands within a third of the bar or worse,
and mostly in the same shape: XGBoost neutral-to-better, the neural net (the
scorer that decides) worse, because a tree can ignore a correlated column
that is NaN on most rows while the MLP must impute and spend capacity on it.
The box-score data is close to tapped out. **SP3's value therefore has to come
from the paradigm, not from new columns**: a coherent joint distribution over
(winner, method, round) produced by simulating the fight, which is a different
object from three independent marginal heads and is judged on joint log-loss,
not on whether one more feature moves the winner marginal. E1 (hazard +
decision on v1 features) is the load-bearing experiment for exactly this
reason -- it isolates the paradigm from the data -- and E4's latent in-fight
state is the one remaining route to information the current columns do not
express, since it models the *sequence* rather than another aggregate of it.

**Two methodological results SP3 and SP4 inherit.** (1) A selection leak was
found and removed: per-corner "data missing" flags encoded future career
length (the unmapped corner lost 76% of its fights, 90% in the debut slice),
and the shipped form keeps such flags in the TABLE and out of both model
matrices, verified by a data-level property (over the rows where exactly one
corner is unmapped, the block's columns take exactly one distinct value
tuple) rather than by a rule about column names. Any SP3 feature joined from
an external source inherits that check. (2) A budget belongs to a FEATURE
TABLE, not to a recipe -- the v1 table's 14 epochs at temperature 1.1 did not
carry over -- so a simulator shipping on a new table re-derives its own
through `scripts/refit_decision.py --reports <set>`.

**Carried forward from SP2** (full list in the plan's Completion notes): the
external snapshot is static and its coverage decays (0.534 of 2026 rows
already unmapped), so SP4 needs a refreshable source or the only shipped
block drifts to neutral; the secondary daily source has four prerequisites
before its writes can be enabled (fighter-table adaptation, the
`make_dataset.py` rebuild collision, a provenance column, `round_stats`);
`mma.wiki_cards.parse_background` is built and fixture-tested but deliberately
unwired from `prospective.predict_event`; three rankings name-match misses
would close by extending `mma.prospective.fold_accents`; and one row carries a
`dob` error giving age 4.6, harmless to a linear age term but not to a
quadratic one, which SP3's hazard model may well want.

#### SP2.1 — was it the features or the architecture? (pre-registered, 2026-09-08)

The seven rejections above share a shape — XGBoost neutral-to-better, the
deployed MLP worse — and that shape has two readings. One is the information
reading used above: the box score is tapped out. The other is a **capacity**
reading: the MLP's trunk was sized when the table had 46 columns, and a
fixed-width net must spend capacity on every member of a correlated group
where a tree picks one per split. The second reading would mean the SP2
rejections measured the architecture, not the features — so it was written
down as a hypothesis, with arms, a 25-configuration search space and a
mechanical decision rule fixed **before anything ran**
(`docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md`,
`models/walkforward/sp2_1_decision.json`).

Four rejected blocks (`trajectory`, `notice`, `context`,
`opponent_adjusted`), each verified to reproduce its committed SP2 number
exactly, were combined into an 87-column table and searched. The same 25
configurations were searched over the shipped 54-column table as a **control
arm**, without which a gain would have been misattributed to the features.

| arm | features | architecture | pooled | Δ vs deployed 0.6476 | ships |
|---|---|---|---|---|---|
| A1 | combined | deployed | 0.6475 | −0.0001 | no |
| C (control) | shipped | best of 25 | 0.6463 | −0.0013 | no |
| T (treatment) | combined | best of 25 | 0.6464 | −0.0012 | no |

Neither cleared the 0.003 bar; T − C = +0.0001, inside σ_seed = 0.000346. The
rule's fourth branch applied: **record and revert.** Nothing shipped, the
blocks were unregistered again, and the deployed hash was still
`b617b96dae45`. (SP2.2 later re-registered all four blocks — not because
they cleared anything, but because the blend that cleared was scored on the
table containing them; see the SP2.2 subsection.)

**What this changes for SP3.** It removes the alternative explanation, and
therefore strengthens rather than merely repeats the SP2 conclusion. Before
SP2.1 the claim was "these features did not help the deployed model"; after
it the claim is "**these features do not help a model given a wider, better
regularised trunk and 25 attempts to find one — and the deployed trunk is not
leaving anything on the table either**". The box-score feature space and the
MLP architecture are *both* at their ceiling. SP3's value must therefore come
from the paradigm — a coherent joint distribution over (winner, method,
round), judged on joint log-loss — and not from new columns or a bigger net.
E1 (hazard + decision on v1 features) is the load-bearing experiment for
exactly that reason, and E4's latent in-fight state remains the one route to
information the current columns do not express, because it models the
*sequence* rather than another aggregate of it. Concretely: SP3 should not
spend effort widening the trunk, and should not re-open the four archived
blocks (their registrations are commented out in `mma.feature_blocks`, their
accumulators still live) unless a *new* source or a *new* model class changes
what they are worth.

**A live alternative SP4 must weigh before deciding what gets deployed.** The
strongest number measured anywhere in SP2.1 is not an arm at all: an
equal-weight **blend of the XGB and torch winner probabilities** scores 0.6459
on the shipped table and 0.6436 on the combined one, against the deployed
0.6476, with every evaluation slice improving
(`models/walkforward/blend_a{0,1}_xgb_torch.json`). It is deliberately **not**
a ship candidate here — a blend is not one of SP2.1's five pre-registered
arms, and adopting it on these numbers would be the after-the-fact
search-widening the pre-registration forbids. But SP4 chooses what actually
serves, and by then it will be choosing among a simulator, the deployed
ensemble and a blend. Two things must be settled first: the blend inherits
roughly half of XGBoost's seed noise, newly measured at σ ≈ 0.0009–0.0013
(about 3× the torch ensemble's), so its confirmation needs several XGB seeds
rather than one; and switching or widening the deployed scorer touches nine
places listed in `open_questions[1]` of the decision file, of which
`versioning.MODEL_ARTIFACT_GLOBS` is the sharp one — it currently hashes only
the torch artifacts, so a blended deployment would stop hashing half of what
serves predictions. **Resolved by SP2.2 below**: the blend was pre-registered
as its own experiment, cleared its bar at two seed sets, and shipped; all
nine places were updated, `MODEL_ARTIFACT_GLOBS` included.

**One methodological result SP3 and SP4 inherit.** XGBoost's own seed noise
was measured for the first time here (`models/walkforward/noise_floor_xgb.json`):
σ ≈ 0.00087–0.00125 for a single fit, against 0.000346 for the 5-seed torch
ensemble. A combined-table XGB result of −0.0039 with every slice improving —
comfortably over the bar — averaged −0.0024 once re-run at three fresh seeds.
No SP2 rejection rests on a single-seed XGB screen alone (the screen never
dropped a block without a torch run), but any future screen on a single
boosted fit must use ≥3 model seeds; `--model-seed` and
`scripts/noise_floor.py --candidate xgb` exist for that.

#### SP2.2 — the blend ships (pre-registered, 2026-09-08)

SP2.1's strongest number was a diagnostic it could not act on. SP2.2 turned
it into a pre-registered experiment with two candidates, its own noise floor
and a mechanical decision rule fixed before anything ran
(`docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md`,
`models/walkforward/sp2_2_decision.json`). **It shipped.**

**What is deployed.** The equal-weight (0.5/0.5) average of a **5-seed
XGBoost ensemble** and the **5-seed torch ensemble**, temperature-scaled
*after* averaging (deployed **T = 0.80**, the median of the harness's eight
per-fold fitted temperatures), on the S1 table
`base,external,trajectory,notice,context,opponent_adjusted` (11,238 x 87)
with `external_missing`, `same_country`, `notice_unknown`, `home_country_a`
and `home_country_b` in the table and out of both model matrices. Deployment
budgets re-derived on S1 (`refit_decision_b1.json`): torch 6 epochs at
temperature 1.15, XGB 109/73/71 trees per seed. **Hash `5aa33460ef40`**
(was `b617b96dae45`), and the hash now covers the XGBoost artifacts too,
which it did not before.

| candidate | table | pooled | Δ vs deployed 0.6476 | fresh seeds 5–9 | ships |
|---|---|---|---|---|---|
| B0 | `base,external` | 0.6453 | −0.0023 | −0.0022 | no |
| **B1** | S1 (87 col) | **0.6437** | **−0.0039** | **−0.0037** | **yes** |

σ_blend = **0.0001** across three disjoint seed sets on both members
(`noise_floor_blend.json`) — the quietest scorer measured in this repo,
against 0.000346 for the torch ensemble and 0.00087–0.00125 for a single
XGBoost fit. The bar stayed 0.003. B1 clears at seeds 0–4 and again at 5–9
against its own paired incumbent, worst fold +0.0061 / +0.0064 against the
0.01 tolerance, all four slices improving at seeds 0–4. It beats both of its
own members on the same table (XGB 0.6462, torch 0.6475).

**The methodological weakness, recorded as one.** The pre-registered ECE gate
(rule 4) **failed as written** — B1's pooled ECE 0.0124 against the single
incumbent value 0.0088 the rule named — and was **amended after the numbers
were seen**. The evidence for amending: the incumbent's own pooled ECE spans
0.0088–0.0160 (sd 0.0038) across three disjoint seed sets, so the 0.0036 gap
is inside one sd of the metric doing the testing and 0.0088 was the
incumbent's *best* of three; the sign of the gap reverses at 5, 15 and 20
bins (10 is only the ECE helper's default); and the reliability curves show
a couple of bins each way rather than a slope. The replacement is the
project's standard form — mean across three disjoint seed sets, 2σ
tolerance — and B1 passes it (+0.0017 against ≈0.0056). **Amending a
pre-registered gate post hoc is exactly what this protocol exists to
prevent.** The mitigations are transparency (the amendment is dated and
labelled post-hoc inside the pre-registration, the original preserved
verbatim), the use of a standard form rather than a bespoke threshold, and
the fact that the log-loss bar B1 actually cleared was never touched. An
isotonic remediation was tried and failed decisively (0.7049 against 0.6437,
ECE 0.0289 against 0.0124), which is evidence that temperature scaling was
never the binding constraint.

**The four blocks, honestly sized.** SP2.1 declared `trajectory`, `notice`,
`context` and `opponent_adjusted` dead, and that verdict stands *for the
MLP*: the torch arm reads 0.6475 on S1 against 0.6476 on S0. The XGBoost arm
reads 0.6462 against 0.6490 — they were never dead to the trees. But B1
against B0, the same blend on the previous table, is only **−0.0016**, which
clears no bar of its own. The blend is what cleared; the blocks ride along
with it. They also carry maintenance: `external`, `notice` and `context`'s
nationality half all come from the static `ehan03/jds-mma-data` snapshot
(UFC coverage ends 2024-12-14), where `external_missing` is already 0.534 of
2026 rows and `notice` is unknown on 100% of 2025–2026 rows.

**Two silent deployment bugs found and fixed**, both worth SP4's attention as
a class: (1) `scripts/build_features.py` defaulted to `base`-only blocks
while the weekly Action invokes it bare, so the next refresh would have
rebuilt the *shipped* table with five blocks missing and retrained on it with
nothing crashing; the default is now the sidecar beside the committed table.
(2) A served matchup in an unseen weight class raised `XGBoostError` that
`prospective.predict_fight` did not catch, which would have killed a whole
card's predictions; `align_to_booster` now maps unseen categories to missing,
and the xgboost floor moved to `>=3.0`.

**What this changes for SP3.** SP2.1's conclusion was that the box-score
feature space and the MLP architecture were *both* at their ceiling, and it
used that to argue SP3's value must come from the paradigm rather than from
columns or capacity. SP2.2 does not overturn that, but it adds a term the
earlier reasoning missed: **a *combination of model families* still had
headroom that neither family had alone.** −0.0039 was available from two
models this repo already had, without a single new column and without a
wider net; the same four blocks that were worthless to the MLP were worth
−0.0028 to the trees. The ceiling SP2.1 measured was a ceiling *per model
family*, not a ceiling on the prediction.

Concretely, for SP3:

- **The simulator must be evaluated as a potential blend member, not only as
  a torch replacement.** Its E1–E4 arms are still judged by the SP1 bar on
  the harness, but the comparison that decides deployment is now
  *blend-with-simulator against the deployed blend*, and a simulator that
  loses to the torch member head-to-head can still earn a place if it
  decorrelates. `mma.candidates.BlendCandidate` already implements the
  construction and `mma.blend` holds its arithmetic, so the harness can score
  such a candidate without new machinery — but the pre-registration for it
  must fix the member set and the weights **in advance**, exactly as SP2.2
  did, and must not let a three-model blend become a post-hoc search.
- **Decorrelation is now a first-class property to look for.** The reason the
  blend works is that the two members disagree in a useful direction; a
  simulator that produces a coherent joint distribution over (winner, method,
  round) is a plausibly *more* different object than XGBoost is from the MLP.
- **The blend's method and round heads are worse than the torch member's**
  (macro-F1 0.3385 against 0.3728, 0.2793 against 0.3068) because averaging
  pulls the class-weighted neural heads toward the trees' majority-class
  behaviour. SP3 is judged on joint log-loss, and that regression is the most
  concrete thing a simulator could fix.
- The blend's fixed T = 0.80 is an extrapolation the walk-forward never
  validated (the harness fits one per fold; a refit-mode blend report cannot
  exist because protocol B trains on the inner-validation year), so any
  SP3 candidate that joins the blend inherits an open calibration question.

**What this changes for SP4.** The promotion gate in `scripts/roll_window.py`
scores a torch-only candidate against a torch-only incumbent. Since SP2.2 it
detects a blended incumbent and **aborts** rather than reporting a number
about half a model — which, with the pre-existing in-sample abort for
refit-through-latest incumbents, leaves the promotion path fully inert. SP4
must resolve it, and the spec's existing preference (move the gate onto the
walk-forward harness) resolves both aborts at once.

### SP3 — Fight simulator

Goal: a Monte Carlo simulator whose empirical outcome distribution is the
prediction, meeting the SP1 simulator bar.

**Model.** Two learned components on pre-fight features (the v3 table plus
round index and scheduled rounds), each symmetrized by running both corner
orderings and averaging:

- **Hazard model.** Discrete-time survival: one training row per
  (fight, round actually fought), label ∈ {A KO/TKO, A submission, B KO/TKO,
  B submission, round completes}. Rounds never reached are simply absent,
  which handles censoring correctly. Inputs: pre-fight features, round
  number, scheduled rounds, and (v1) nothing about the fight so far.
- **Decision model.** P(A wins on the scorecards | features, scheduled
  rounds), trained on fights that went the distance (draws excluded).

Base learners: XGBoost multiclass and the existing torch trunk with a 5-way
hazard head and a decision head (multi-task on the same rows), each as a
seed ensemble with per-fold temperature calibration. The harness decides
which base learner (or the average) ships.

**Simulation.** For each fight and each of N runs (default 10,000): draw a
model member (ensemble seed, or bootstrap of the hazard rows for XGBoost),
then for r = 1…R draw the round outcome from the member's hazard
distribution; if all rounds complete, draw the winner from the member's
decision probability. Two randomness layers are recorded separately: the
across-member spread of each probability is reported as the uncertainty
interval; the within-member outcome frequencies are the prediction. N is
chosen so Monte Carlo standard error on P(A wins) is below 0.005.

**Experiments inside SP3, each judged by the harness:**

- E1: hazard + decision on v1 features (isolates paradigm from data).
- E2: same on v3 features.
- E3: Bayesian hierarchical hazard model (PyMC; fighter random effects on
  finish propensity and durability, weakly informative priors) as an
  alternative member sampler — posterior draws replace ensemble seeds. Kept
  only if it beats E2 or materially improves the debut slice.
- E4: latent in-fight state — simulate per-round strike/takedown counts from
  a generative model fit on `round_stats`, and condition later rounds'
  hazards on cumulative simulated damage (a Markov-chain fight model in the
  style of Holmes, McHale & Żychaluk 2023). Highest ceiling, highest cost;
  attempted only if E2 ships, and gated separately.

**Fallback if the simulator wins on joint outcome but loses the winner
marginal:** a documented hybrid where the direct winner model's probability
is imposed and the simulator supplies P(method, round | winner) by
re-weighting simulated runs. This is a defined branch, not an ad-hoc patch.

- **Acceptance:** SP3 bar from SP1 met on fresh seeds; `src/mma/simulator.py`
  with a pure function `simulate(features_row, n_runs, members) -> OutcomeDistribution`;
  tests for label construction (censoring), round masking for three-round
  fights, symmetry under corner swap, Monte Carlo error bound, and that the
  winner marginal equals 1 minus the opponent's; harness report committed.

### SP4 — Ship, app, prospective loop

Goal: the winning predictor is what the app shows, the weekly Action runs,
and the track record grades.

- The deployed model is refit on all data through the latest event with the
  recipe SP1 selected; the weekly Action does the same on every refresh.
  **Since SP2.2 that means both members of the blend**, each with its own
  budget from `refit_decision_b1.json`, plus the blend's own fixed
  post-average temperature — which no refit decision can derive, and which
  SP4 should therefore validate rather than inherit.
- `inference.py` exposes the outcome distribution; `display_priors.json`
  logic is retired if the simulator's marginals are already base-rate
  consistent (verified on the harness), otherwise retained for the marginals.
- App: probability of win with interval, an outcome table ("A by KO/TKO in
  R2: 11%"), P(distance), and the "simulated N times" framing; explanations
  move to the hazard model's top factors.
- Prospective records store the full joint distribution; grading scores joint
  log-loss and marginals; `roll_window.py`'s promotion gate switches to the
  SP1 bar on the walk-forward metric — **now required rather than preferred**:
  since SP2.2 the gate aborts on a blended incumbent as well as on an
  in-sample refit one, so the promotion path is inert until this move
  happens. The weekly Action rebuilds `round_stats` and the derived external
  table.
- README rewritten around the new ladder (v1 heads → simulator), the
  walk-forward results, and the honest "market still wins / market gap"
  section carried forward.
- **Acceptance:** suite green; app boots headless; one weekly Action run
  green end to end; a prospective prediction file produced by the new model
  with joint distributions.

## 5. Evaluation protocol summary (locked for this pass)

- Expanding-window walk-forward, fold years 2018–2025(+2026 partial).
- Primary metrics: winner log-loss; joint-outcome log-loss.
- Bars: winner Δ > max(0.003, 2·σ_seed) with no fold worse by 0.01; simulator
  joint Δ > 0.01 with winner marginal within σ_seed of incumbent.
- Final numbers re-scored with fresh seeds; slices always reported.
- Market comparison remains evaluation-only on the 2021+ odds-matched subset
  and is re-run for the shipped model.
- Prospective track record is the only true holdout and is never used for
  selection.

## 6. Uncertainty semantics

Two sources are kept distinct everywhere they are shown:

- **Model uncertainty** — spread across ensemble members / posterior draws.
  Reported as an interval around every probability. Wide intervals are
  expected for debutants and post-snapshot fighters.
- **Outcome randomness** — the fight itself. Represented by the outcome
  distribution; not an interval.

The betting layer (later) needs both; the app shows both.

## 7. Tooling used in this pass

- **Subagents in parallel worktrees** for SP2 and SP3 once SP1 lands, with
  the two-stage review convention from earlier phases.
- **PyMC** (available skill) for the SP3 E3 Bayesian hazard variant.
- **Research lookup / web search** already used for the data-source and
  literature sweep; re-used for any specific method question (Glicko-2
  parameters, discrete-time survival with XGBoost).
- **Context7** for library documentation (xgboost multiclass with sample
  weights, torch, pymc) instead of memory.
- Prediction-market and Kaggle-odds connectors are deferred to the betting
  spec.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| New Kaggle schema differs from what the research agent saw, or the 11,441 fight rows include non-UFC or upcoming fights | SP0 reconciliation script and a hard review step before switching sources |
| Walk-forward × ensembles is slow on this machine | Feature caching per fold, small search spaces, local-disk venv, XGBoost first for screening, torch only for finalists |
| External snapshot entity matching is incomplete | Match by ufcstats id only; report match rate; missingness flags; dedicated slice |
| Simulator winner marginal underperforms the direct model | Defined hybrid fallback; simulator still ships for method/round |
| Multiple-comparison optimism across blocks and experiments | Block-level selection, fixed bars, fresh-seed final scoring |
| iCloud corrupts `.git` or stalls training again | Scratch venv on local disk; `git status` verification before every commit; recommend moving the repo |

## 9. Out of scope (deferred to later specs)

- Value-betting layer: devigging, edge and Kelly sizing, backtest against the
  Kaggle daily odds dataset (moneyline + method props), live odds API,
  prediction-market connectors. Odds remain evaluation-only in this pass.
- Judge-level scorecard data (UFC-DataLab, SCORE Network) and judge-effect
  models for decision-bound fights.
- Any scraping of Sherdog, Tapology, or Fight Matrix; paid data.
- Six-class method targets, live in-fight prediction.
