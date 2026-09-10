# MMA Fight Prediction

![Weekly data refresh](https://github.com/dylanmryan/mma-prediction/actions/workflows/refresh-data.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**[▶ Try the live app](https://mma-prediction-lxqvmgheqvzccdpyord3q9.streamlit.app/)** — pick any two UFC fighters, get calibrated win probabilities with uncertainty.

Predicting UFC fight winners, method of victory, and finish round —
an Elo rating system, a gradient-boosted ensemble, a multi-task
neural-network ensemble, and the calibrated **blend of the two that now
serves**, each evaluated honestly by expanding-window walk-forward over
2018–2026, plus an interactive Streamlit matchup explorer.

**Highlights**

- **Point-in-time discipline, machine-verified**: every feature is built only
  from data available before each fight; a truncation-invariance test proves
  no feature can see the future.
- **Baseline ladder**: coin flip → Elo (0.553 acc, 0.683 log-loss) →
  5-seed XGBoost ensemble (0.616, 0.646) → 5-seed calibrated neural
  ensemble (0.622, 0.648) → **the calibrated blend of those two, which is
  what ships** (0.622, 0.644) on 4,804 walk-forward fights, each scored by a
  model that had never seen its year.
- **Features earned, not assumed**: eight feature blocks measured against a
  pre-registered bar. Exactly one ever cleared it on its own; four more ship
  only because a *second model family* turned out to use what the neural net
  could not, and three ship in no form at all ([Features](#features)). A
  selection leak in per-corner missingness flags was found and removed along
  the way.
- **Uncertainty done properly**: spread across the five per-seed blends
  (booster *i* paired with net *i*), MC dropout from the neural member, a
  temperature fitted on held-out data *after* the blend average. The method
  and round probabilities used to be rescaled to historical base rates before
  display; since the simulator ships they are marginals of one coherent joint
  distribution and are shown raw, with the calibration measured weekly
  (`models/display_calibration.json`) rather than corrected.
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
`scripts/build_features.py --blocks`). The table on disk is
**11,238 fights × 87 columns**, built from
`base,external,trajectory,notice,context,opponent_adjusted`
(`data/processed/features_blocks.json` records which blocks produced it, and the
serving path reads the same sidecar so a trained column cannot go missing at
prediction time). Run bare, `scripts/build_features.py` rebuilds *those* blocks
— it reads the sidecar rather than defaulting to `base`, which is what the
weekly Action depends on.

Only two of those six blocks ever cleared the pre-registered bar as a block.
`base` is the v1 contract; `external` cleared on its own; the other four were
measured, rejected, re-tested and rejected again, and ship only as part of the
feature table the **blend** was scored on — see
[the blend](#two-model-families-one-scorer--and-the-gate-that-had-to-be-amended)
below, which is where the honest accounting for that lives.

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
- **`trajectory`** — rating *dynamics* on top of the base block's rating
  *levels*: Glicko-2 deviation and volatility, recent Elo momentum, drawdown
  from a fighter's own peak, tenure, and two age interactions.
- **`notice`** — short-notice replacement and missed weight, from the same
  `ehan03/jds-mma-data` snapshot. Its bout list stops at 2024-12-14, so
  `notice_unknown` is the served state for every future fight.
- **`context`** — the fight's setting rather than either record: referee
  tendency, home advantage, and post-fight bonus history.
- **`opponent_adjusted`** — each core rate priced against what that fight's
  opponent had historically *allowed*, plus mean opponent Elo in wins and
  losses.

The last four never cleared the bar as blocks and are not claimed to have.
They are in the table because they are worth something **to the XGBoost
member of the deployed blend** and nothing to the neural one; the size of
that contribution is measured and reported below.

Five columns live in the table and are held out of **both** model matrices
(`mma.tensors.DROPPED`, `mma.models.xgb.MODEL_EXCLUDED`) as leak guards:
`external_missing`, `same_country`, `notice_unknown`, `home_country_a` and
`home_country_b`. The first is the coverage-selection leak described below;
the `home_country` pair is the same leak in a second channel.

### Block results

Eight blocks were measured against the bar; **one cleared it**. Every
rejected block's harness report is committed under `models/walkforward/` —
negative results are deliverables here, not deleted branches. Pooled winner
log-loss, torch deciding (the scorer deployed at the time), against that
block's incumbent:

| Block | What it added | XGB | torch | Δ vs incumbent | Ships? |
|---|---|---|---|---|---|
| `external` | pre-UFC career and origin | 0.6506 | **0.6476** | **−0.0034** | **cleared the bar** |
| `recency` | training window + recency weights | 0.6525 | 0.6499 | −0.0011 | no |
| `notice` | short-notice replacement, missed weight | 0.6501 | 0.6472 | −0.0004 | in the table (SP2.2) |
| `trajectory` | Glicko-2, Elo momentum, peak-minus-current | 0.6486 | 0.6472 | −0.0004 | in the table (SP2.2) |
| `context` | referee rates, home country, bonus history | 0.6499 | 0.6474 | −0.0002 | in the table (SP2.2) |
| `rankings` | official weekly divisional rank | 0.6499 | 0.6487 | +0.0011 | no |
| `in_fight` | per-round and strike-target profile | 0.6540 | 0.6528 | +0.0018 | no |
| `opponent_adjusted` | rates versus what opponents allowed | 0.6519 | 0.6532 | +0.0022 | in the table (SP2.2) |

Seven of eight fail, and mostly in the same shape: the tree ensemble is
roughly neutral or slightly better while the neural net — the scorer that
decides — is worse. A tree can ignore a correlated column that is NaN on most
rows; the MLP has to impute a median for it on every one of those rows and
spend capacity on the result. `in_fight`, `opponent_adjusted`, `trajectory`
and `rankings` are all recombinations of information the base table already
carries. The box score, in other words, is close to tapped out.

That shape — *the tree is fine with these columns and the deployed net is
not* — was read twice as a reason to reject, and the second reading turned
out to be the interesting one. It is what the blend below exploits.

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

### Two model families, one scorer — and the gate that had to be amended

That blend got its own pre-registered experiment
(`docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md`,
`models/walkforward/sp2_2_decision.json`) and **it ships**. The deployed
scorer is no longer the neural ensemble on its own: it is the equal-weight
(0.5/0.5) average of a **5-seed XGBoost ensemble** and the **5-seed neural
ensemble**, temperature-scaled *after* averaging, on the 87-column table.

**The design was fixed before anything ran.** Two candidates and nothing
else: **B0**, the blend on the shipped `base,external` table, and **B1**, the
identical construction on the 87-column table SP2.1 had built and reverted.
Fitted or searched blend weights, three-model blends, stacking and per-slice
weighting were all explicitly out of scope — the 2026-07-15 model-v2 session
had already found a fitted logistic stack *worse* than either model alone,
and fitting weights on the same folds that judge the candidate is the
selection failure this project keeps getting burned by. The 0.3 / 0.5 / 0.7
weight sweep (0.6456 / 0.6453 / 0.6462 on S0) exists only to show the surface
is flat near 0.5; the shipped weight is 0.5 whichever cell won.

**The blend needed a noise floor of its own, and got one first.** Built three
times from disjoint seed sets on *both* members — {0–4}, {5–9}, {10–14} — it
scores 0.6453 / 0.6451 / 0.6452, i.e. **σ_blend = 0.0001**: the most stable
scorer measured anywhere in this repo, against 0.00035 for the 5-seed neural
ensemble and 0.00087–0.00125 for a single XGBoost fit. Averaging two
independent families cancels seed noise from both. The bar stayed
`max(0.003, 2σ)` = **0.003**, fixed by that measurement before any candidate
was scored against it.

| Candidate | Feature table | Pooled | Δ vs deployed 0.6476 | Fresh seeds 5–9 | Ships? |
|---|---|---|---|---|---|
| B0 | `base,external` | 0.6453 | −0.0023 | −0.0022 | no |
| **B1** | 87-column | **0.6437** | **−0.0039** | **−0.0037** | **yes** |

B1 clears at seeds 0–4 and clears again on the fresh-seed re-score at seeds
5–9 against its own paired incumbent — the confirmation step that caught
SP2.1's false positive. Worst fold is +0.0061 and +0.0064 against the 0.01
fold-regression tolerance. All four evaluation slices improve at seeds 0–4
(debut −0.0127, `external_missing` −0.0150, womens −0.0092, five_round
−0.0016); on fresh seeds three of four improve and five_round is +0.0012.
B0 never clears, so the pre-registered rule for choosing *between* two
winners never fired.

The blend also beats both of its own members on the same table: XGBoost
0.6462, the neural ensemble 0.6475, the blend 0.6437 (all three on the harness
form, so the comparison is like for like; the deployed fixed-temperature blend
reads 0.6432). Calibrating after the
average is doing real work on calibration and nothing on log-loss — the
uncalibrated blend reads 0.6455 with ECE 0.0162 against 0.6453 / 0.0091
calibrated.

#### The weakness: a pre-registered gate was amended after the numbers were seen

This is the least comfortable paragraph in this README, and it is here
because leaving it out would make everything else in the file worth less.

Rule 4 of the pre-registration was an ECE gate: *a shipping candidate must
not have a worse pooled ECE than the incumbent (0.0088).* B1's pooled ECE is
**0.0124** on the harness form, so it **failed that gate as written**. (Every
ECE in this subsection is the harness form's — see "the gate judged a form
that does not ship" below, which is a second, separate weakness.) The rule
reserved that case
for a human call, and the human's call was to **re-specify the gate** — after
seeing the numbers. That is exactly the kind of move this project's whole
protocol exists to prevent, and calling it anything softer than a weakness
would be dishonest.

What the re-specification rested on, all of it measured and committed
(`models/walkforward/noise_floor_ece.json`):

- **The gate compared one number to one number, with no known precision.**
  The incumbent's own pooled ECE across three disjoint seed sets is
  0.0088 / 0.0160 / 0.0101 — mean 0.0116, sd **0.0038**. The 0.0088 the rule
  named was the incumbent's *best* of the three. The 0.0036 gap the gate
  turned on is smaller than one standard deviation of the metric doing the
  testing.
- **The gap does not survive a different binning of the same predictions.**
  At 10 bins B1 is worse by +0.0037; at 5, 15 and 20 bins it is *better*
  (−0.0023, −0.0064, −0.0049). Ten bins is only the default of the ECE
  helper; nothing in the pre-registration justified it.
- **The reliability curves show no systematic miscalibration on either
  side** — a couple of bins over on one side and under on the other, not a
  slope.

The replacement is the form every other bar in this project uses: *mean
pooled ECE across three disjoint seed sets, tolerance 2σ of the pooled
spread*. It demands more than the original (three seed sets, not one), and
B1 passes it — 0.01337 ± 0.00100 against the incumbent's 0.01163 ± 0.00384,
a difference of **+0.0017** against a 2σ tolerance of **≈0.0056**.

The mitigations are mitigations, not a defence: the amendment is written
down, dated, and marked as post-hoc inside the pre-registration itself; the
original wording is preserved verbatim there and in the decision artifact;
the replacement is the project's standard form rather than a threshold picked
to fit; and **the log-loss bar B1 actually cleared was never touched**. A
reader who thinks the ECE gate should have stood has every number needed to
say so.

And the mitigation list needs one correction of its own. The replacement gate
is stricter on one axis — three seed sets rather than one — but it is
**looser on the axis that decided the outcome**: it introduced a ±2σ
tolerance where the original had none. B1 fails the original gate outright
(0.0124 against 0.0088) and also fails a mean-vs-mean comparison with no
tolerance (0.01337 against 0.01163, +0.0017 the wrong way). It passes only
because the amendment allows a candidate to be worse by up to 0.0056.
"Stricter in what it demands" was the wrong summary and is corrected here and
in the plan's amendment block.

#### The gate judged a form that does not ship

Both versions of the ECE gate compared numbers taken from walk-forward
reports, and every walk-forward report scores the **harness** form — one
temperature refit on each fold's inner-validation year. The deployed scorer
applies a single fixed temperature and cannot fit per fold, so no version of
the gate was ever applied to the model that ships. (One limit on the
correction: the deployed form can only be re-scored where a per-row prediction
dump was committed, and that is B1 at seeds 0–4 alone — so its ECE below is a
single measurement, not the three-seed-set mean the amended gate's arms use.)

Re-scoring the committed per-row dump under the deployed form gives pooled
**log-loss 0.6432 and ECE 0.0108** (5/10/15/20 bins: 0.0034 / 0.0108 / 0.0108
/ 0.0155) against the harness form's 0.6437 and 0.0124 — better on both, and
comfortably inside the amended gate's own threshold of **0.017241** (the
incumbent mean plus its 2σ tolerance), which it clears by 0.0064. The gate
was not re-run on this form: rule 4 was resolved on the harness numbers, and
re-applying a gate to a different form afterwards would be a second post-hoc
move on top of the one already recorded. What is recorded instead is where
the shipped scorer falls against the same threshold
(`deployed_form.against_the_amended_ece_gate` in the decision artifact).

This is not a cosmetic distinction, because the fixed temperature the project
originally shipped was **the wrong one**. It was 0.80, the median of the eight
per-fold fits — a rule borrowed from *training-budget* derivation, where a
median epoch count is a defensible central tendency of a budget. A median
temperature has no calibration justification and no harness run ever measured
it. Re-scored, 0.80 gives pooled ECE **0.0177**: worse than the published
0.0124, and *over* the amended gate's own threshold. The deployed value is now
the walk-forward one (0.85), the only rule that is both fixed at serving time
and validated without using the evaluation rows to choose it, and the
alternatives are all scored side by side in
`models/walkforward/blend_temperature.json`.

One remediation was tried and failed decisively. If a single scalar cannot
fix a *shape* mismatch between two differently-calibrated probability
streams, an isotonic regression fitted on the same inner-validation year
should. It is far worse on both axes: **0.7049** log-loss against B1's
0.6437, ECE 0.0289 against 0.0124, all eight folds worse and five of them
beyond the 0.01 fold-regression tolerance (2019 alone by +0.195). It does not
ship in any form, and it is evidence that temperature scaling was never the
binding constraint.

#### Four dead blocks, alive through the other member

SP2.1 declared `trajectory`, `notice`, `context` and `opponent_adjusted`
dead. That verdict was reached against the MLP alone, and it still stands as
stated: the neural arm reads 0.6475 on the 87-column table against 0.6476 on
the shipped one, which is nothing. The XGBoost arm reads **0.6462 against
0.6490** — the blocks were never dead to the trees, and a blend is how that
reaches the served prediction.

The honest magnitude, though, is small. **B1 (0.6437) against B0 (0.6453) —
the same blend on the previous table — is −0.0016**, which does not clear a
bar of its own. *The blend* is what cleared the bar; the four blocks are a
fraction of it that would have shipped nothing alone. They are in the table
because they came free with a candidate that cleared, not because they
earned their own row.

They are not free to maintain. Three of the six shipped blocks draw on the
static `ehan03/jds-mma-data` snapshot, whose UFC coverage ends 2024-12-14:
`external`, whose `external_missing` is already **0.534 of 2026 rows**;
`notice`, whose columns are unknown on **100%** of 2025–2026 rows by
construction; and `context`'s nationality half. `trajectory` and
`opponent_adjusted` are fight-history recombinations and do not decay. Every
year that passes without a refreshable source moves more of this table to
"unknown".

#### Two deployment bugs the shipping work found

Neither would have raised an alarm on its own, which is why both are recorded
here.

1. **The weekly refresh would have silently rebuilt the shipped table with
   five blocks missing.** `scripts/build_features.py` defaulted to `base`
   only, and `.github/workflows/refresh-data.yml` invokes it bare. Nothing
   would have crashed — the preprocessor and the XGBoost matrix are both
   fitted on whatever columns exist — so the next retrain would simply have
   deployed a worse model with no error anywhere. The default is now the
   blocks recorded in the sidecar next to the committed table; a named
   `--blocks` still wins, so experiments are unaffected.
2. **One unseen weight class would have killed a whole card's predictions.**
   XGBoost 3.x matches categoricals by value, and a served row's
   `weight_class` is categorised from the one value present. A division the
   models never trained on — a "Catchweight" off a Wikipedia card — raised
   `XGBoostError` rather than becoming a missing value, and
   `prospective.predict_fight` did not catch it. `mma.models.xgb.align_to_booster`
   now rebuilds a served frame's categoricals from each booster's own
   category list, so an unseen value becomes missing — which is what the
   neural member already did with it. The XGBoost floor moved to `>=3.0` for
   the same reason: under 2.x categoricals match by *code*, which would have
   made a served weight class silently wrong instead of loud.

Everything measured is committed: `models/walkforward/blend_b*.json`, the two
noise floors (`noise_floor_blend.json`, `noise_floor_ece.json`),
`blend_temperature.json` — the deployed temperature's derivation, every
candidate rule scored side by side, and the round-trip check that the re-score
reproduces `blend_b1.json`'s own pooled numbers — and `sp2_2_decision.json`,
which quotes the rules and the amendment verbatim from the plan file, carries
the deployed form's metrics in `deployed_form`, and is regenerable by
`scripts/sp2_2_decision.py`.

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
and both members of the deployed blend are then **refit on every decisive
fight through the latest event** — there is no longer a historical year held
out from the model the app serves. The [prospective track record](#prospective-track-record) —
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
shipped 87-column feature table. The blend is the deployed scorer; the two
rows above it are its own members, measured on the same folds:

| Model | Accuracy | Log-loss | Brier | ECE |
|---|---|---|---|---|
| Elo baseline | 0.553 | 0.683 | 0.245 | 0.030 |
| XGBoost (5-seed ensemble) | 0.616 | 0.646 | 0.228 | 0.019 |
| Neural net (5-seed ensemble, calibrated) | 0.622 | 0.648 | 0.228 | 0.011 |
| **Blend — the deployed form** (0.5/0.5, one fixed post-average T = 0.85) | **0.622** | **0.643** | **0.226** | **0.0108** |
| Blend — the harness form (a temperature refit on every fold) | 0.622 | 0.644 | 0.226 | 0.0124 |

(`models/walkforward/{xgb_ens5_s1,torch_a1_combined,blend_b1,elo}.json`; the
deployed row from `models/walkforward/blend_temperature.json`. The two blend
rows carry a fourth decimal of ECE because that is where they differ.)

**Why the blend has two rows.** The harness fits one temperature per fold on
that fold's inner-validation year. The deployed scorer has no held-out year
and applies a single fixed temperature to every prediction it will ever make,
so it is a different scorer, and the row a reader of this table cares about is
the deployed one. The fixed value is derived by re-scoring the harness's own
committed per-row predictions under a **walk-forward rule** — fold *Y* gets a
temperature fitted on the pooled out-of-fold predictions of the folds strictly
*before* Y, which is exactly the information deployment has; at serving time
every fold is "before", so the served value is fitted on all of them
(**T = 0.85**, `scripts/derive_blend_temperature.py`). Pooling those folds is
what makes the deployed row an out-of-sample measurement rather than a
temperature scored on the rows that chose it. Everything else — the other
three rows, the candidate table above, the bar B1 cleared — is the harness
form, which is the right basis for comparing *recipes*.

Pooled method macro-F1: XGBoost 0.331, neural net 0.373, blend 0.339;
finish-round macro-F1 (finishes only, 2,446 fights): XGBoost 0.187, neural
net 0.307, blend 0.279. The blend's non-winner heads sit between its
members, which is the cost of averaging: the class-weighted neural heads
identify submissions and early finishes where the trees default toward the
majority class, and averaging pulls that back. Winner log-loss is the
pre-registered headline metric and the blend is better on it than either
member.

For continuity with the earlier record: on the previously shipped
`base,external` table the same protocol gave the neural ensemble 0.6476 and
a single XGBoost fit 0.6506, and on the v1 46-column table 0.6510 and 0.6537
(`models/walkforward/{xgb,torch}_v1.json`) — so `external` is worth −0.0034
to the neural scorer, and the blend a further −0.0039 on top of it.

**Noise floor and pre-registered bar.** Re-running the neural walk-forward
with three disjoint 5-seed sets gives pooled log-losses of 0.6510 / 0.6516 /
0.6510, i.e. **σ_seed ≈ 0.00035** (an n=3 estimate; 95% CI roughly
0.00018–0.0022, `models/walkforward/noise_floor.json`). The same measurement
on the deployed blend gives 0.6453 / 0.6451 / 0.6452, **σ_blend = 0.0001**
(`noise_floor_blend.json`), and on a single XGBoost fit 0.00087
(`noise_floor_xgb.json`). The pre-registered bar for a challenger to
*replace* the incumbent is **0.003 pooled log-loss** — roughly 9σ for the
neural ensemble and 30σ for the blend, so a change has to be far larger than
seed-to-seed wobble, and must not regress any single fold by more than 0.01,
before it ships.

**Fresh-seed re-score.** Every shipped change is confirmed on seeds it was
never chosen on, against a paired incumbent built the same way. The
`external` block: pooled **0.6516 → 0.6473, Δ −0.0043**, worst fold +0.0006,
against a paired incumbent that reproduces `torch_v1_seeds5.json`
bit-exactly. The blend: pooled **0.6471 → 0.6434, Δ −0.0037**, worst fold
+0.0064. Both clear the bar on fresh seeds by about what they cleared it by
on seeds 0–4 — and the step is not a formality, since it is what caught
SP2.1's XGBoost false positive.

**Refit strategy.** The harness then asked whether early-stopping on a
held-out year (protocol A) is actually necessary, or whether a fixed budget
taken from those early-stopping runs, trained on *all* data through the
newest year (protocol B), does as well. On the same folds and the shipped
87-column table: XGBoost 0.6462 → 0.6455, neural net 0.6475 → 0.6473 — both
within the noise floor (`models/walkforward/refit_decision_b1.json`). Since
B is not worse and uses every available fight, both members of the deployed
blend are trained on all **11,238 decisive fights through 2026-08-08** with
that fixed budget (neural net: 6 epochs, temperature 1.15 on every seed;
XGBoost: 109/73/71 trees for the winner/method/round heads, on each of five
seeds). The budget belongs to a *feature table*, not to the recipe: the
previous table's budget was 10 epochs at temperature 1.07 and 105/61/75
trees (`refit_decision_v3.json`) and the v1 table's was 14 epochs at 1.1 and
82/80/76 (`refit_decision.json`); every file records which table it was
taken on. `models/torch/metrics_val.json` and `models/xgb_metrics_val.json`
record the recipe and quote the harness numbers as their evidence. A
fresh-seed re-score of the recipe (seeds 5–9 instead of 0–4) reproduced the
same result — Δ +0.0003, still inside σ_seed — confirming the recipe choice
wasn't a seed-lucky fluke (`fresh_seed_rescore` in
`refit_decision_b1.json`).

**One thing the refit decision does not cover, deliberately.** The blend's
*own* post-average temperature cannot be derived in fixed-budget mode: the
harness fits it on each fold's inner-validation year, which protocol B trains
on, so a refit-mode blend report cannot exist by construction. It is derived
instead by re-scoring the harness's committed per-row predictions — a
**walk-forward temperature**, fitted for fold *Y* on the pooled out-of-fold
predictions of the folds before *Y*, and at serving time on all of them:
**T = 0.85** (`scripts/derive_blend_temperature.py`,
`models/walkforward/blend_temperature.json`). Because each fold is scored by a
temperature that never saw it, the pooled result is a genuine out-of-sample
measurement of the *fixed*-temperature form — 0.6432 log-loss, ECE 0.0108,
against the per-fold-fitted harness form's 0.6437 and 0.0124.

This replaced **T = 0.80**, the median of B1's eight per-fold fitted
temperatures (0.73, 0.76, 0.93, 0.76, 1.00, 0.78, 0.88, 0.82), which shipped
until 2026-09-09. That median is the rule the refit recipe uses for a training
*budget*; nothing justified it for a temperature and no run had measured it.
Measured at last, it scores pooled ECE **0.0177** — worse than the number the
project was publishing, and over the amended ECE gate's own threshold. The
fitted values still trend upward across folds (0.73 in 2018 to 0.82–1.00 in
the recent ones), the same temperature drift SP2 recorded as its follow-up 6;
the walk-forward derivation tracks that drift by construction rather than
averaging it away.


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
deployed refit applies the harness-derived temperature 1.15 to every seed.
Since SP2.2 it is one of two members of the deployed blend rather than the
scorer on its own, and a second temperature (0.85, the walk-forward value) is
applied to the blend *after* the two members are averaged. Uncertainty comes from ensemble spread
(mean 0.087 on the original validation window) and MC dropout; the app's
seed band is now the spread of the five per-seed blends, and the MC-dropout
histogram is the neural member's parameter uncertainty re-centred on the
blend's mean — XGBoost has no dropout. Per the Phase 3 ablation, the
era-proxy `*_missing` flags are excluded from its inputs.

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
3. Predicts every matched fight with the exact committed blend
   (`mma.inference.predict_symmetrized` over `BlendedPredictor`) and writes one JSON record per
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
table. Four redeployments have happened since: the SP2 feature set and
re-derived budget (`b617b96dae45`), the SP2.2 blend (`5aa33460ef40`), and two
pre-merge fixes to that blend's own deployment. First, its mixing weight and
post-average temperature used to be module constants the model hash never
covered, so editing either one would have silently changed every recorded
probability under an unchanged hash; both now live in a committed artifact,
`models/blend.json`, hashed alongside the model weights
(**`6207d19d615b`** — that number itself, and every prediction, was unchanged,
since only where the two constants live moved). Second, the temperature in
that artifact was the *wrong* fixed value: the median of the harness's
per-fold fits, a training-budget rule with no calibration justification, which
re-scores to ECE 0.0177. It is now the walk-forward value 0.85
(**`b863389f1760`**), and unlike the previous redeployment this one **does**
change every probability the model will make: each moves exactly as
σ(logit(p)·0.80/0.85), a small shrink toward 0.5 — slightly less confident and
measurably better calibrated (pooled ECE 0.0177 → 0.0108). Predictions already
recorded under an older hash are left as they were made. The next weekly run opens a
new section for the current hash rather than mixing different scorers'
predictions into one row. 29 further
predictions are awaiting results, and the rest cover events that haven't
happened yet. Grading itself can lag a finished event by days to weeks,
because it depends on the Kaggle mirror picking up the result — the same
lag documented for the weekly data refresh above — but continues
automatically as results land. Nothing here is cherry-picked: every
prediction this pipeline ever makes gets a row, win or lose.

A walk-forward retraining hook (`scripts/roll_window.py`) watches this
track record: once 150 graded prospective fights have accumulated since
the current model's data cutoff, it reports a pre-registered promotion
protocol. The gate operates on the **full 5-seed torch ensemble**, not a
proxy: it retrains the ensemble on a pushed-forward cutoff into a temp dir,
scores both that candidate and the committed incumbent ensemble on the same
newest-2-years held-forward slice (each as its complete artifact, per-seed
temperatures included), and promotes only if the candidate beats the
incumbent by more than 0.002 log-loss. **Since SP2.2 that ensemble is only
half of the served scorer**, so `--execute` now detects a blended incumbent
from the artifacts on disk and aborts before retraining or scoring anything,
rather than reporting a number about half a model as if it described the
model; the dry-run text the weekly Action prints carries the same caveat.
Moving this gate onto the blend, or onto the walk-forward harness, is SP4
work. On promotion
the candidate ensemble is *staged* into `models/torch` (the incumbent is
backed up on disk first) and nothing else happens — the script performs no
git writes. A human then re-runs the refit recipe (bare `scripts/train_xgb.py`,
`scripts/train_torch.py`, `scripts/train_hazard.py`,
`scripts/check_display_calibration.py` — a split-protocol
candidate never ships as-is), runs the suite, reviews the metrics diff, and
commits by hand; the new model's artifact hash (`mma.versioning.model_version`,
computed over every artifact that scores — the torch weights and preprocessing
stats, the twenty-five per-seed XGBoost boosters, `models/blend.json` and
`models/simulator.json`) becomes the new
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
CC0, via kagglehub) and scores the committed scorer against the devigged
market-implied probability on the same fights. (The script loads whatever
serves today, so a re-run would score the blend; the committed numbers below
are the frozen July 2026 artifact, computed against the neural ensemble of
that date, and are not recomputed.) **The odds are
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
  `mma.versioning.model_version()`, a hash of **everything that scores** —
  the torch weights and preprocessing statistics, the twenty-five per-seed
  XGBoost boosters (winner/method/round, plus SP3's hazard and decision), and
  the two config artifacts whose hand-set numbers are as load-bearing as any
  trained one, `models/blend.json` (weight, temperature) and
  `models/simulator.json` (`n_runs`, `alpha`, `sim_seed`, `default_rounds`) —
  so the track record splits by model, not by commit.
  Before SP2.2 the hash covered only the torch half, which was correct while
  XGBoost was an explainer and would have covered half of what serves now;
  SP3 needed the same expansion again for the simulator. It is verified
  sensitive to a mutation on any member. `models/display_calibration.json`
  stays excluded: since SP3 nothing recalibrates a displayed number, so that
  file is a measurement of the model rather than an input to a prediction, and
  hashing it would open a new track-record section for byte-identical
  predictions whenever the weekly measurement moved a decimal.
  Retraining (weekly refresh or walk-forward promotion) starts a new
  section automatically; retraining is deterministic, so an unchanged
  dataset yields an unchanged version. The evidence behind each deployed
  model lives in `models/walkforward/` (the harness reports, noise floors,
  and refit decision), and the metrics files quote it. After a weekly data
  refresh the refit reuses the committed budget (6 epochs / T 1.15;
  109/73/71 trees on each of five seeds) on the newer data, but the harness
  reports are *not* re-run by the Action — re-run `scripts/run_walkforward.py`,
  `scripts/noise_floor.py`, and `scripts/refit_decision.py --reports b1` by
  hand to refresh the evidence (the train scripts warn when the harness's
  data is older than the training cutoff). Automating that is SP4.

## Interactive app

`app.py` is a Streamlit front end over the committed **blend**: pick two
fighters, a weight class, round count, and title-fight flag, and it renders
the win probability, the spread across the five per-seed blends, an
MC-dropout uncertainty histogram, method-of-victory / finish-round
breakdown, both fighters' Elo trajectories, and a "Why this prediction?"
panel breaking down the top factors driving the call — all from the
artifacts already checked into `models/torch/` and
`models/xgb_<head>_seed*.json`, no training required. Predictions are
symmetrized across both fighter orderings
(`mma.inference.predict_symmetrized`), and **the thing being symmetrized is
the blend**: each corner ordering is blended and calibrated first, then the
two blended probabilities are averaged, so the reported probability is
always self-consistent. The "Why this prediction?" panel averages TreeSHAP
over all five boosters and both orientations — the XGBoost winner head is
now half the deployed scorer rather than a companion model that merely
agreed with it — and says plainly what it still is not: an attribution of
the whole blend, since the neural half is not decomposed and the
post-average temperature rescales the blended logit.

Run it locally:

```bash
.venv/bin/pip install -e ".[dev,app]"
.venv/bin/streamlit run app.py
```

Deploy to Streamlit Community Cloud: push this repo to a public GitHub
remote, go to [share.streamlit.io](https://share.streamlit.io), click
**New app**, and point it at this repo/branch with `app.py` as the entry
point.
