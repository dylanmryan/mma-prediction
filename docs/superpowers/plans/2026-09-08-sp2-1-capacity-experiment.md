# SP2.1: Combined Blocks × Model Capacity — Pre-Registered Experiment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Status:** Pre-registered 2026-09-08, *before* any configuration below was run.

**Question.** SP2 measured eight feature blocks one at a time and shipped one. Two things in those results suggest the measurements may have understated the features:

1. **Several rejected blocks have negative (helpful) point estimates** that individually fall short of the 0.003 bar. Combined, they might clear it.
2. **The two scorers disagree in sign on exactly the blocks with the most new columns.** `opponent_adjusted` improved XGBoost by 0.0018 while worsening torch by 0.0022; `trajectory` improved XGBoost by 0.0020 and torch by only 0.0004. A gradient-boosted ensemble picks the better-conditioned member of a correlated pair per split; a fixed-width MLP must spend capacity on all of them. That is a **capacity** hypothesis, not an information hypothesis — and the deployed scorer is the MLP, whose architecture was tuned when the table had 46 columns.

**Hypothesis (H1).** The rejected blocks contain signal the *deployed architecture* cannot exploit. Giving torch a wider/better-regularised trunk over a combined feature set clears the bar where each block alone did not.

**Null / rival hypothesis (H0).** The gains are architecture-only: a better torch config helps just as much on the already-shipped feature set, and the extra columns add nothing. **This is why the experiment has a control arm** — without it, any improvement would be misattributed to the features.

---

## Pre-registered design

### Feature sets

- **S0 (shipped):** `base,external` — the current deployed table, 11,238 × 54, with `external_missing` and `same_country` in the table but excluded from the model.
- **S1 (combined):** S0 **plus** every block whose torch point estimate was negative, **plus** `opponent_adjusted` on the strength of its XGBoost result and the capacity hypothesis:
  `trajectory`, `notice` (flag held out of the model, its shipping form), `context` (home-country flags held out of the model, its `context_nohome` form), `opponent_adjusted`.
  **Excluded on purpose:** `in_fight` (+0.0018 torch, +0.0003 XGB — positive on both, no capacity story) and `rankings` (+0.0011 torch, +0.0001 XGB, and 13.8% coverage).
- **Recency** is a training setting, not columns: the 8-year half-life (`--half-life 8`), which was the best cell of the SP2 grid, is applied as a **variant** of each arm rather than a separate arm.

### Arms (fixed now; no additions later)

| Arm | Features | Torch config | Purpose |
|---|---|---|---|
| A0 | S0 | deployed (default) | incumbent, already measured: pooled 0.6476 |
| A1 | S1 | deployed (default) | pure combination effect |
| **C** | **S0** | **search** | **control: is the architecture alone suboptimal?** |
| **T** | **S1** | **search** | **treatment: capacity × features** |
| A2 | winner of {C, T} | + `--half-life 8` | does recency stack on the winner? |

### Search space (fixed now)

25 configurations, seeded `rng(0)`, sampled over:
`hidden` ∈ {(128,64), (256,128), (256,128,64), (384,192), (512,256)}; `dropout` ∈ [0.2, 0.5]; `lr` ∈ [3e-4, 3e-3] log-uniform; `weight_decay` ∈ [1e-5, 3e-3] log-uniform; `embedding_dim` ∈ {4, 8, 16}; `method_scale` ∈ [0.25, 0.75]; `round_scale` ∈ [0.1, 0.4].
Each configuration is scored with **2 seeds (0,1)** for ranking. The identical 25 configurations are used for both C and T, so the arms differ only in their feature set.

### Decision rule (mechanical, fixed now)

Let `I` = the shipped incumbent `models/walkforward/torch_external_diffsonly_extslice.json` (pooled winner LL 0.6476), σ_seed = 0.000346, bar = 0.003.

1. Rank the 25 configs within each arm by 2-seed mean pooled winner log-loss. Take the best of each arm and run it as a full **5-seed** ensemble.
2. **Ships only if** the 5-seed result clears the bar vs `I` (`bar_check ... ships: true`) **and** a **fresh-seed re-score with seeds 5–9 also clears it**, measured against the fresh-seed paired incumbent. Both are required; the search is best-of-25 and therefore carries selection optimism, which the fresh-seed arm is there to strip out.
3. **Attribution, decided before seeing the numbers:**
   - If **T clears and C does not** → H1 supported; ship S1 with the searched config.
   - If **both clear and T − C > σ_seed** → both matter; ship S1 with the searched config, and report the split.
   - If **both clear and T − C ≤ σ_seed** → H0: the gain is architecture, not features. **Ship the searched config on S0** (the simpler table), and record that the extra blocks contributed nothing even at higher capacity.
   - If **neither clears** → the blocks are dead and the architecture is adequate. Record and revert.
4. A0/A1 are reported for completeness but cannot ship on their own (A1 is a no-search combination; if it happened to clear the bar it would still need the fresh-seed confirmation in rule 2).

### What would make this experiment wrong, stated in advance

- Testing more than the five arms above, or widening the search space after seeing results, converts this from one test into a search and invalidates the bar. If a promising direction appears, it is recorded as a *future* pre-registration, not added here.
- The fresh-seed confirmation is not optional and its threshold is not negotiable after the fact.
- Restoring the reverted blocks must reproduce their SP2 numbers first (see Task 1) — otherwise a "gain" could be a restoration bug.

---

### Task 1: Branch and restore the reverted blocks, verifying against their SP2 numbers

**Files:** `src/mma/feature_blocks.py`, `src/mma/history.py`, `src/mma/snapshots.py`, `src/mma/features.py`, `src/mma/inference.py`, their tests.

- [x] **Step 1: Branch**

```bash
git checkout main && git status --short && git checkout -b sp2-1-capacity
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
```
Expected: clean; `532 passed, 1 skipped`.

- [x] **Step 2: Restore `trajectory`, `notice`, `context`**

Each has a commented-out registration in `feature_blocks.py` and its supporting module kept (`mma.glicko` + `run_glicko` in `build_ratings.py`; `mma.notice` + `data/external/fight_notice.parquet`; `mma.context`). Uncomment the registrations and restore whatever wiring in `features.py` / `inference.py` the block comments say was removed. For `notice` and `context`, register them in their **shipping form** — the flags that failed the constant-vector leak check (`notice_unknown`; `home_country_a`/`home_country_b`) stay in the table but go into `tensors.DROPPED` + `models/xgb.MODEL_EXCLUDED`, exactly as `external`'s artifact flags do.

- [x] **Step 3: Restore `opponent_adjusted` from git history**

Its accumulators were fully reverted; recover them from commit `2ad725c` (`git show 2ad725c^:src/mma/history.py` etc. — note that commit *is* the revert, so the implementation is in its parent's tree or in the diff). Restore the block and its unit tests.

- [x] **Step 4: Verify each restoration reproduces its SP2 measurement**

For each restored block, rebuild `base,external,<block>` and re-run the XGB walk-forward, comparing against that block's committed SP2 report:

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external,trajectory
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_restore_trajectory
```
The pooled winner log-loss must match the committed `models/walkforward/xgb_trajectory.json` to 4 decimals. Repeat for `notice`, `context`, `opponent_adjusted` against their committed reports. **Any mismatch is a restoration bug — fix it before proceeding.** (Reports named `*_restore_*` are scratch; delete them before committing.)

- [x] **Step 5: Build S1 and run the leakage and parity gates**

```bash
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external,trajectory,notice,context,opponent_adjusted | tail -3
~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_serving_parity.py tests/test_feature_blocks.py tests/test_external.py tests/test_context.py -q
```
Blocking: truncation-invariance and serving parity must pass on the combined table. Run the data-level constant-vector leak check over every restored block's modelled columns, on the rows where exactly one corner's external data is unavailable — the combined table has more channels for the coverage artifact than any single block did, so this is the highest-risk step in the experiment. Report the column count of S1 and per-column coverage.

- [x] **Step 6: Commit**

```bash
git add -A && git status --short
git commit -F - <<'EOF'
Restore the rejected blocks for the SP2.1 capacity experiment

Each restoration is verified to reproduce its committed SP2 walk-forward
number before use, so a later gain cannot be a restoration artifact.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 2: Arms A0 and A1 — the pure combination effect

- [x] **Step 1: A1 (S1, deployed config)**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_a1_combined
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_a1_combined
~/.venvs/mma/bin/python scripts/block_decision.py --candidate models/walkforward/torch_a1_combined.json --incumbent models/walkforward/torch_external_diffsonly_extslice.json
```
A0 is the committed incumbent; no run needed. Record A1's pooled, per-fold and slice numbers, and the XGB number (the capacity hypothesis predicts XGB improves more than torch here).

- [x] **Step 2: Commit the reports and the numbers.**

---

### Task 3: The search — arms C and T

**Files:** `scripts/config_search.py` (new)

- [x] **Step 1: Write `scripts/config_search.py`**

Samples the 25 configurations from the fixed space above with `numpy.random.default_rng(0)`, and for a given `--blocks` set and `--drop-columns` runs each through the walk-forward harness with `--seeds 0,1`, writing one report per config under `models/walkforward/search/<arm>/` plus a ranked summary JSON. It must reuse `scripts/run_walkforward.py`'s machinery rather than reimplementing scoring (import it, or shell out — either is fine, say which). Print progress; the whole arm is ~25 × 2 seeds and will take a while, so make it resumable (skip a config whose report already exists).

- [x] **Step 2: Run arm C (control: S0, searched)**

```bash
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external
~/.venvs/mma/bin/python scripts/config_search.py --arm C --blocks base,external
```

- [x] **Step 3: Run arm T (treatment: S1, searched)**

```bash
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external,trajectory,notice,context,opponent_adjusted
~/.venvs/mma/bin/python scripts/config_search.py --arm T --blocks base,external,trajectory,notice,context,opponent_adjusted
```

- [x] **Step 4: Best-of-arm at 5 seeds**

Take each arm's top config by 2-seed mean and run it as a full 5-seed ensemble (`--seeds 0,1,2,3,4`), named `torch_C_best` / `torch_T_best`. Apply `block_decision.py` against the shipped incumbent. Record the distribution of the 25 configs per arm (min / median / max), not just the winner — the spread says how much of the "best" is selection.

- [x] **Step 5: Commit the search outputs and the summary.**

---

### Task 4: Fresh-seed confirmation and the attribution decision

- [x] **Step 1: Fresh seeds for whichever arms cleared at 5 seeds**

Run the winning config(s) with `--seeds 5,6,7,8,9`, and build the matching fresh-seed paired incumbent the way SP2 did (v-recipe with the same `--drop-columns` on the same table, verified to reproduce `torch_v1_seeds5`-style bit-exactly where applicable).

- [x] **Step 2: Arm A2 — does recency stack?** *(N/A — gated on the winner of {C, T} clearing the bar; neither did, so A2 was not run.)*

Only on the winner of {C, T}, and only if it cleared: re-run with `--half-life 8`. This is the fifth and last arm.

- [x] **Step 3: Apply the attribution rule from the pre-registration verbatim**

Write `models/walkforward/sp2_1_decision.json` with every arm's pooled numbers, the search distributions, the fresh-seed results, the rule as a string, and the resulting decision. Do not deviate from the rule; if the numbers are ambiguous in a way the rule does not cover, say so explicitly and stop for a human call rather than inventing a tie-break.

- [x] **Step 4: Commit.**

---

### Task 5: Ship or revert, document, merge

- [x] **Step 1:** *(N/A — nothing shipped, so nothing was retrained or redeployed.)* If something ships, update the train scripts' defaults (re-deriving the refit budget on the new table/config via `scripts/refit_decision.py --reports <set>` as SP2 Task 13 did), retrain, record the new model hash, verify determinism, exercise an end-to-end matchup, and confirm the app boots.
- [x] **Step 2:** If nothing ships, revert the feature wiring (keeping the reusable modules and every report), exactly as SP2 did for its seven rejections.
- [x] **Step 3:** README + plan Completion notes + spec: the question, the design, the arms, the numbers, the attribution, and what it implies for SP3. A negative result here is a genuinely useful finding — it would say the box-score data *and* the architecture are both near their ceiling, which is the strongest possible argument for the simulator paradigm.
- [x] **Step 4:** Full suite, `make_dataset.py` / `build_features.py` byte-identity, merge to main with `--no-ff`. Do not push.

---

## Completion notes (filled in during execution)

**Outcome in one line:** the pre-registration's fourth attribution branch --
*neither arm clears; the blocks are dead and the architecture is adequate;
record and revert*. Nothing shipped, nothing was retrained, and the deployed
model hash is still SP2's `b617b96dae45`.

**Restoration verification (Task 1).** Stronger than the pre-registration
asked for: each block reproduced its committed SP2 report's pooled metrics,
all eight fold metrics, all slice metrics and the per-fold fit budgets
*exactly*, not merely to 4 decimals.

| Block | Table rebuilt | Reproduced XGB pooled | Committed report |
|---|---|---|---|
| `trajectory` | `base,external,trajectory` | 0.6486 | `xgb_trajectory.json` |
| `notice` | `base,external,notice` | 0.6501 | `xgb_notice_noflag.json` |
| `context` | `base,external,context` | 0.6499 | `xgb_context_nohome.json` |
| `opponent_adjusted` | `base,opponent_adjusted` | 0.6519 | `xgb_opponent_adjusted.json` |

`opponent_adjusted` is measured on `base` alone because that is the feature
set its SP2 report was computed on, before `external` shipped. Its code was
never committed in a working state (`2ad725c` committed the *reverted* tree,
so the accumulators are in no git tree at all) and had to be rebuilt from the
SP2 plan's implementation contract, then pinned by the committed number. Two
details are only recoverable that way and both are load-bearing: a fight with
no recorded statistics contributes no versus-expectation pair (the career
counters still take it as zero, which is why the block's coverage is 7,046
rows and not 7,092), and `td_def`'s mirror is the opponent's takedown
*accuracy* rather than 1 − accuracy. Every other combination tried misses the
committed number.

**S1 (Task 1, Step 5): 11,238 × 87** — 33 columns on top of S0's 54; 76
modelled by XGB and 71 by torch after the five held-out flags
(`external_missing`, `same_country`, `notice_unknown`, `home_country_a`,
`home_country_b`). Coverage of the new columns, as a share of the 11,238 rows
(booleans are 100% by construction — False means "not observed"):

| Group | Columns | Coverage |
|---|---|---|
| `notice` | `notice_shortfall_days_diff`, `missed_weight_over_lbs_diff` | 49.8% (and **0% of 2025–2026**) |
| `trajectory` | Glicko triple, `elo_delta_3/5` | 100% |
| `trajectory` | `elo_peak_minus_current_diff`, `years_since_ufc_debut_diff` | 72.6% |
| `trajectory` | `age_x_fights_diff` 96.9%, `age_squared_a/b` 98.1% | — |
| `context` | `bonus_rate_diff` 72.6%, referee rates 95.1% | — |
| `opponent_adjusted` | four `*_vs_exp_diff` 62.7%, `td_def_vs_exp_diff` 51.4% | — |
| `opponent_adjusted` | `avg_opp_elo_wins_diff` 61.7%, `avg_opp_elo_losses_diff` 49.6% | — |

**Leak check on the combined table:** passed, and it is the check this
experiment was most likely to fail — S1 has four coverage channels where each
single block had one. `tests/test_external.py::test_no_modelled_column_identifies_the_unmapped_corner`
rebuilds the table with the mapped corner of every half-matched fight erased
and requires that no column which moves is in either model matrix. Only
`home_country_a`/`_b` move, and both were already held out. Truncation
invariance and both serving-parity tests also passed on S1.

**A1 — the combination under the deployed configs (Task 2).** S1, no search.

| Scorer | Pooled | Incumbent | Δ | `ships` |
|---|---|---|---|---|
| torch | 0.6475 | 0.6476 (`torch_external_diffsonly_extslice`) | **−0.0001** | false (2018 +0.0048, 2021 +0.0025) |
| XGB | 0.6467 | 0.6506 (`xgb_external_diffsonly_extslice`) | **−0.0039** | true, all four slices better |

That split is the capacity prediction, quantified: the identical extra columns
move gradient boosting 0.0039 and the fixed-width MLP 0.0001. It is also the
result that did not survive re-seeding — see the fresh-seed section.

**Arms C and T — the 25-configuration search (Task 3).** The identical 25
configs, sampled once with `rng(0)` into `models/walkforward/search/configs.json`
and re-checked on every later run, so the arms differ only in their feature
set. 2-seed ranking distributions of pooled winner log-loss:

| Arm | Features | min | median | max |
|---|---|---|---|---|
| C (control) | S0 `base,external` | 0.6463 | 0.6478 | 0.6495 |
| T (treatment) | S1 combined | 0.6463 | 0.6483 | 0.6503 |

Both spreads are ~0.003 wide against a torch seed-noise floor of 0.00035, so
best-of-25 buys a visible share of its "best" from selection — which is
precisely what rule 2's fresh-seed confirmation exists to strip out. Note the
two minima are identical to 4 dp while T's median is 0.0005 *worse*: the extra
columns do not help the median configuration at all.

Best-of-arm at 5 seeds, against the shipped incumbent (0.6476):

| Arm | Config | Architecture | Pooled | Δ | `ships` |
|---|---|---|---|---|---|
| C | `config_15` | hidden (512,256), dropout 0.490, lr 3.10e-4, wd 1.38e-3, emb 16, method 0.741, round 0.387 | 0.6463 | **−0.0013** | false |
| T | `config_19` | hidden (128,64), dropout 0.354, lr 2.55e-3, wd 1.46e-5, emb 16, method 0.671, round 0.120 | 0.6464 | **−0.0012** | false |

All 50 runs completed. The winning architectures are at opposite ends of the
space (the widest trunk for S0, the narrowest for S1), which is itself a sign
that the search surface is flat relative to the noise.

**Fresh-seed confirmation (Task 4).** Applied to A1-XGB, the one result that
cleared any bar. C and T failed at 5 seeds and rule 2's confirmation is a
requirement for *shipping*, not for recording a failure, so their torch
fresh-seed re-scores were deliberately not run. XGBoost has no seed ensemble
to re-seed — its stochasticity is `subsample`/`colsample` under one
`random_state` — so `run_walkforward.py` grew a `--model-seed` pass-through,
and `--model-seed 0` was verified to reproduce both committed seed-0 reports
exactly before use.

| `--model-seed` | A1 (S1) | Incumbent (S0) | Δ |
|---|---|---|---|
| 0 (committed) | 0.6467 | 0.6506 | −0.0039 |
| 1 | 0.6472 | 0.6496 | −0.0024 |
| 2 | 0.6488 | 0.6512 | −0.0024 |
| 3 | 0.6493 | 0.6516 | −0.0023 |
| mean, fresh seeds 1–3 | 0.64843 | 0.65080 | **−0.00237** |
| mean, all four seeds | 0.6480 | 0.65075 | **−0.00275** |

**FAILS.** The seed-0 clearance was seed luck; re-seeding halves the effect
and it lands inside the 0.003 bar either way.

**Noise floors.** torch `sigma_seed` = **0.000346** (5-seed ensemble,
`models/walkforward/noise_floor.json`, measured in SP1). XGB `sigma_seed`,
measured here for the first time in this project
(`models/walkforward/noise_floor_xgb.json`): **0.00087** on the incumbent
recipe and **0.00125** on the A1 recipe — 2.5–3.6× the torch ensemble's. That
is the price of scoring a single fit instead of a 5-seed average, and it is
the whole explanation for why A1-XGB looked like a ship candidate at one seed.

**A2 (recency on the winner):** not run. The pre-registration gates it on the
winner of {C, T} having cleared the bar; neither did.

**Attribution and decision.** Recorded mechanically in
`models/walkforward/sp2_1_decision.json` by `scripts/sp2_1_decision.py`.
C = −0.0013 (`ships: false`), T = −0.0012 (`ships: false`), T − C = +0.0001,
inside σ_seed = 0.000346. **Branch four: "neither clears → the blocks are dead
and the architecture is adequate. Record and revert."** H1 (the capacity
hypothesis) is not supported, and H0's architecture-only story is not
supported either — a 25-config search over five architecture families and
five hyper-parameters could not move the control arm a *third* of the bar. The
deployed MLP, whose shape was tuned when the table had 46 columns, is not
leaving anything on the table at 54 or at 87.

Two facts fell outside the rule and were recorded as `open_questions` rather
than decided, per the pre-registration's instruction to stop for a human call:
A1-XGB's seed-0 clearance (resolved by the fresh-seed re-score above) and
whether the scorer ladder has flipped (open; see Follow-ups 1).

**Revert (Task 5).** `data/processed/features.parquet` rebuilt at
`base,external` and verified **byte-identical to `main`'s** (both blobs hash
`a3b42d78d150…`), 11,238 × 54. The four `register(...)` calls in
`feature_blocks.py` are commented out again, SP2-style, each carrying its
SP2.1 result next to its SP2 one. Everything reusable stayed: every
walk-forward report, `mma.glicko`, `mma.notice`, `mma.context`, the
accumulators in `mma.history` / `mma.snapshots`, `scripts/config_search.py`,
`scripts/sp2_1_decision.py`, `scripts/blend_check.py`, `--model-seed`, the XGB
noise floor, and every permanent test added here — including the whole-table
leak check, which passes on the reverted table.

**Where `opponent_adjusted` lives now,** so it is not lost a second time:
unregistered but live in `src/mma/history.py` (`VS_EXPECTATION_RATES`,
`_FighterState.allowed`, the `*_vs_exp` and `avg_opp_elo_*` accumulators) and
`src/mma/snapshots.py`, computed unconditionally like every base rate. Its
unit tests in `tests/test_history.py` call `build_history` directly and stand
alone without the registration; `tests/test_snapshots.py` keeps the serving
path covered by registering the archived block specs into a throwaway
registry. The commented-out registration in `feature_blocks.py` says all of
this in place.

**Suite:** 588 passed, 1 skipped, before and after the revert. **Deployed hash
after: `b617b96dae45` — unchanged from SP2.**

**Follow-ups:**

1. **The blend is the strongest untested lead in this project, and it is not a
   ship candidate under this pre-registration.** An equal-weight average of the
   XGB and torch winner probabilities scores **0.6436 on S1** and **0.6459 on
   S0** against the deployed scorer's 0.6476 — −0.0040 and −0.0017 — with every
   slice improving. Measured by `scripts/blend_check.py`
   (`models/walkforward/blend_a{0,1}_xgb_torch.json`) precisely so the human
   call is made with the third option on the table. A blend is not one of the
   five pre-registered arms; SP2.1's rules cannot ship it, and reading it as a
   result of this experiment would be the exact "widening the search after
   seeing results" failure the pre-registration names. It needs its own
   pre-registration, whose design has to answer: the blend inherits half of
   XGB's 0.00125 seed sd, so its confirmation needs multiple XGB seeds (or an
   XGB seed ensemble) rather than one; and `open_questions[1]` in the decision
   file already lists the nine places a scorer change would touch, from
   `inference.Ensemble` to `versioning.MODEL_ARTIFACT_GLOBS` (which would stop
   hashing the artifacts actually serving predictions).
2. **XGBoost's seed noise is large enough to have been mismeasuring SP2's
   screens.** σ ≈ 0.0009–0.0013 for a single XGB fit, against the 0.0018–0.0020
   deltas several SP2 block screens reported. To be precise about what this
   does and does not impeach: **no SP2 rejection rests on the XGB screen
   alone** — the screen only ever *dropped* a block whose XGB number was worse
   than the incumbent by more than 0.002, and in fact it never dropped one, so
   every block was decided by a torch 5-seed run whose σ is 0.000346. The
   damage is confined to the *narrative*: statements of the form "XGB liked it
   by 0.0018" are one-sigma-and-a-half claims, not measurements. Future XGB
   screening should use **≥3 model seeds** and report the mean; `--model-seed`
   and `scripts/noise_floor.py --candidate xgb` now exist for exactly that.
3. **`mma.explain`'s strict feature-name check was relaxed during this work
   and should be re-examined.** It previously required the served row's
   columns to match the booster's exactly; it now takes the booster's own
   `feature_names`, raising on a missing column and ignoring extra ones. That
   was needed because the feature table was wider than the deployed model
   while the experiment ran — a state that no longer exists after this revert.
   The relaxation is defensible on its own terms (an extra column genuinely is
   not part of a model that was not trained on it), but it removes a tripwire
   that would catch a table/model mismatch in serving, so verify that the
   trade is still the one wanted before SP3 widens the table again.
4. **A negative result about the ceiling, worth stating as a finding.** Eight
   blocks measured one at a time in SP2, four of them re-measured together at
   higher capacity here, and a 25-configuration architecture search on the
   shipped table: none of it moves the pooled number more than 0.0013. The
   box-score feature space and the fixed-width MLP are both at their ceiling.
   See the spec's SP2.1 section for what that implies for SP3.
