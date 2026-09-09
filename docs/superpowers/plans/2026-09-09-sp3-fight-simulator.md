# SP3: Round-by-Round Fight Simulator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace three loosely-coupled classification heads with one coherent generative process — a per-round hazard model plus a decision model, played out by Monte Carlo — so that winner, method, round, "goes the distance" and any derived prop all come from the same simulated fights and cannot contradict each other.

**Why now.** SP2 measured eight feature blocks and shipped one; SP2.1 then showed that neither a combined feature set nor 25 re-tuned architectures moved the deployed model. The box-score feature space and the MLP are both at their ceiling. What has *not* been tried is changing what the model predicts. The method head currently manages 0.39 macro-F1 and the finish-round head 0.31 against a majority-class baseline — those are the weakest parts of the system, and they are exactly what a simulator targets.

**Architecture:** `src/mma/hazard.py` builds the per-round training rows and the decision rows from the existing processed tables. `src/mma/simulator.py` is a pure Monte Carlo over a fitted member: for each round, draw from the hazard distribution; if every round completes, draw from the decision model. `HazardCandidate` wraps fitting + simulation behind the walk-forward harness's standard `fit_predict` protocol so the simulator is judged by exactly the same bar as everything else.

**Tech Stack:** Python 3.11, pandas, numpy, xgboost, torch (CPU), pytest. Local-disk venv `~/.venvs/mma`; every command is `OMP_NUM_THREADS=1 ~/.venvs/mma/bin/{python,pytest}`. Suite ~16 s; an XGB walk-forward ~15 s; a blend walk-forward ~2 min.

**Commit convention:** end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. `git status --short` before each commit.

---

## Data (verified 2026-09-09)

From `data/processed/fights.parquet`, 11,238 decisive fights:

| | count |
|---|---|
| Hazard rows (one per fight per round actually fought) | **25,336** |
| Finishes | 6,276 — R1 3,588 · R2 1,756 · R3 840 · R4 55 · R5 35 · R6 2 |
| Decisions (went the distance) | 4,925 |
| Scheduled rounds | 3 → 9,719 · 5 → 887 · 2 → 404 · 1 → 183 · NA → 45 |

Note the tail: 2 fights recorded a sixth round, 587 fights were scheduled for 1 or 2 rounds (early era), and 45 have no scheduled-round value. The simulator must handle all of these rather than assuming 3-or-5.

## Locked evaluation rules

Inherited from the v3 spec §4 SP1/§5 and unchanged by this plan:

- **Incumbent:** the deployed blend. Winner marginal `models/walkforward/blend_b1.json` pooled **0.6437** (harness form; the deployed fixed-temperature form is 0.6432 — use the harness form for like-for-like comparison since candidates are scored in harness form). Composed joint-outcome log-loss for the same report is the baseline the simulator must beat; read it from the report rather than assuming.
- **Simulator ships** iff joint-outcome log-loss beats the composed baseline by **more than 0.01** *and* its winner marginal is not worse than the incumbent's by more than σ_seed (0.000346).
- **Fresh-seed confirmation is mandatory** before anything ships (seeds 5–9), against a fresh-seed paired incumbent. This has already caught one false positive in SP2.1.
- Negative results are deliverables: every experiment gets a row in the Completion notes whether or not it ships, and rejected code is reverted while its reports and reusable modules stay.

## Scoring the simulator fairly — decided before any run

The existing `evaluate.joint_outcome_log_loss` composes `P(w)·P(m)·P(r|finish)`, i.e. it assumes independence between winner, method and round. That is exactly the weakness the simulator is meant to fix, so scoring the simulator through the same composition would throw away its advantage. Instead:

- The simulator emits a **joint distribution over outcome cells** — (winner ∈ {A,B}) × (method ∈ {ko_tko, submission}) × (round ∈ 1..R), plus (winner) × decision. The realised cell's probability is read directly and scored as `−log p`.
- The **baseline** is scored on the same cell set via the composition it already uses, so both are `−log P(realised cell)` and directly comparable.
- **Monte Carlo zero-probability is a real hazard**: a cell with no simulated occurrences would score `−log 0`. Apply **Laplace smoothing with α = 1 over the valid cells for that fight** (valid = cells reachable given `scheduled_rounds`), stated here in advance, applied identically to every candidate, and covered by a test. Report the fraction of realised cells that had zero raw simulated mass — if it is more than ~1%, N is too small.
- `N = 10,000` runs per fight per member, fixed now. Verify the Monte Carlo standard error on P(A wins) is below 0.005 (it should be ≈ 0.005 at p=0.5, N=10,000 for a single member; averaging across members reduces it further). Record the measured error.

---

### Task 1: Branch, hazard/decision row construction

**Files:** create `src/mma/hazard.py`, `tests/test_hazard.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git status --short && git checkout -b sp3-simulator
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
```

- [ ] **Step 2: Write the failing tests**

`build_hazard_rows(features, fights) -> pd.DataFrame`: one row per (fight, round actually fought), carrying the fight's feature columns plus `round_no`, `scheduled_rounds`, and a 5-class label `hazard_label ∈ {a_ko, a_sub, b_ko, b_sub, survive}`. Semantics to pin with a hand-built fixture:
- A fight won by A via KO in round 2 of 3 produces exactly two rows: round 1 → `survive`, round 2 → `a_ko`.
- A decision over 3 rounds produces three `survive` rows (the fight survived every round; the decision model handles who won).
- Corner orientation follows the feature table's `swapped` flag, so `a_*` always means the feature table's corner A — **assert this explicitly**, because getting it backwards would silently invert every method prediction.
- Rounds never reached are absent (this is what makes the censoring correct); a fight finished in R1 contributes one row, not three.
- A fight with NA `scheduled_rounds` or an out-of-range `finish_round` (the two R6 fights) is handled without raising — decide and document whether it is dropped or clamped.
- `build_decision_rows(features, fights)`: fights whose method is `decision`, labelled by which corner won.
Also test that hazard rows for the whole table total 25,336 (the verified count) and that decision rows total 4,925.

- [ ] **Step 3: Implement, run, commit.**

---

### Task 2: The simulator (pure, no model)

**Files:** create `src/mma/simulator.py`, `tests/test_simulator.py`

- [ ] **Step 1: Write the failing tests**

`simulate(hazard_probs, decision_prob, scheduled_rounds, n_runs, rng) -> OutcomeDistribution`, where `hazard_probs` is an `(R, 5)` array (per round, the 5-class distribution) and `decision_prob` is P(A wins on the cards). Pure — no model, no I/O. Properties to pin:
- **Degenerate cases:** hazard = certainty of `a_ko` in round 1 → P(A wins) = 1, P(KO) = 1, P(round 1) = 1. Hazard = certain `survive` every round → the outcome distribution equals the decision model exactly, with all mass on "decision".
- **Marginals are coherent:** P(A wins) + P(B wins) = 1; method probabilities sum to 1; round probabilities conditional on a finish sum to 1.
- **Round masking:** for `scheduled_rounds = 3` no mass lands in rounds 4–5.
- **Monte Carlo error:** with a fixed rng and hazard implying a known analytic P(A wins), the simulated value is within 4 standard errors of it (compute the analytic value in the test, do not hardcode a number from a run).
- **Determinism:** same rng seed → identical output.
- **Laplace smoothing:** an unreachable-in-simulation but valid cell gets non-zero probability; an invalid cell (round 4 of a 3-round fight) gets exactly zero and is excluded from the smoothing denominator.
- `OutcomeDistribution` exposes the joint cell probabilities and the derived marginals (winner, method, round, `p_distance`), with `p_distance` equal to the decision cells' total mass.

- [ ] **Step 2: Implement, run, commit.** Keep `simulate` free of pandas — arrays in, arrays out — so it is fast enough to call for every fight in a walk-forward fold.

---

### Task 3: `HazardCandidate` in the harness

**Files:** modify `src/mma/candidates.py`, `scripts/run_walkforward.py`; modify `src/mma/walkforward.py` (joint scoring from a cell distribution); tests

- [ ] **Step 1:** `HazardCandidate` implementing the standard `fit_predict(features, fold, sample_weight) -> (pred, info)`:
  - fits an XGBoost 5-class model on the fold's hazard rows and a binary model on its decision rows, as a seed ensemble;
  - for each evaluation fight, predicts the per-round hazard distribution and the decision probability, runs `simulate`, and returns the joint cell distribution **plus** the marginals in the existing `pred` shape (`winner`, `method`, `round`) so every existing metric keeps working unchanged;
  - symmetrises by running both corner orderings and averaging, as `predict_symmetrized` does for the current models;
  - `info` records both members' fit diagnostics and the measured Monte Carlo standard error.
- [ ] **Step 2:** extend `walkforward.score_rows` to score a joint cell distribution directly when the candidate supplies one (falling back to the composed form otherwise), so the simulator and the baseline are both scored as `−log P(realised cell)`. Test that the composed path is unchanged for existing candidates — every committed report must still reproduce.
- [ ] **Step 3:** expose as `--candidate hazard`. Commit.

---

### Task 4: E1 and E2 — does the paradigm pay?

- [ ] **E1 — v1 features** (`base` only): isolates the paradigm from the data. `--candidate hazard --name hazard_e1 --blocks base`.
- [ ] **E2 — shipped features** (`base,external,trajectory,notice,context,opponent_adjusted`): the real candidate. `--name hazard_e2`.
- [ ] For each, report the joint-outcome log-loss against the incumbent's composed joint, the winner marginal against the incumbent's winner, method and round macro-F1, all four slices, and the zero-mass-cell fraction.
- [ ] Apply the SP1 simulator bar. **Report the numbers and stop for a call if the result is marginal** — specifically, if the joint improves by more than 0.005 but less than the 0.01 bar, or if the joint clears while the winner marginal regresses by more than σ_seed (which is the case the spec's hybrid fallback was written for).
- [ ] Commit the reports.

---

### Task 5: The torch variant (only if E2 is close or better)

Gate: run this only if E2's joint-outcome log-loss is within 0.02 of the bar. If E2 is far off, the paradigm has been measured and the honest move is to stop and record it rather than spend a day on a second base learner.

- [ ] Extend the existing multi-task trunk with a 5-way hazard head and a decision head, trained on the hazard rows; seed-ensembled and temperature-calibrated as the current torch member is. Evaluate as `hazard_torch`, and as a blend member alongside XGBoost (the deployed scorer is a blend, so a simulator that only helps as a blend member is still a shippable result).
- [ ] Commit the reports.

---

### Task 6: Decision, fresh seeds, ship or revert

- [ ] Fresh-seed confirmation (seeds 5–9) for anything that clears, against a fresh-seed paired incumbent.
- [ ] Write `models/walkforward/sp3_decision.json` via a regenerable `scripts/sp3_decision.py`: every experiment, the bar quoted verbatim, which branch applies, the Monte Carlo error, the zero-mass fraction, and the decision.
- [ ] **If it ships:** deployment is a bigger change than SP2.2's — the simulator replaces the method and round heads and supplies a winner marginal. Work through `src/mma/inference.py` (a simulator-backed predictor alongside `BlendedPredictor`), `src/mma/versioning.py` (hash the hazard artifacts and any simulation parameters — SP2.2's lesson: anything that moves a probability must be hashed, including `n_runs` and the smoothing α), `scripts/predict_upcoming.py` and `src/mma/prospective.py` (prediction records gain the full joint distribution), `app.py` (an outcome table — "A by KO/TKO in R2: 11%" — plus P(distance), which is the user-visible payoff of this whole sub-project), and `scripts/build_display_priors.py` (a coherent simulator may make the mean-matching correction unnecessary — measure rather than assume).
- [ ] **If it does not ship:** revert the wiring, keep `hazard.py`, `simulator.py`, their tests and every report, and record the result as prominently as SP2's seven rejections.

---

### Task 7: Document and merge

- [ ] README: the simulator's story and its numbers, positive or negative; if it ships, replace the outputs section since the model now predicts a joint distribution rather than three marginals.
- [ ] Plan Completion notes; spec SP3 marked done with what it implies for SP4.
- [ ] Full suite, byte-identity checks, `--no-ff` merge to main.

---

## Completion notes (filled in during execution)

- Hazard rows / decision rows built: _…_
- Monte Carlo standard error at N=10,000: _…_ ; zero-mass cell fraction: _…_
- E1 (v1 features): joint _…_, winner _…_
- E2 (shipped features): joint _…_, winner _…_, method F1 _…_, round F1 _…_
- Torch variant: _…_
- Fresh-seed: _…_
- Decision: _…_
- Follow-ups: _…_
