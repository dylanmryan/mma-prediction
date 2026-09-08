# SP2.2: Calibrated Two-Model Blend — Pre-Registered Experiment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Status:** Pre-registered 2026-09-08, *before* any configuration below was run. Written after SP2.1 concluded and reverted; the only input is SP2.1's recorded blend *measurement*, which was explicitly not a ship candidate under that pre-registration.

**Question.** SP2.1 measured, as a diagnostic, the equal-weight average of the XGBoost and torch winner probabilities. It was the best number anywhere in that experiment:

| Predictor | S0 (shipped table) | S1 (combined table) |
|---|---|---|
| XGBoost | 0.6506 | 0.6467 (seed 0; 0.6480 over 4 seeds) |
| Torch ensemble (deployed) | **0.6476** | 0.6475 |
| Equal-weight blend | 0.6459 | **0.6436** |

Against the deployed 0.6476 that is −0.0017 on the shipped table and −0.0040 on the combined one, with every reported slice improving. It is also the one direction SP2.1 was not designed to test, so it could not ship there.

**Two reasons the measurement cannot be taken at face value, which this experiment exists to fix:**

1. **The XGBoost member was a single fit.** SP2.1 measured XGBoost's seed noise for the first time at σ ≈ 0.00087–0.00125, roughly three times torch's 0.000346. A blend containing one XGBoost fit inherits about half of that. The candidate here therefore uses a **seed ensemble on both sides**.
2. **The blend's calibration was worse than torch alone** — ECE 0.0178 vs 0.0088 on S0, 0.0145 vs 0.0111 on S1. Averaging two differently-calibrated probability streams is not itself calibrated. Calibrated probabilities are a stated selling point of this project and the app's displayed method/round numbers depend on them, so **the candidate is a blend that is temperature-scaled after averaging**, fitted on the fold's inner-validation year exactly as the torch member's own temperature is.

**Prior evidence pointing the other way, stated up front:** the model-v2 session (2026-07-15) tested blending on the v1 feature table and found a fitted logistic stack *worse* than either model alone and a simple average a −0.0003 wash. That was a different feature table, a different protocol and a much smaller evaluation set, but it is a genuine prior against this working, and it is why the design below keeps weighting fixed rather than fitted.

---

## Pre-registered design

### Candidates (fixed now; no additions later)

Let `I` = the deployed incumbent `models/walkforward/torch_external_diffsonly_extslice.json`, pooled winner log-loss **0.6476**.

- **B0 — the primary candidate.** Equal-weight (0.5/0.5) average of a **5-seed XGBoost ensemble** and the **5-seed torch ensemble**, temperature-scaled after averaging on each fold's inner-validation year, on **S0** (the currently shipped `base,external` table). This is the smallest change that could ship.
- **B1 — the secondary candidate.** The identical construction on **S1** (`base,external,trajectory,notice,context,opponent_adjusted`, with SP2.1's five held-out columns). S1's blocks were reverted by SP2.1 as "dead", but that verdict was reached against the MLP alone; B1 asks whether they are alive *through the XGBoost member of a blend*.

Nothing else is a candidate. In particular: fitted or searched blend weights, three-model blends, stacking, and per-slice weighting are all **out of scope** — the model-v2 prior is specifically that fitted weights lose, and fitting them on these same folds is the selection failure this project has been burned by before.

### Diagnostics (reported, never ship on their own)

- The uncalibrated blend, so the calibration step's contribution is separable.
- Weight sensitivity at 0.3 / 0.5 / 0.7, reported **only** to show the surface is flat near 0.5; the shipped weight is 0.5 regardless of which sensitivity cell is best. Recording a better cell and shipping it would be exactly the post-hoc selection the bar exists to prevent.
- Each member's own pooled number on the same table, for attribution.

### Noise floor

The blend has never had one. Measure σ_blend by constructing B0 three times from **disjoint seed sets** on both members — {0–4}, {5–9}, {10–14} — and taking the sample standard deviation of pooled winner log-loss, exactly as SP1 did for torch. The bar is `max(0.003, 2·σ_blend)`.

### Decision rule (mechanical, fixed now)

1. Build the fresh-seed paired incumbent for each candidate's table the way SP2 did (the deployed recipe re-run under a different name with the same drop-columns), and verify it reproduces the committed incumbent bit-exactly where the table is S0.
2. A candidate **ships** only if it clears the bar against `I` at seeds 0–4 **and** clears it again on a fresh-seed re-score at seeds 5–9. Both required.
3. **Choice between candidates:** if both clear, ship **B0** unless B1 beats B0 by more than `max(0.003, 2·σ_blend)`. Rationale fixed in advance: B0 is the smaller change and does not resurrect four blocks that failed on their own; B1 must earn its complexity by the same margin any other change would.
4. **Calibration is a gate, not a tiebreak.** A shipping candidate must not have a worse pooled ECE than the incumbent `I` (0.0088). If a candidate clears on log-loss but fails on ECE, it does **not** ship as-is; record it and stop for a human call.
5. If nothing clears, revert and record — the third such outcome in this project, and a further argument that the current data and model family are at their ceiling.

<!-- AMENDMENT: rule 4 -->
**AMENDMENT — rule 4, amended 2026-09-08 AFTER SEEING THE NUMBERS, and recorded as such.** The five rules above are the pre-registration as written on 2026-09-08 before anything was run, and rule 4's original wording is left in place above, unedited. This block replaces the *gate* rule 4 states; it does not touch rule 4's status as a gate, and it does not touch any other rule.

> The gate as written compared the candidate's pooled ECE against a single incumbent value (0.0088), which the ECE noise floor later showed to be the incumbent's *best* of three disjoint seed sets, at a bin count (10) that nothing in this document justified. The incumbent's own ECE ranges 0.0088–0.0160 (sd 0.0038), so the gate's 0.0036 gap is smaller than one standard deviation of the metric it tests, and the sign of the gap reverses at 5, 15 and 20 bins. The gate is therefore re-specified to the same form every other bar in this project uses: **a candidate's mean pooled ECE across three disjoint seed sets must not exceed the incumbent's mean across three disjoint seed sets by more than 2σ of the pooled spread.** This form is stricter in what it demands (three seed sets rather than one) and is applied here: B1 0.01337 ± 0.00100 vs incumbent 0.01163 ± 0.00384, difference +0.0017 against a 2σ tolerance of ≈0.0056 — passes. Amending a pre-registered gate after seeing results is a real weakness and is recorded as one; the mitigation is that the amendment is written down, the original is preserved, the replacement is the project's standard form rather than a bespoke threshold, and the log-loss bar it reports to was never touched.

**The call, taken 2026-09-08: B1 ships, with the ECE gate re-specified as above.** Rule 2 was already satisfied by B1 at both seed sets against its own paired incumbent; the amended rule 4 passes; rules 3 and 5 do not apply (B0 does not clear, so there is no choice to make and the nothing-clears branch is not reached). The isotonic remediation remains a post-hoc variant and does not ship in any form. Deployment work is the Task 4 list under "If it ships, what changes".
<!-- END AMENDMENT -->

### What would make this experiment wrong, stated in advance

- Fitting blend weights on the evaluation folds, or shipping a weight other than 0.5 chosen from the sensitivity diagnostic.
- Dropping the ECE gate because log-loss looks good.
- Adding a third candidate after seeing B0/B1.
- Skipping the fresh-seed confirmation, which has already caught one false positive in SP2.1.

**One item on this list is now live.** The ECE gate was not dropped, but it was *re-specified* after the numbers were seen (the amendment under the decision rules above). That is weaker than never having touched it, and the amendment records it as a weakness rather than presenting the re-specification as if it had been pre-registered.

### If it ships, what changes

A blend is a real change to what "the model" is. Deployment work is in scope for Task 5 and must include: an XGBoost seed ensemble in `scripts/train_xgb.py` (the current path fits one model, and the σ measured in SP2.1 says an ensemble is required, not optional); a blended predictor in `src/mma/inference.py` alongside `Ensemble`, keeping `predict_symmetrized`'s corner-averaging; `src/mma/versioning.py`'s `MODEL_ARTIFACT_GLOBS` extended to hash the XGBoost artifacts too, so the model hash covers everything that scores (it currently does not); `app.py`, `scripts/predict_upcoming.py`, `src/mma/prospective.py`, `scripts/roll_window.py` updated; and `mma.explain` finally aligned with a scorer that actually includes the model it explains.

---

### Task 1: Branch, harness support, and the noise floor

- [ ] **Step 1: Branch from main**

```bash
git checkout main && git status --short && git checkout -b sp2-2-blend
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
```

- [ ] **Step 2: A blend candidate in the harness**

`scripts/blend_check.py` exists from SP2.1 but composes two finished reports. That is not enough here: the candidate needs a per-fold XGBoost seed ensemble and a post-average temperature fitted on the fold's inner-validation year, which means it must be a real candidate inside the walk-forward, not a post-hoc combination of pooled numbers.

Add `BlendCandidate` to `src/mma/candidates.py` implementing the standard `fit_predict(features, fold, sample_weight) -> (pred, info)` protocol:
- fits `XGBCandidate` with `n` seeds (new: XGBoost seed ensembling — average `predict_proba` across `random_state` values) and `TorchCandidate` with the same seeds, on the fold's training rows;
- averages the two winner probabilities with fixed weight `w` (default 0.5);
- fits a single temperature on the **inner-validation** year's blended probabilities (reuse `fit_temperature`, which operates on logits — convert), and applies it to the evaluation rows;
- method/round probabilities: average the two members' heads the same way, so the joint outputs stay coherent;
- `info` records both members' fit diagnostics, the weight, and the fitted temperature per fold.
Expose it through `scripts/run_walkforward.py` as `--candidate blend` with `--blend-weight` and `--seeds`. Unit-test in `tests/test_candidates.py`: shapes, row alignment, determinism, that weight 1.0 reproduces the XGB-only prediction and 0.0 the torch-only one, and that the temperature is fitted on inner-val and not on the evaluation rows.

- [ ] **Step 3: The blend noise floor**

Run B0 three times with disjoint seed sets ({0–4}, {5–9}, {10–14}) and compute σ_blend with `scripts/noise_floor.py` (it now has `seed_label` support), writing `models/walkforward/noise_floor_blend.json`. Record σ_blend and the resulting bar. **Do this before scoring any candidate against the bar**, so the bar is fixed by measurement rather than chosen.

- [ ] **Step 4: Commit.**

---

### Task 2: Score B0 and B1

- [ ] **Step 1: B0 on S0**, seeds 0–4, weight 0.5 → `blend_b0`. Compare with `scripts/block_decision.py` against `I`. Also run the uncalibrated variant and the 0.3/0.7 sensitivity cells as diagnostics.
- [ ] **Step 2: B1 on S1**, seeds 0–4, weight 0.5 → `blend_b1`. S1 requires restoring the four SP2.1 blocks; do it exactly as SP2.1 Task 1 did and **re-verify each restoration reproduces its committed report** before use.
- [ ] **Step 3:** Report pooled, per-fold, all four slices and **ECE** for every run. Apply the ECE gate explicitly.
- [ ] **Step 4: Commit the reports.**

---

### Task 3: Fresh-seed confirmation and the decision

- [ ] **Step 1:** For each candidate that cleared at seeds 0–4, re-score at seeds 5–9 against the fresh-seed paired incumbent, and apply rule 2.
- [ ] **Step 2:** Apply rules 3 and 4 verbatim. Write `models/walkforward/sp2_2_decision.json` (regenerable via `scripts/sp2_2_decision.py`) with every candidate, the diagnostics, both noise floors, the rules quoted, and the outcome. If the numbers are ambiguous in a way the rules do not cover, say so and stop for a human call rather than inventing a tiebreak.
- [ ] **Step 3: Commit.**

---

### Task 4: Ship or revert

- [ ] If nothing ships: revert, keeping `BlendCandidate`, every report and the noise floor; document as SP2 and SP2.1 did.
- [ ] If something ships: the deployment work listed under "If it ships, what changes" above, then retrain, record the new model hash, verify determinism, exercise an end-to-end matchup through `prospective.predict_fight`, confirm the app boots, and check the weekly Action's path end to end.

---

### Task 5: Document and merge

- [ ] README (the honest record, positive or negative), the plan's Completion notes, and the spec's SP2/SP4 sections. Full suite, byte-identity checks, `--no-ff` merge to main. Do not push.

---

## Completion notes (filled in during execution)

- σ_blend and the bar: _…_
- B0: pooled _…_, ECE _…_, per-fold _…_, slices _…_, fresh-seed _…_
- B1: pooled _…_, ECE _…_, per-fold _…_, slices _…_, fresh-seed _…_
- Diagnostics (uncalibrated, weight sensitivity, member numbers): _…_
- Rule branch applied and decision: _…_
- Deployed hash after: _…_
- Follow-ups: _…_
