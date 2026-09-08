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

- [ ] **Step 1: Branch**

```bash
git checkout main && git status --short && git checkout -b sp2-1-capacity
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
```
Expected: clean; `532 passed, 1 skipped`.

- [ ] **Step 2: Restore `trajectory`, `notice`, `context`**

Each has a commented-out registration in `feature_blocks.py` and its supporting module kept (`mma.glicko` + `run_glicko` in `build_ratings.py`; `mma.notice` + `data/external/fight_notice.parquet`; `mma.context`). Uncomment the registrations and restore whatever wiring in `features.py` / `inference.py` the block comments say was removed. For `notice` and `context`, register them in their **shipping form** — the flags that failed the constant-vector leak check (`notice_unknown`; `home_country_a`/`home_country_b`) stay in the table but go into `tensors.DROPPED` + `models/xgb.MODEL_EXCLUDED`, exactly as `external`'s artifact flags do.

- [ ] **Step 3: Restore `opponent_adjusted` from git history**

Its accumulators were fully reverted; recover them from commit `2ad725c` (`git show 2ad725c^:src/mma/history.py` etc. — note that commit *is* the revert, so the implementation is in its parent's tree or in the diff). Restore the block and its unit tests.

- [ ] **Step 4: Verify each restoration reproduces its SP2 measurement**

For each restored block, rebuild `base,external,<block>` and re-run the XGB walk-forward, comparing against that block's committed SP2 report:

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external,trajectory
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_restore_trajectory
```
The pooled winner log-loss must match the committed `models/walkforward/xgb_trajectory.json` to 4 decimals. Repeat for `notice`, `context`, `opponent_adjusted` against their committed reports. **Any mismatch is a restoration bug — fix it before proceeding.** (Reports named `*_restore_*` are scratch; delete them before committing.)

- [ ] **Step 5: Build S1 and run the leakage and parity gates**

```bash
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external,trajectory,notice,context,opponent_adjusted | tail -3
~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_serving_parity.py tests/test_feature_blocks.py tests/test_external.py tests/test_context.py -q
```
Blocking: truncation-invariance and serving parity must pass on the combined table. Run the data-level constant-vector leak check over every restored block's modelled columns, on the rows where exactly one corner's external data is unavailable — the combined table has more channels for the coverage artifact than any single block did, so this is the highest-risk step in the experiment. Report the column count of S1 and per-column coverage.

- [ ] **Step 6: Commit**

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

- [ ] **Step 1: A1 (S1, deployed config)**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_a1_combined
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_a1_combined
~/.venvs/mma/bin/python scripts/block_decision.py --candidate models/walkforward/torch_a1_combined.json --incumbent models/walkforward/torch_external_diffsonly_extslice.json
```
A0 is the committed incumbent; no run needed. Record A1's pooled, per-fold and slice numbers, and the XGB number (the capacity hypothesis predicts XGB improves more than torch here).

- [ ] **Step 2: Commit the reports and the numbers.**

---

### Task 3: The search — arms C and T

**Files:** `scripts/config_search.py` (new)

- [ ] **Step 1: Write `scripts/config_search.py`**

Samples the 25 configurations from the fixed space above with `numpy.random.default_rng(0)`, and for a given `--blocks` set and `--drop-columns` runs each through the walk-forward harness with `--seeds 0,1`, writing one report per config under `models/walkforward/search/<arm>/` plus a ranked summary JSON. It must reuse `scripts/run_walkforward.py`'s machinery rather than reimplementing scoring (import it, or shell out — either is fine, say which). Print progress; the whole arm is ~25 × 2 seeds and will take a while, so make it resumable (skip a config whose report already exists).

- [ ] **Step 2: Run arm C (control: S0, searched)**

```bash
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external
~/.venvs/mma/bin/python scripts/config_search.py --arm C --blocks base,external
```

- [ ] **Step 3: Run arm T (treatment: S1, searched)**

```bash
~/.venvs/mma/bin/python scripts/build_features.py --blocks base,external,trajectory,notice,context,opponent_adjusted
~/.venvs/mma/bin/python scripts/config_search.py --arm T --blocks base,external,trajectory,notice,context,opponent_adjusted
```

- [ ] **Step 4: Best-of-arm at 5 seeds**

Take each arm's top config by 2-seed mean and run it as a full 5-seed ensemble (`--seeds 0,1,2,3,4`), named `torch_C_best` / `torch_T_best`. Apply `block_decision.py` against the shipped incumbent. Record the distribution of the 25 configs per arm (min / median / max), not just the winner — the spread says how much of the "best" is selection.

- [ ] **Step 5: Commit the search outputs and the summary.**

---

### Task 4: Fresh-seed confirmation and the attribution decision

- [ ] **Step 1: Fresh seeds for whichever arms cleared at 5 seeds**

Run the winning config(s) with `--seeds 5,6,7,8,9`, and build the matching fresh-seed paired incumbent the way SP2 did (v-recipe with the same `--drop-columns` on the same table, verified to reproduce `torch_v1_seeds5`-style bit-exactly where applicable).

- [ ] **Step 2: Arm A2 — does recency stack?**

Only on the winner of {C, T}, and only if it cleared: re-run with `--half-life 8`. This is the fifth and last arm.

- [ ] **Step 3: Apply the attribution rule from the pre-registration verbatim**

Write `models/walkforward/sp2_1_decision.json` with every arm's pooled numbers, the search distributions, the fresh-seed results, the rule as a string, and the resulting decision. Do not deviate from the rule; if the numbers are ambiguous in a way the rule does not cover, say so explicitly and stop for a human call rather than inventing a tie-break.

- [ ] **Step 4: Commit.**

---

### Task 5: Ship or revert, document, merge

- [ ] **Step 1:** If something ships, update the train scripts' defaults (re-deriving the refit budget on the new table/config via `scripts/refit_decision.py --reports <set>` as SP2 Task 13 did), retrain, record the new model hash, verify determinism, exercise an end-to-end matchup, and confirm the app boots.
- [ ] **Step 2:** If nothing ships, revert the feature wiring (keeping the reusable modules and every report), exactly as SP2 did for its seven rejections.
- [ ] **Step 3:** README + plan Completion notes + spec: the question, the design, the arms, the numbers, the attribution, and what it implies for SP3. A negative result here is a genuinely useful finding — it would say the box-score data *and* the architecture are both near their ceiling, which is the strongest possible argument for the simulator paradigm.
- [ ] **Step 4:** Full suite, `make_dataset.py` / `build_features.py` byte-identity, merge to main with `--no-ff`. Do not push.

---

## Completion notes (filled in during execution)

- Restoration verification: _each block's reproduced vs committed pooled number_
- S1: _columns, coverage, leak-check result_
- A1 (combination, deployed config): _torch …, XGB …_
- Arm C (control): _25-config distribution min/median/max; best 5-seed …_
- Arm T (treatment): _25-config distribution; best 5-seed …_
- Fresh-seed: _…_
- A2 (recency on the winner): _…_
- **Attribution and decision:** _…_
- Deployed hash after: _…_
- Follow-ups: _…_
