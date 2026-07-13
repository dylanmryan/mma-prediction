# Elo v1.1 experiments

**Branch:** `elo-v1-1`. **Goal:** test three hypotheses about the Elo/feature
layer against the existing baseline, using a pre-registered, mechanically
applied acceptance rule. **Hard rule honored throughout:** no code path read
2024+ outcomes; every decision below uses 2021-2023 validation metrics only.

## Baseline (committed, unchanged)

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| XGBoost (46 features) | 0.5952 | 0.6578 | 0.2329 |
| Torch 5-seed ensemble | 0.6058 | **0.6537** | 0.2311 |

## Acceptance rule (pre-registered, applied mechanically)

A change is **KEPT** only if:
1. Torch-ensemble validation log-loss improves by **more than 0.002** vs the
   baseline (0.6537), **and**
2. XGBoost validation log-loss does not worsen by more than 0.002.

Everything else is rejected and reverted, regardless of how "close" or
directionally promising the result looks. This report documents negative
results honestly, per the protocol.

## Experiment A: style-Elo ablation

**Question:** do `striking_elo_diff` and `grappling_elo_diff` earn their
keep, or is the plain overall-Elo differential doing all the work?

**Method:** dropped both columns from the feature table in memory
immediately after loading `features.parquet`, via a small `MMA_EXCLUDE_FEATURES`
env-var hook added temporarily to `scripts/train_xgb.py` / `train_torch.py`
(`MMA_EXCLUDE_FEATURES=striking_elo_diff,grappling_elo_diff`). Both models
retrained from scratch on the ablated feature set; no ratings/features
artifacts changed.

| Model | Accuracy | Log-loss | Brier | Δ log-loss vs baseline |
|---|---|---|---|---|
| XGBoost (44 features) | 0.6145 | 0.6573 | 0.2326 | −0.0005 (improved) |
| Torch 5-seed ensemble | 0.5979 | 0.6518 | 0.2304 | −0.0019 (improved) |

**Decision: REJECTED (not kept).** Torch log-loss improved by 0.0019, which
falls short of the required >0.002 threshold. XGBoost log-loss also improved
slightly (no worsening either way), so nothing here breaks rule 2 — the
change simply doesn't clear rule 1's bar. Interesting side note: XGBoost's
*accuracy* jumped substantially (0.5952 → 0.6145) with the ablation, but its
log-loss (the pre-registered metric) barely moved — a reminder that accuracy
and log-loss can diverge, and the protocol correctly ignores the accuracy
swing.

**B trigger check:** removing the style Elos did **not** hurt log-loss by
more than 0.001 on either model (both metrics moved in the *improving*
direction, just not far enough to be kept). Per the protocol, Experiment B
("tune `GRAPPLING_EVENT_WEIGHT`") only runs if removal hurts by >0.001 — it
does not, so **Experiment B is skipped**. The style Elos are, at best,
neutral-to-mildly-net-negative noise on this validation window; there is no
signal that the current 5.0 exchange rate is mis-tuned in a way worth a grid
search.

*Code changes reverted*: `scripts/train_xgb.py`, `scripts/train_torch.py`
(exclude hook), all overwritten model/feature artifacts restored from git.
Working tree confirmed byte-identical to baseline (`git show HEAD:<path> | md5`
matched local `md5` for every touched artifact).

## Experiment B: grappling exchange-rate grid

**Skipped.** Precondition from Experiment A (style Elos hurting log-loss by
>0.001 when removed) was not met — see above. No runs performed, no code
touched.

## Experiment C: Glicko-1 style rating deviation (RD)

**Question:** does a confidence/uncertainty signal alongside the Elo rating
(RD, à la Glicko-1) carry predictive information the model can use — e.g. as
a proxy for "how much do we actually know about this fighter"?

**Implementation** (`src/mma/elo.py`, alongside the existing Elo — Elo math
itself untouched):

- `RD_INITIAL = 350.0`, `RD_FLOOR = 60.0`.
- Growth during layoffs: `RD_new = min(sqrt(RD² + c²·months_inactive), 350)`,
  with `c = sqrt((350² − 60²) / 24) ≈ 70.39`, chosen so a fighter sitting at
  the RD floor who then goes fully inactive for exactly 24 months (2 years)
  has RD grow back to the 350 ceiling by month 24. (Growth saturates *faster*
  in absolute time for fighters starting from a higher RD than the floor,
  which is the correct behavior of the capped-sqrt model, not a bug — see
  the unit tests that existed for this, described below.)
- Shrink per fight: standard Glicko-1 single-game `d²` update,
  `RD_new = 1 / sqrt(1/RD_old² + 1/d²)`, floored at 60, computed from the
  two fighters' pre-fight overall-Elo ratings and the opponent's pre-fight
  RD. RD is purely informational — it never feeds back into the Elo rating
  update itself.
- `run_elo()` extended to track and emit `pre_rd` per fighter per fight
  (chronological single pass, same structure as the existing overall/
  striking/grappling tracking; debutants start at `RD_INITIAL`).
- `build_features()` extended with `rd_diff` (A-minus-B differential, via
  the existing diff machinery) and `rd_total` (sum of both fighters' pre-fight
  RD — a combined-uncertainty signal orthogonal to the relative-uncertainty
  `rd_diff`).

**TDD:** RD math was developed test-first with hand-computed cases before
implementation (`tests/test_elo_rd.py`, `tests/test_elo_rd_engine.py`,
11 + 5 tests, all green): growth with no inactivity is a no-op; growth from
the floor over exactly 24 months lands at 350; growth caps at 350; the
two-debutant shrink case hand-computed to RD ≈ 290.23; a floor-binding case
(raw update ≈ 59.15, floored to exactly 60); a non-floor-binding case
(≈ 63.93); monotonicity in own RD; and an integration test confirming
`pre_rd` grows from the *previous shrunk value* (not a naive reset) across a
5-month gap (350 → 290.23 → 330.11). Also verified: no-contests don't update
RD (skipped exactly like the Elo pass), `pre_rd` never drops below the
floor, and rebuilding `ratings.parquet`/`features.parquet` still passes the
existing no-leakage truncation-invariance test.

**Method:** rebuilt `ratings.parquet` (Elo tuning grid and top-10 peak Elo
table reproduced identically — RD is additive and doesn't touch the Elo
math) and `features.parquet` (8190 rows, 48 columns, +2 for `rd_diff` /
`rd_total`), then retrained both models from scratch.

| Model | Accuracy | Log-loss | Brier | Δ log-loss vs baseline |
|---|---|---|---|---|
| XGBoost (48 features) | 0.6085 | 0.6566 | 0.2322 | −0.0012 (improved) |
| Torch 5-seed ensemble | 0.5952 | 0.6520 | 0.2304 | −0.0017 (improved) |

`rd_diff` ranked #7 in XGBoost winner-model feature importance (0.0269),
ahead of `striking_elo_diff` and `elo_diff` — a genuinely interesting signal,
just not a large enough one on this validation window.

**Decision: REJECTED (not kept).** Same pattern as Experiment A: torch
log-loss improved by 0.0017, short of the required >0.002. XGBoost log-loss
improved too, so rule 2 isn't the blocker — rule 1's bar is.

*Code changes reverted*: `src/mma/elo.py`, `src/mma/features.py`,
`tests/test_features.py` restored to baseline; `tests/test_elo_rd.py` and
`tests/test_elo_rd_engine.py` deleted (per protocol: if every experiment is
rejected, only this document is a committed change, and the RD unit tests
are dropped along with the code they tested). All ratings/features/model
artifacts confirmed byte-identical to baseline via `md5` comparison against
`git show HEAD:<path>`.

## Conclusions

All three planned directions were tried; **none cleared the pre-registered
bar**. This is a genuinely negative result set, reported honestly rather than
cherry-picked or re-thresholded after the fact:

- **Style Elos (striking/grappling) are not pulling meaningful weight** on
  this validation window — removing them is *directionally* neutral-to-mildly-
  positive for both models but not significantly so. There's no evidence the
  current `GRAPPLING_EVENT_WEIGHT = 5.0` exchange rate is mis-tuned enough to
  be worth a grid search (Experiment B's precondition wasn't met), so it was
  correctly skipped rather than run for its own sake.
- **A Glicko-style RD signal is a plausible, well-motivated idea** — the
  math is sound (TDD'd against hand-computed cases), it wires in cleanly
  alongside the existing Elo without touching it, and `rd_diff` shows up as
  a real (if modest) feature-importance signal in XGBoost. But on this
  validation window it improves torch log-loss by only 0.0017, short of the
  0.002 bar, so it doesn't meet the bar for a permanent change.
- **A striking pattern across both rejected experiments**: torch log-loss
  improvements of 0.0017-0.0019 are consistently *just* under the 0.002
  threshold. That's suggestive of two things worth flagging for a future
  pass (not acted on here, since the rule is mechanical and pre-registered):
  (1) the current 1,507-fight validation set may be too small for torch's
  seed-to-seed noise (mean seed spread ~0.086-0.094) to cleanly resolve
  effects in this size range, and (2) it's possible a *combination* of small
  wins (e.g., RD *and* something else) could clear the bar even if no single
  change does alone — untested here since the protocol runs each experiment
  independently.
- **No code changes were kept.** The working tree at the end of this session
  contains only this document; `README.md` result numbers are unchanged
  because nothing changed. The full test suite is green: 125 passed, 1
  skipped (same as session start).
