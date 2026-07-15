# Model v2 experiments

**Branch:** `model-v2`. **Goal:** disciplined attempt to beat the committed v1
models via a blend, hyperparameter search, and (if needed) new point-in-time
features, using a pre-registered, mechanically applied acceptance rule.
**Hard rule honored throughout:** the 2024+ test set is spent (one-time eval
already done, see `models/final_test_metrics.json` and the README's "Final
held-out test results" section) — no code in this session read 2024+
outcomes for any decision. All selection happened on the 2021-2023
validation years.

## Baseline (committed, unchanged going in)

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| XGBoost (46 features) | 0.5952 | 0.6578 | 0.2329 |
| Torch 5-seed ensemble | 0.6058 | **0.6537** | 0.2311 |

## Acceptance rule (pre-registered, applied mechanically)

The final v2 configuration ships **only if** it improves validation winner
log-loss by **more than 0.004** vs 0.6537 — a deliberately higher bar than a
single experiment's, because this protocol allows multiple shots (E1-E5) and
the bar must account for that multiple-comparisons risk. Method/round heads:
an independent small win (>0.01 macro-F1 at no winner cost) may also ship on
its own, judged separately from the winner bar.

Dev metric throughout: validation winner log-loss (accuracy secondary).

**Baseline suite status:** the full test suite was verified green very
recently (129 passed + 1 skipped). A fresh baseline run at session start was
killed after 10+ minutes of grinding under heavy machine load (unrelated
workloads; the suite was passing when killed — 54%+ complete, zero
failures). Baseline assumed from that recent verification; the suite is
re-run in full at the end of the session.

## Running results table (validation winner metrics, 1,507 fights)

| Exp | Candidate | Acc | Log-loss | Δ LL vs 0.6537 | Verdict |
|---|---|---|---|---|---|
| — | Torch 5-seed baseline (committed) | 0.6058 | 0.6537 | — | reference |
| — | XGB baseline (committed) | 0.5952 | 0.6578 | +0.0041 | reference |
| E1 | LR blend [XGB, torch, Elo], CV OOF | 0.6025 | 0.6557 | +0.0020 | worse than torch alone |
| E1 | Simple avg (XGB + torch, v1 models) | 0.6111 | 0.6534 | −0.0003 | wash |
| E2 | Best-of-40 XGB config | 0.6125 | 0.6554 | +0.0017 | XGB improved, still < torch |
| E3 | Best-of-25 torch config, 2-seed mean | — | 0.6516 | −0.0021 | carried to 10-seed |
| E3 | Best torch config, 10-seed ensemble | 0.6145 | 0.6509 | −0.0028 | best single model |
| E4 | v2 features (53 cols), XGB best params | 0.6038 | 0.6572 | +0.0035 | features hurt XGB |
| E4 | v2 features, best torch cfg, 10-seed | 0.6052 | 0.6536 | −0.0001 | features hurt torch |
| E5 | Simple avg (best XGB + best torch) | 0.6098 | **0.6505** | **−0.0032** | best overall — still < bar |
| E5 | LR blend [best XGB, best torch, Elo], CV OOF | 0.6138 | 0.6532 | −0.0005 | stack loses to average |

**Bar: < 0.6497 required (−0.004). Best achieved: 0.6505 (−0.0032). NOT
CLEARED — nothing ships.** Method-head clause: apparent +0.019 macro-F1 on
the v2-features ensemble failed fresh-seed replication (0.3635 vs committed
0.3639) — not invoked. Details per experiment below.

## E1: Stacked blend

**Method:** logistic regression over [XGB winner prob, torch ensemble prob,
Elo expected score (from `elo_diff`)] on the 1,507 validation fights. Blend
weights fit via 5-fold CV *within* validation; the reported dev metric is the
out-of-fold (OOF) blended prediction, which is the honest estimate. A
full-validation fit is shown only for reference (it is optimistic by
construction, being fit and scored on the same rows).

| Predictor | Accuracy | Log-loss | Brier |
|---|---|---|---|
| XGB alone | 0.5952 | 0.6578 | 0.2329 |
| Torch ensemble alone | 0.6058 | 0.6537 | 0.2311 |
| Elo alone | 0.5740 | 0.6777 | 0.2424 |
| Simple avg (XGB+torch) | 0.6111 | 0.6534 | — |
| **Blend, 5-fold CV OOF (honest)** | 0.6025 | **0.6557** | 0.2319 |
| Blend, full-val fit (optimistic ref) | 0.6072 | 0.6529 | 0.2306 |

Fold weights were stable (xgb ≈ 1.6-1.9, torch ≈ 2.0-2.6, elo ≈ 0.4-0.9).

**Decision: no gain.** The honest OOF blend (0.6557) is *worse* than the
torch ensemble alone (0.6537) — the stack learns nothing beyond what the
torch model already encodes, and the extra fitted parameters cost more than
the XGB/Elo signals add. The simple unweighted XGB+torch average is a
−0.0003 wash. E1 contributes nothing toward the bar on its own; it is
revisited in E5 only as a refit over the best E2/E3 models.

## E2: XGBoost hyperparameter search

**Method:** random search, 40 configs (seeded rng(0)), over
max_depth {3..6}, learning_rate {0.02..0.1 log-uniform}, min_child_weight
{1..10}, subsample/colsample_bytree {0.6..1.0}, reg_lambda {0.5..8
log-uniform}; existing early stopping (2000 rounds cap, 50-round patience) on
validation kept as-is. Winner head only. Committed artifacts untouched.

Top 5 by validation log-loss:

| # | depth | lr | mcw | subsample | colsample | lambda | best_iter | Acc | LL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 0.053 | 10 | 0.89 | 0.82 | 6.68 | 128 | 0.6125 | **0.6554** |
| 2 | 3 | 0.021 | 7 | 0.89 | 0.61 | 4.09 | 449 | 0.6145 | 0.6554 |
| 3 | 4 | 0.040 | 4 | 0.80 | 0.99 | 4.30 | 162 | 0.6151 | 0.6554 |
| 4 | 6 | 0.091 | 8 | 0.65 | 0.95 | 0.59 | 35 | 0.5965 | 0.6559 |
| 5 | 4 | 0.041 | 6 | 0.97 | 0.62 | 3.81 | 142 | 0.6138 | 0.6562 |

Distribution over all 40: min 0.6554, median 0.6579, max 0.6625; 18/40 beat
the committed XGB baseline (0.6578).

**Read:** the best configs improve XGB by −0.0024 log-loss (0.6578 → 0.6554)
and lift accuracy to ~0.612-0.615 — but the median config lands exactly on
the baseline, i.e. the committed hand-picked params were already near the
center of the reachable range, and the "best of 40" number carries selection
optimism from picking the winner on the same validation set. Even taking
0.6554 at face value, tuned XGB remains worse than the torch ensemble
(0.6537). E2's value, if any, is as a slightly stronger blend ingredient for
E5. Best config carried forward: #1 (depth 5, lr 0.053, mcw 10,
subsample 0.89, colsample 0.82, lambda 6.68).

## E3: Torch hyperparameter search

**Method:** random search, 25 configs (seeded rng(0)), over hidden
{(128,64),(256,128),(192,96,48)}, dropout {0.2..0.4}, lr {3e-4..3e-3
log-uniform}, weight_decay {1e-5..1e-3 log-uniform}, loss weights method
{0.25..0.75} / round {0.1..0.4}, embedding dim {4,8}. Each config trained on
2 seeds (0,1) with per-seed temperature scaling; ranked by 2-seed mean
calibrated validation log-loss. The committed v1 defaults were run through
the identical harness as a control ("config −1") so the comparison is
within-protocol and within-torch-version. (Ops note: torch training
initially stalled on glacial reads of libtorch from the iCloud-synced
project venv; the search ran from a scratch venv on local disk with
identical package versions — torch 2.13.0, xgboost 3.2.0 — and the control
reproduced v1's per-seed numbers exactly, 0.6587/0.6554.)

Top 5 + control by 2-seed mean log-loss:

| # | hidden | dropout | lr | wd | m_scale | r_scale | emb | seed LLs | mean LL |
|---|---|---|---|---|---|---|---|---|---|
| 19 | (128,64) | 0.303 | 2.5e-3 | 1.4e-5 | 0.67 | 0.12 | 8 | 0.6533/0.6500 | **0.6516** |
| 4 | (128,64) | 0.277 | 3.0e-3 | 9.2e-4 | 0.59 | 0.30 | 8 | 0.6548/0.6500 | 0.6524 |
| 21 | (128,64) | 0.245 | 4.2e-4 | 4e-5 | 0.54 | 0.27 | 8 | 0.6549/0.6508 | 0.6528 |
| 15 | (192,96,48) | 0.394 | 3.4e-4 | 5.3e-4 | 0.74 | 0.39 | 8 | 0.6543/0.6528 | 0.6535 |
| 17 | (192,96,48) | 0.385 | 5.8e-4 | 1.2e-4 | 0.47 | 0.38 | 8 | 0.6536/0.6536 | 0.6536 |
| −1 (control, v1 defaults) | (128,64) | 0.3 | 1e-3 | 1e-4 | 0.5 | 0.25 | 4 | 0.6587/0.6554 | 0.6571 |

Distribution over the 25 sampled configs: min 0.6516, median 0.6550, max
0.6575. The control ranked 23rd of 26 — the search found real headroom over
the hand-picked v1 config (all top-5 use embedding dim 8; higher lr + tiny
weight decay dominates).

**10-seed ensemble of the best config** (#19) on v1 features, per-seed
temperature scaling, seeds 0-9:

| Seeds in ensemble | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Cumulative val LL | .6533 | .6504 | .6496 | .6502 | .6512 | .6509 | .6509 | .6507 | .6508 | .6509 |

Final: **0.6509 log-loss / 0.6145 accuracy / 0.2298 Brier** (n=1507).
"More seeds is usually free improvement" — verified as *marginal* here:
10-seed vs 5-seed is −0.0003, within the ±0.001 noise floor visible in the
trajectory. Δ vs committed baseline: −0.0028 (0.6537 → 0.6509), short of the
0.004 bar on its own, and this number carries best-of-26 selection optimism.

## E4: New point-in-time features

**Trigger:** E1 contributed nothing and E2's best (0.6554) is still above the
torch baseline, so the E1-E3 running total was clearly short of the bar; E4
was run per protocol (E3 results pending at the time E4 started; both legs
finished before the E5 decision).

**Added** (all built inside the existing chronological accumulators, so
point-in-time correct by construction):

- `five_round_fights_diff` — count of past 5-round bouts (championship-pace
  experience)
- `title_fights_diff` — count of past title fights
- `avg_opp_elo_wins_diff` — average pre-fight overall Elo of opponents the
  fighter has *beaten* (strength-of-victims signal)
- `ko_losses_diff` — count of losses by KO/TKO (chin proxy)
- `wc_change_a` / `wc_change_b` — flag: this fight is in a different weight
  class than the fighter's previous fight
- `height_reach_diff` — height×reach interaction (frame-size proxy),
  A-minus-B differential

Feature table: 8,190 rows × 53 columns (46 → 53). **Leakage guard:** the
truncation-invariance test compares *every* column of the rebuilt table
against a table built from pre-2015-truncated fights — all 53 columns,
including the 7 new ones, passed (2,998 pre-2015 rows equal). New unit
tests cover the accumulators (debut zeros, KO-loss counting, wc-change
semantics, five-round/title counting, avg-opp-Elo-of-wins) and the feature
plumbing (diff signs under corner swap, NaN propagation).

**XGB leg** (winner, same early stopping):

| Config | Features | Acc | LL | Δ LL vs same-params-old-features |
|---|---|---|---|---|
| Baseline params | v1 (46) | 0.5952 | 0.6578 | — |
| Baseline params | v2 (53) | 0.6005 | 0.6588 | +0.0010 (worse) |
| Best-E2 params | v1 (46) | 0.6125 | 0.6554 | — |
| Best-E2 params | v2 (53) | 0.6038 | 0.6572 | +0.0018 (worse) |

`height_reach_diff` ranked #4 in importance in both runs, but overall
validation log-loss *worsened* — the new columns add more variance than
signal for XGB on this window. Torch leg: see E5 assembly below.

## E5: Final assembly and decision

**Candidates on the 1,507 validation fights** (blend weights, where fitted,
via 5-fold CV within validation; OOF metric reported):

| Candidate | Accuracy | Log-loss | Δ LL vs 0.6537 |
|---|---|---|---|
| Committed baseline (torch 5-seed, v1 cfg) | 0.6058 | 0.6537 | — |
| Best-E2 XGB | 0.6111 | 0.6555 | +0.0018 (worse) |
| Best-E3 torch, 10-seed | 0.6145 | 0.6509 | −0.0028 |
| **Simple avg (best XGB + best torch)** | 0.6098 | **0.6505** | **−0.0032** |
| LR blend [XGB, torch, Elo], CV OOF | 0.6138 | 0.6532 | −0.0005 |
| LR blend [XGB, torch], CV OOF | 0.6138 | 0.6529 | −0.0008 |
| Best-E3 torch 10-seed on v2 features | 0.6052 | 0.6536 | −0.0001 |

Same story as E1: the fitted stack is *worse* than a plain average — with
only two strong, highly correlated inputs, the logistic weights just add
variance. The best honest candidate is the simple average at 0.6505.

**Decision — pre-registered bar applied mechanically:** the bar requires
beating 0.6537 by MORE than 0.004, i.e. validation winner log-loss below
0.6497. Best candidate: 0.6505. **NOT CLEARED (gain 0.0032 <
0.004). v2 does not ship; no winner-model change is committed.** The gain is
also inflated by selection (best-of-26 torch configs, best-of-40 XGB
configs, and best-of-7 assembly variants, all chosen on the same validation
years), so the honest expected out-of-sample gain is smaller still.

**Method-head clause check:** the v2-features 10-seed ensemble (seeds 0-9)
showed method macro-F1 0.3828 vs committed 0.3639 (+0.019, above the +0.01
clause threshold) at winner parity — a plausible-looking win, with a
plausible mechanism (`ko_losses` is a method-relevant chin proxy). Before
acting on it, the observation was replication-tested with fresh seeds 10-19,
same config, same features: method macro-F1 came back **0.3635** —
indistinguishable from the committed 0.3639. The "win" was seed noise, not
signal. **Method clause: not invoked; nothing ships there either.** (Winner
LL on the replication was 0.6522 vs 0.6536 for seeds 0-9 — that spread of
0.0014 between two 10-seed ensembles is a useful direct measurement of the
ensemble-level noise floor, and it further contextualizes every ±0.003
"improvement" in this document.)

## Conclusions

**Nothing ships.** Every code change from this session was reverted; the
committed models, features, and app behavior are unchanged, and this
document is the only committed artifact — the same outcome, honestly
reported, as the Elo v1.1 session.

What was learned, in order of confidence:

1. **The torch config has real headroom, just not enough of it.** A proper
   random search beats the hand-picked v1 config by ~0.005 log-loss at the
   2-seed level and ~0.003 at the 10-seed-ensemble level (0.6537 → 0.6509),
   with a consistent recipe (embedding dim 8, lr ≈ 2.5e-3, tiny weight
   decay). That is a genuine but sub-bar improvement — and the bar exists
   precisely because multi-shot validation reuse inflates such gains.
2. **Stacking is dead in this regime.** Fitted logistic blends lost to a
   simple average in both E1 and E5. Two strong, correlated predictors plus
   a weak one give a stack nothing to work with at n=1507.
3. **The six new point-in-time features don't pay for the winner head.**
   XGB got worse (+0.001-0.002 LL); torch got worse (0.6509 → 0.6536 with
   the same config). `height_reach_diff` earns high XGB feature importance
   while degrading log-loss — importance is not utility.
4. **The tempting method-head win was seed noise.** +0.019 macro-F1
   evaporated to +0.000 under fresh-seed replication. The replication also
   measured the 10-seed-ensemble noise floor directly: ~0.0014 log-loss
   between identical-config ensembles differing only in seeds. Effects of
   the size chased here (0.003-0.005) sit barely above that floor, which
   validates the deliberately high 0.004 bar.
5. **Ops:** torch runs stall when the venv lives under an iCloud-synced
   Desktop folder (libtorch page-ins block for minutes); a scratch venv on
   local disk with identical package versions reproduced v1's numbers
   exactly and ran the whole search in minutes. Worth institutionalizing
   for any future training session.

**Reverted:** `src/mma/history.py`, `src/mma/features.py`,
`src/mma/snapshots.py`, `tests/test_history.py`, `tests/test_features.py`
(E4 feature code and its unit tests — dropped with the code they test, per
the elo-v1.1 precedent). No model/feature artifacts were modified at any
point: all experimental training wrote to the session scratchpad only.
Committed test suite re-verified green at session end.

**For a future pass:** the best-E3 recipe (embedding 8, higher lr, minimal
weight decay) is the single most promising carry-forward; if a future
experiment independently clears its own pre-registered bar, adopting that
config family is where to start. Prospective (post-2024) evaluation remains
the only unspent adjudicator.
