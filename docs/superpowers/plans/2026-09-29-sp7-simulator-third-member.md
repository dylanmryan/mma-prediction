# SP7 — the simulator's own winner opinion, as a third blend member

**Status:** pre-registered 2026-09-29. Nothing below has been measured.

## Why this, and the prior — which the bar audit just made worse

SP3 built a Monte Carlo simulator to answer *how* and *when* a fight ends. It
also produces, unavoidably, an opinion about *who wins* — the simulated fights
have winners. SP3 measured that opinion once, found it slightly worse than the
blend's, imposed the blend's marginal over it, and discarded it.

It is better than either member SP6 went looking for:

| | standalone winner LL | vs the blend |
|---|---|---|
| deployed blend B1 | 0.6437 | — |
| **calibrated simulator** (`hazard_e2_cal.json`) | **0.6459** | +0.0022 |
| uncalibrated simulator (`hazard_e2.json`) | 0.6479 | +0.0042 |
| SP6's logistic | 0.6507 | +0.0070 |
| SP6's Bradley–Terry | 0.7069 | +0.0632 |

And it reads a genuinely different view of the data: per-round hazard classes
over `round.csv`, not the 87-column differential row. SP6's two candidates
failed at opposite ends — logistic was strong but 0.9110-correlated, BT was
0.1846-correlated but far too weak. This candidate is the only one available
that is strong *and* plausibly decorrelated.

**The prior is nonetheless poor, and it got worse this week.**
`models/bar_audit.json` measured the harness's detection floor at **0.0044**.
SP6's 1→2 member gain — the whole blend, the project's one shipped win — was
0.0028, and it does not itself clear that floor. A 2→3 gain that exceeds
0.0044 would have to be larger than the gain from introducing the second
family at all. That is possible (a less correlated member reduces more
variance) but it is not the way to bet.

This is worth running anyway for two reasons, both of which survive a
negative: the correlation number is informative about the simulator whether or
not the arm ships, and the hazard model is **already fitted inside the
deployed hybrid**, so the member costs no additional training.

## 1. The mechanism, and the screen that runs before the harness does

SP6's gate, reused: a fixed-weight average gains only when its members
disagree, and that is measurable without the outcome. The reference is the
incumbent pair's own correlation, **0.8518**.

SP6 also showed one clause is not enough — BT cleared the correlation gate
easily and still added nothing, because it was too weak to be worth averaging
in. So the screen has **two clauses, both required**:

| clause | threshold | status |
|---|---|---|
| standalone winner LL within 0.005 of the blend's | ≤ 0.6487 | **already met** (0.6479 uncalibrated, from `hazard_e2.json`) |
| prediction correlation below the incumbent pair's | < 0.8518 | **unmeasured — this is the screen** |

**The screen decides whether the harness arm runs at all.** If correlation is
≥ 0.8518 the arm is abandoned, the number is recorded as the finding, and no
walk-forward run is spent.

**Prediction recorded before measuring:** correlation lands in **0.60–0.80**.
The simulator's hazard members are fitted on features derived from the same
table, so it will not be BT-like; but the hazard likelihood, the per-round
censoring and the Monte Carlo composition are a different enough head that it
should sit clearly below the torch/xgb pair. If it comes back above 0.85, the
honest reading is that the simulator's winner is mostly the feature table
speaking again and SP3 lost nothing by discarding it.

## 2. The member, and why it enters uncalibrated

`HazardCandidate(calibrate=False)`, its winner marginal only.

**Uncalibrated is the pre-registered form**, even though the calibrated
variant scores better standalone, because that is how the incumbent members
enter: `BlendCandidate` averages raw xgb and torch streams and fits **one**
temperature after averaging. Feeding it a pre-calibrated member would
double-calibrate that member alone and make the arm a different experiment
from SP2.2's.

The calibrated variant is a **diagnostic, reported, and cannot ship**. If the
primary fails and the diagnostic looks better, that is a new experiment
needing its own pre-registration — not a post-hoc swap.

## 3. The arm — fixed now

| arm | winner head | weights |
|---|---|---|
| **S1** | mean(xgb, torch, simulator) | 1/3 each |

**One shipping arm is the whole search**, deliberately. At a 0.0044 bar, a
second arm would cost `se·√(2 ln 2)` = 0.0026 of additional multiplicity —
more than half the bar again — so arms are expensive now in a way they were
not when the bar was 0.003. Everything else below is a diagnostic that cannot
ship: the simulator alone (calibrated and not), the full pairwise correlation
matrix, and a torch+simulator pair with xgb dropped.

Method and round heads are untouched: `weight` continues to govern them, the
simulator continues to supply the conditional structure, and only the winner
marginal changes.

## 4. The bar

Read from `models/bar_audit.json`, not retyped: the `harness_detection_floor`
measured on the reference pair, **0.00438**, with `n_arms = 1` so the
best-of-N inflation is zero.

This supersedes `max(0.003, 2σ_seed)` for this experiment and the reason is
recorded in `docs/EXPERIMENTS.md`: σ_seed answers whether a refit lands in the
same place, not whether a delta survives the next four thousand fights, and
the second error is twenty times the first.

**Fresh-seed confirmation is still required to ship** — seeds 5–9 for every
seeded member, against a fresh-seed paired incumbent. SP2.1's best-of-25
winner cleared at −0.0039 and re-scored at −0.0024; that rule does not relax
because the search is small.

**Joint clause, inherited from SP3:** the joint outcome log-loss must not
regress. The winner marginal feeds the joint through
`impose_winner_marginal`, so a winner gain that degrades the joint is not a
gain. Tolerance: no worse than the incumbent hybrid's.

## 5. The primary endpoint

Pooled out-of-fold **winner** log-loss, walk-forward, fold years 2018–2025,
via `scripts/run_walkforward.py`, against the deployed hybrid scored on the
same folds and the same table. Accuracy is reported and never gated: across
4,856 fights its 95% CI is ±1.4 points and the bar is worth about a third of
one point.

## 6. Guards

- `extra=()` must remain **byte-identical** to SP2.2's two-member blend. The
  existing test stands; a new one covers the simulator branch.
- The hazard fit is **reused, not duplicated**. The hybrid already fits it;
  fitting it a second time for the winner stream would double the run cost and
  risk the two copies diverging.
- Serving parity: if S1 ships, `mma.serving` must build the same winner head
  at inference time. A member that exists only in the harness is the failure
  `feature_blocks` was restructured to make impossible.
- The temperature is fitted on inner-validation rows only, never on an
  evaluation row — `BlendCandidate` already raises if they overlap.

## 7. Decision rule

1. Run the screen. Correlation ≥ 0.8518 → **stop**, record, no harness run.
2. Otherwise run S1 at seeds 0–4. Winner Δ ≤ −0.00438 **and** joint not worse
   → proceed; else **stop and record**.
3. Re-run at seeds 5–9 against a fresh-seed paired incumbent. Both clauses
   again → ship. Either fails → **stop and record**.

No clause is a human call. Negative results are deliverables and get their
section in `docs/EXPERIMENTS.md` either way.

## 8. What this does not do

It does not revisit blend weights (fitted stacking lost to a plain average in
model-v2, and fitting on these folds is the selection failure this project has
been burned by). It does not touch the method or round heads. It does not add
features. And it does not reopen any block the bar audit showed to be below
the harness's floor — those are unanswerable on this dataset, not merely
unproven.
