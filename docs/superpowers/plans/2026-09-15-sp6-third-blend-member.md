# SP6 — a structurally different third member for the blend

**Status:** pre-registration written 2026-09-15, **before any member was built
or scored**. Completion notes appended after.

**Goal.** The blend is the only thing that has ever cleared a bar in this
project, and it worked because two learners were wrong in different places.
Both members read the same 87-column table and both are feature-discriminative.
This asks whether a member that is different in *kind* adds anything further.

---

## Why this, and the prior — which is not good

The blend is the project's one success, so repeating its mechanism is the best
remaining idea. But the honest expectation is small, and stating it first is
the point of writing this before measuring.

Measured on the current table, out of fold, n=4,856:

| | pooled winner log-loss |
|---|---|
| torch alone (5-seed) | 0.6487 |
| xgb alone (5-seed) | 0.6488 |
| **the deployed blend of the two** | **0.6459** |

**Going from one member to two bought 0.0028** — which is itself *just under*
the 0.003 bar on this table (it cleared at −0.0039 on the smaller table SP2.2
measured it on). Second members are the cheap ones; third members are
normally worth less. So the realistic expectation here is on the order of
**0.001, a third of the bar**, and the most likely outcome is another recorded
negative.

Two further reasons for pessimism, both worth stating now:

- **Elo and Glicko are already features.** `elo_diff`, `glicko_mu_diff`,
  `glicko_phi_diff` and `glicko_sigma_diff` are in the matrix both members
  read, so a latent-skill member may be largely re-deriving a column they
  already have.
- **A weaker member usually drags an average down.** Equal weighting only
  helps when the new member is *differently* wrong, not merely also wrong.

## 1. The mechanism, and why it is falsifiable BEFORE the outcome

A fixed-weight average can only gain when its members disagree. That is
measurable directly, and it does not need the outcome:

| | measured on the incumbent pair |
|---|---|
| corr(predictions), torch vs xgb | **0.8518** |
| they pick different winners on | **14.5%** of fights |
| and the blend of them gains | 0.0028 |

So **0.8518 is the reference**: a third member that correlates with the
existing two *more* than they correlate with each other has less to add than
the second member did, and any gain it shows is more likely luck than
diversity. This is pre-registered as the mechanism secondary and is reported
for every arm whether or not it ships.

This is a better gate than SP5's, because it is checkable independently of
whether the arm wins.

## 2. The two candidate members

**L — regularised logistic regression** on the same model matrix. The
canonical third member of a linear/trees/net trio: a completely different
bias-variance profile, and it cannot represent the interactions the other two
live on. Expected to be clearly *worse* standalone; the only question is
whether it is differently wrong. Cheap to fit (seconds).

**BT — hierarchical Bradley–Terry latent skill**, fitted by MAP with partial
pooling toward a per-division mean. Structurally the most different thing
available: a *latent-variable* estimator rather than a feature-discriminative
one. It sees only who fought whom and who won — **not the 87 columns** — which
is why its predictions should be the least correlated with the incumbents'.
It also shrinks thin records toward the division mean, which matters because
**20.5% of rows have a debutant in one corner**.

MAP rather than full MCMC: the blend consumes a probability, not a posterior,
and NUTS over ~4,600 latent skills refit per fold across 8 folds is hours of
compute for a number the blend discards. If BT clears anything, a Bayesian
version is a follow-up, not this experiment.

## 3. The arms — fixed now

**Shipping candidates (3).** Equal weights, never fitted. `BlendCandidate`'s
own docstring records that fitted stacking weights lost to a plain average in
the model-v2 session, and that fitting them on these folds is the selection
failure this project has been burned by; SP6 does not reopen it.

| arm | members | weights |
|---|---|---|
| **C1** | torch + xgb + L | 1/3 each |
| **C2** | torch + xgb + BT | 1/3 each |
| **C3** | torch + xgb + L + BT | 1/4 each |

**Diagnostics (reported, never gated):** L alone, BT alone, and the full
pairwise prediction-correlation matrix across all four members.

**Three candidates is the whole search.** No other member types, no other
weightings, no architecture search inside L or BT beyond the single
regularisation strength each selects on the fold's inner-validation year
(which is fitting, not searching — it never sees an evaluation row). Anything
added later is recorded in the completion notes as an addition, with its
reason, and counted in §4.

## 4. The bar and the multiplicity

The standard v3 §5 bar: **max(0.003, 2σ)** against the paired incumbent, which
for every arm here is **the deployed two-member blend at 0.6459**, scored on
the same folds and the same table.

σ_blend is 0.0001 — the most stable scorer this project has measured — so 2σ
is 0.0002 and the 0.003 floor governs.

Three candidates of selection optimism: σ√(2 ln 3) = **0.00015**. Negligible
next to the bar, which is the one comfortable thing about this experiment.

**Fresh-seed confirmation is still required to ship** (seeds 5–9 for every
seeded member, against the fresh-seed paired incumbent). SP2.1's best-of-25
winner cleared at −0.0039 and re-scored at −0.0024; the rule stands regardless
of how small the search is.

## 5. The primary endpoint

**Pooled out-of-fold winner log-loss**, walk-forward, fold years 2018–2025,
via `scripts/run_walkforward.py`. Not accuracy: across 4,856 fights the 95% CI
on accuracy is ±1.4 points, and the bar is worth about a third of one point,
so accuracy cannot resolve it. Accuracy is reported, never gated.

## 6. Guards

1. **Out of fold only.** Fold year *Y* comes from a model fitted on fights
   before *Y−1*, early-stopped on *Y−1*. Every member, including BT, is refit
   per fold: BT's latent skills are estimated from training fights ONLY, and a
   fighter with no training fight gets the division prior, never a skill fitted
   on the fight being predicted.
2. **BT sees no outcome from its own fold.** Its input is (fighter_a,
   fighter_b, winner, date) restricted to `fold.train`. A test asserts that
   erasing the evaluation rows from BT's input changes none of its evaluation
   predictions.
3. **Symmetry.** Every member must satisfy p(A beats B) = 1 − p(B beats A);
   BT does by construction and a test pins it.
4. **The temperature is fitted after averaging**, on the inner-validation year
   only, exactly as the two-member blend does — a three-way average of
   differently-calibrated streams is no more calibrated than a two-way one.
5. **Every arm reported, including the losers**, with per-fold numbers and its
   correlation matrix.

## 7. Decision rule

- **Ships** iff it beats the deployed blend by more than max(0.003, 2σ) **and**
  the fresh-seed re-score clears it too.
- **Does not ship** otherwise, and is recorded in `README.md` beside the other
  negatives with its numbers.
- **No arm ships on the strength of its correlation matrix, a slice, or a
  standalone member's quality.** The primary endpoint decides; the correlation
  is there to explain the result, not to produce one.

## 8. What this does not do

It does not touch the simulator, the hazard model, the feature table or
serving. It does not fit blend weights. It does not use odds. A passing arm
becomes a deployment only through the existing refit path.

---

## Completion notes

*(appended after the arms are scored — nothing above this line changes)*
