# Market edge analysis — where (if anywhere) does this model beat the book?

**Status:** pre-registration written 2026-09-09, before any slice or market was
measured. Completion notes appended after.

**Goal:** two questions, both asked out of fold.

- **Part A.** The pooled moneyline answer is already known and negative: the
  market beats the deployed hybrid's winner marginal by **+0.036** log-loss
  (`models/market_benchmark_oof.json`). Is there any *subset* of fights where
  that flips, and if so is the flip real or is it the best of many looks?
- **Part B.** The **method-prop** market (KO/TKO, submission, decision for each
  corner) has never been compared against. This is where the simulator should
  have its comparative advantage — books price method props crudely, and since
  SP3 we emit a coherent joint distribution rather than three independent heads.

---

## Why this document exists before the measurement

Testing many slices manufactures a winner. This repo has already come within one
commit of publishing a "+17.8% ROI" that was an artefact of scoring an in-sample
model against an out-of-sample market. The slice list below is therefore **fixed
now**. Anything added later must be recorded in the completion notes as an
addition, with the reason, and counted in the multiple-comparisons arithmetic.

---

## Non-negotiable guards

1. **Out of fold only.** Every model probability comes from the walk-forward
   dump — fold year *Y*'s predictions come from a model fitted on fights before
   *Y−1* and early-stopped on *Y−1*. The deployed model trains through the
   latest event, so its own predictions on these fights are in-sample and are
   not used for any headline. This is the same discipline
   `scripts/build_odds_benchmark.py --mode oof` already enforces, and the same
   dump.
2. **Pre-registered slice list** (below), fixed before measurement.
3. **Every slice reported, including the losers**, with n, and with the
   multiple-comparisons arithmetic stated.
4. **Log-loss is not profit.** Any slice or market that looks better also gets a
   realistic ROI at several edge thresholds, settled against the *actual offered
   odds with the vig in them*, never against the devigged probability. Bet
   counts reported. A tiny-n positive ROI is not a finding.
5. **Any edge is retrospective.** These fights have been reused across five
   experiments (SP1, SP2, SP2.1, SP2.2, SP3). Nothing here is a prospective
   result and nothing here is a betting strategy.

---

## Pre-registered slice list — Part A (moneyline)

Eight slices. Four are the harness's existing ones (`mma.walkforward.slice_masks`),
so their definitions are inherited rather than invented here; four are new and
their exact rules are stated. All are evaluated on the **intersection of the
walk-forward out-of-fold rows with the odds-aligned fights**, in
`features.parquet`'s corner convention.

| # | slice | exact definition | source |
|---|---|---|---|
| 1 | `debut` | `debut_a or debut_b` | harness |
| 2 | `womens` | `weight_class` starts with `"Women"` | harness |
| 3 | `five_round` | `scheduled_rounds.fillna(3) >= 5` | harness |
| 4 | `external_missing` | `external_missing == True` | harness |
| 5 | `youth_gap_5` | both ages known **and** `abs(age_a − age_b) >= 5` years | new |
| 6 | `home_advantage` | `home_country_unknown == False` **and** `home_country_a != home_country_b` — exactly one corner fighting in its own country, the other travelling | new |
| 7 | `title_fight` | `title_fight == True` | new |
| 8 | `short_notice` | `notice_unknown == False` **and** (`short_notice_30_a or short_notice_30_b`) | new |

**On slice 5 (`youth_gap_5`).** The 2026 economics literature flags youth as
residually mispriced. The rule chosen is the *age-gap* form rather than an
absolute age cut, because a residual mispricing of youth is a statement about
the market's handling of the age *differential* between two fighters, and the
age-gap form is the one where it has room to show. Threshold 5 years, stated
now. `age_a`/`age_b` are missing on ~2% of rows; those rows are out of the
slice, not imputed.

**On slice 6 (`home_advantage`).** `home_country_a`/`home_country_b` are already
built (`mma.context`); `home_country_unknown` is the fight-level flag saying the
pair cannot be trusted, so it gates the slice. Requiring the two corners to
*differ* is what makes it a travel slice rather than a "someone was home" slice.

A pooled `all` row is reported alongside as the reference. It is not one of the
eight looks — it is the number already published.

## Pre-registered markets — Part B (method props)

Two headline markets, plus a per-year breakdown of each (descriptive, not
separate looks):

| market | outcome space | n classes |
|---|---|---|
| `six_way` | (winner corner) × (KO/TKO, submission, decision) | 6 |
| `method_three_way` | method only, ignoring who wins | 3 |

`method_three_way` is reported because it isolates the simulator's *method*
skill from its winner skill: the six-way number mixes the two, and the winner
half is already known to lose to the market.

### Devigging a six-way market

`mma.odds.devig_pair` is two-way only. A multi-way devig is added:
**proportional (multiplicative) normalisation** — divide each raw implied
probability by the sum of all of them. This is the standard method, it is what
`devig_pair` already does for two outcomes, and the new function must reduce to
`devig_pair` exactly on a two-way market (unit-tested). No power/Shin devig: the
proportional method is the one already in this repo and switching families
mid-analysis would be an unregistered degree of freedom.

### Pre-registered odds sanity filter

A six-way book cannot *under*round. Fights whose six raw implied probabilities
sum to less than **1.00** or more than **1.60** are data errors and are dropped
before scoring. Both bounds are stated now; the excluded count is reported, and
the unfiltered numbers are reported as a sensitivity check so the filter cannot
hide anything.

### Getting the model's six-way distribution

Collapse the simulator's out-of-fold joint cells over rounds:
P(corner, method) = Σ_round cells[corner, method, round], plus the corner's
decision cell. The collapse is **verified, not assumed** — the six probabilities
must sum to 1 for every fight, and must reproduce the winner and method
marginals `mma.joint.marginals_from_cells` already reports.

The existing dump `models/walkforward/preds/hybrid_e2.json` carries only
`p_winner`, so it cannot answer Part B. `scripts/run_walkforward.py`'s
`prediction_dump` is extended to also emit the joint cells and the realised
method label, and the hybrid walk-forward is re-run to produce them. The re-run
must reproduce the committed `models/walkforward/hybrid_e2.json` report
**exactly** (the candidate is deterministic: fixed member seeds and a fixed
`sim_seed`), which is what licenses treating the new dump as the same
out-of-fold predictions the existing benchmark already uses.

## Pre-registered statistics

- **Metrics.** Log-loss, accuracy, Brier — per slice and per market, model and
  market side by side, plus the delta (model − market; negative means the model
  won).
- **Significance.** Paired two-sided t-test on the per-fight log-loss
  differences (model − market) within a slice. Reported as a *nominal* p-value
  and then adjusted.
- **Multiple comparisons.** 8 slices (Part A) + 2 markets (Part B) = **10
  pre-registered looks**. Bonferroni threshold for α = 0.05 is
  **p < 0.005**. The Šidák equivalent is reported too. The best-looking slice
  is never presented without this arithmetic beside it.
- **ROI.** Thresholds 0.00 / 0.05 / 0.10 on the model-minus-market edge,
  settled at the **actual offered decimal odds** (vig included), flat 1-unit
  stakes, bet count reported at every threshold. Part A bets both corners
  (the existing `roi_sweep`'s two directions); Part B bets any of the six prop
  outcomes whose model probability exceeds the offered price's raw implied
  probability by the threshold.

## What is produced

- `models/market_edge_analysis.json`, regenerable via
  `scripts/market_edge_analysis.py`.
- Pure helpers (`mma.odds.devig_multiway`, the collapse, the slice masks, the
  prop ROI) unit-tested.
- Completion notes below.
- The README's market section is updated **only if something survives the
  scepticism**, and if it does, with the retrospective caveat attached rather
  than as a headline.

---

## Completion notes

*(appended after measurement)*
