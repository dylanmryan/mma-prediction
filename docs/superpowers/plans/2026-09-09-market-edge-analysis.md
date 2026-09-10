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

**Verdict: nothing survives. The model loses all ten pre-registered looks.**
There is no slice where it beats the moneyline, and the method-prop market —
the one where the simulator was expected to have a comparative advantage — is a
**wider** loss than the moneyline, not a narrower one. The README was not
touched, because the condition for touching it (something real was found) was
not met.

Artifact: [`models/market_edge_analysis.json`](../../../models/market_edge_analysis.json),
regenerable with
`OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/market_edge_analysis.py`.

### Part A — the eight moneyline slices (out of fold, n = 3,310 matched fights)

Log-loss. **Positive delta = the market wins.** ROI is flat 1-unit stakes at
the 0.00 edge threshold, settled at the actual offered price.

| slice | n | model | market | delta | nominal p | ROI @0.00 (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| *(reference: all matched)* | 3,310 | 0.6463 | 0.6099 | **+0.0364** | ~0 | −6.0% [−10.6, −1.4] |
| `debut` | 384 | 0.6272 | 0.5464 | **+0.0808** | 1e−6 | −17.8% [−30.4, −5.1] |
| `womens` | 541 | 0.6519 | 0.6132 | **+0.0387** | 0.00065 | −1.0% [−13.4, +11.4] |
| `five_round` | 344 | 0.6452 | 0.6372 | **+0.0080** | 0.575 | +3.3% [−11.7, +18.4] |
| `external_missing` | 54 | 0.6272 | 0.6511 | **−0.0239** | 0.686 | +22.9% [−19.8, +65.7] |
| `youth_gap_5` | 1,179 | 0.6202 | 0.5866 | **+0.0336** | 4.5e−5 | −8.4% [−16.1, −0.7] |
| `home_advantage` | 1,453 | 0.6452 | 0.6125 | **+0.0327** | 7e−6 | −6.4% [−13.5, +0.7] |
| `title_fight` | 138 | 0.6173 | 0.6054 | **+0.0119** | 0.570 | +6.9% [−19.2, +32.9] |
| `short_notice` | 300 | 0.6315 | 0.5490 | **+0.0825** | 2e−6 | −26.1% [−39.4, −12.7] |

**Seven of eight go to the market**, five of them at p below the Bonferroni
threshold of 0.005. The only slice where the model is ahead on log-loss is
`external_missing`, by 0.024 on **54 fights**, at p = 0.69. Its ROI looks
spectacular (+22.9% at threshold 0.00, +47.9% at 0.10) and means nothing: 30–48
bets, standard errors of 22–31 percentage points, every interval straddling
zero by a wide margin. This is precisely the shape of result guard 4 was
written to catch, and it is reported here as a loser rather than as a lead.

Three slices show a positive ROI (`five_round` +3.3%, `title_fight` +6.9%,
`external_missing` +22.9%) while **losing** on log-loss. That is guard 4 in
both directions: profit and calibration are different questions, all three
intervals comfortably contain zero, and none of the three is a finding. The
`womens` slice makes the same point in reverse — the market beats it on
log-loss at p = 0.00065, yet its ROI is a statistically indistinguishable
−1.0%.

Note the direction of the two "literature" slices. `youth_gap_5` and
`home_advantage` were included because the 2026 economics literature flags them
as residually mispriced. If they are mispriced, **this model is not the thing
that exploits it** — the market beats it on both by roughly the pooled margin,
so whatever the model knows about age gaps and travel is already in the line.

### Part B — the method-prop market (out of fold, n = 2,890)

The market the simulator was supposed to be good at.

| market | n | model | market | delta | nominal p |
|---|---:|---:|---:|---:|---:|
| `six_way` (corner × method) | 2,890 | 1.6254 | 1.5317 | **+0.0937** | ~0 |
| `method_three_way` (method only) | 2,890 | 0.9959 | 0.9533 | **+0.0426** | ~0 |

Accuracy and Brier agree with log-loss in both: six-way accuracy 0.3263 vs the
book's 0.3737; method-only 0.5125 vs 0.5415. No metric dissents.

Per year, six-way delta (positive = market wins): 2018 **+0.126**, 2019
**+0.093**, 2020 **+0.098**, 2021 **+0.095**, 2022 **+0.096**, 2023 **+0.083**,
2024 **+0.055**. **Seven years out of seven.** There is a mild narrowing trend,
but no year comes close to flipping.

The method-only comparison is the informative one, and it is the one that
disposes of the hypothesis this analysis was built to test. The hope was that
the six-way loss would decompose into "loses the winner half (known), wins the
method half (new)". It does not. Stripping the winner axis out entirely still
leaves the book ahead by 0.043. **The simulator's coherent joint does not price
method better than a crude prop book does**, even when it is not being asked to
say who wins. Roughly half the six-way gap is the already-known winner deficit
and the other half is a genuine method deficit.

ROI, six-way, vs the actual offered price:

| threshold | bets | ROI | 95% CI | p |
|---|---:|---:|---|---:|
| 0.00 | 6,230 | −22.1% | [−28.4, −15.7] | ~0 |
| 0.05 | 2,370 | −11.9% | [−21.9, −1.9] | 0.019 |
| 0.10 | 842 | **+2.8%** | **[−14.4, +20.0]** | **0.75** |

**On the largest positive number in the analysis.** The +2.8% at threshold 0.10
is not an edge. Its interval spans 34 percentage points; it is bracketed by
−11.9% at the threshold below it; and it comes from a model that loses the same
market's log-loss by 0.094 with a t-statistic past 10 and loses it in all seven
years. A flat stake on outcomes priced around 6.5 decimal has a per-bet spread
near 2.5 units, so ~800 bets cannot resolve anything smaller than roughly ±17
points. Betting off the devigged price instead — the looser bar — gives −7.8%
at the same threshold. (An unregistered spot-check one notch further out, at
threshold 0.15, gives −8.6% on 271 bets; recorded here as the ad-hoc check it
is, and it is not in the artifact, whose thresholds are the pre-registered
three.) It is noise, and the error bars are in the artifact so that it stays
labelled as noise.

### Multiple comparisons

Ten pre-registered looks, so Bonferroni α = 0.005 and Šidák α = 0.00512, both
fixed before the first p-value existed. The arithmetic did not end up mattering:
**no look produced a nominal p below 0.05 in the model's favour**, so there was
no best-slice to discount. The correction was needed to protect against a
finding, and there was no finding. Had `external_missing` come in at, say,
p = 0.03, ten looks would have made that entirely unremarkable — a nominal 0.05
somewhere in ten independent looks is expected about 40% of the time under a
true null.

### Deviations from the pre-registration, recorded

1. **The re-run could not reproduce `hybrid_e2.json` *exactly*.** The plan
   required it. Between that report and this work, a data refresh moved
   `features.parquet` from 11,238 to 11,290 fights (max date 2026-08-08 →
   2026-09-05), and the 2025 fold is unbounded above, so it absorbed the 52 new
   rows (935 → 987) and the pooled numbers moved with them
   (winner log-loss 0.6437 → 0.6459). What *was* verified, and is stronger than
   the clause asked for: folds 2018–2024 reproduce metric for metric; the
   out-of-fold winner probabilities match the committed dump to **zero**
   difference on all 4,804 shared rows; and two independent re-runs produced
   byte-identical reports. All 52 new fights are 2026 fights and the prop lines
   stop on 2024-12-07, so none can reach Part B. Artifacts:
   `models/walkforward/hybrid_e2_cells.json` and its dump.
2. **ROI standard errors were added.** The plan pre-registered bet counts only.
   Point estimates alone would have made the +2.8% prop ROI look like a result;
   the interval is what makes it obviously not one. An addition that makes a
   number harder to over-read, not easier.
3. **Part A ROI is reported against both reference prices** (raw offered, and
   devigged), where the plan specified the vig-inclusive form for Part B and
   inherited the existing devigged sweep for Part A. Both are reported for both
   parts so the two halves are computed by one rule. No headline depends on the
   choice; the model loses on either.

Nothing was added to the slice list, and no slice definition was altered after
seeing a number.

### Alignment checks, because a Part B edge would have been suspicious

There was no apparent edge to explain away, but the checks were run anyway and
they are what make the negative result trustworthy rather than merely
convenient:

- **The Part A reference block reproduces
  `models/market_benchmark_oof.json`'s published headline exactly** — n = 3,310,
  model 0.6463 / 0.6211 / 0.2275, market 0.6099 / 0.6604 / 0.2111. This
  pipeline aligns the odds the way the already-validated benchmark does.
- **The famous-fight corner check passes**, all five, including the two
  Silva/Weidman fights whose odds rows are in opposite corner order.
- **The collapse is exact.** The six-way probabilities sum to 1 and reproduce
  `mma.joint.marginals_from_cells`'s winner and method marginals to 2.2e-16 and
  3.3e-16 — machine epsilon, i.e. it is the same reading of the same cells.
- **Hand-verified fights.** *Carlos Prates vs Neil Magny* (2024-11-09): the book
  put 66% on corner A by KO/TKO, and Prates won by KO/TKO. *Khama Worthy vs
  Devonte Smith* (2019-08-17): the book put 65% on corner **B** by KO/TKO, and
  corner **A** won by KO/TKO — Worthy's upset. The second is the load-bearing
  one: the favourite lost, so the alignment cannot be a "favourite == winner"
  artefact.
- **The overround filter changes nothing.** Dropping it (n 2,890 → 2,903) moves
  the six-way delta from +0.0937 to +0.0923. The result is not an artefact of
  the pre-registered data cleaning. 22 fights of 4,867 failed the filter; 18 of
  those were underrounds below 1.0, which no real book prices.

### What this is not

Every number here is **retrospective**. These fights have been reused across
five experiments (SP1, SP2, SP2.1, SP2.2, SP3). The slice list was
pre-registered but the *fights* were not fresh, and no result here has been
confirmed prospectively. That caveat would matter enormously if something had
been found; since nothing was, it cuts the other way — a negative result on
heavily-mined data is if anything the more credible direction, because
data reuse biases towards spurious *findings*, not spurious nulls.

Nothing here is a betting strategy: no bankroll management, no line shopping,
no closing-line timing, no transaction costs, no limits, and one bookmaker in
one region.

### What would change the answer

The prop result is the substantive one, and it is a clean negative for the
hypothesis that motivated SP3's joint. If it is to be revisited, the thing to
test is not another slice of the same fights — it is whether the *round*
distribution (which this prop market does not price, and which the collapse
throws away) carries skill the method marginal does not. That would need a
market that prices rounds, and it would need to be pre-registered before it is
measured.

**Test counts:** 1,135 passed / 1 skipped, up from 1,082 / 1 (53 new: 9 for
`devig_multiway`, 4 for the extended prediction dump, 40 for the analysis
helpers).
