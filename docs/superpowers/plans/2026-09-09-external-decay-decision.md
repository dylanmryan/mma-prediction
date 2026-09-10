# External-snapshot decay: does the `external` block still earn its place?

**Status: pre-registered 2026-09-09. Committed BEFORE any candidate was run.**
Nothing below was written with a candidate number in hand. The numbers that
appear here are prior, already-committed measurements (SP2 Task 11's notes and
`data/external/README.md`); every number this decision turns on is produced
after this file is committed and lands in
`models/walkforward/external_decay_decision.json`.

---

## 1. The question

The `external` feature block shipped in SP2 at **−0.0034** pooled winner
log-loss against `torch_v1`, and SP2's own completion notes record that the
gain was **entirely historical**: fold deltas of −0.0071 … −0.0005 over
2018–2023 and **+0.0015 (2024)** and **−0.0020 (2025)** on the two most recent
folds, row-weighted **−0.0047** against **−0.00064**.

Its source is a static snapshot. `ehan03/jds-mma-data`, commit `ec77f537`, UFC
event coverage ending **2024-12-14**, last upstream commit December 2025. A
fighter who debuts after that date is unmatched by construction, so coverage
does not merely stop improving — it *decays*, as the share of rows containing
at least one post-snapshot fighter grows with every event.

Measured decay, from the committed table (`external_missing` by fight year):

| year | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| `external_missing` | 0.071 | 0.119 | 0.122 | 0.138 | **0.359** | **0.534** |

The pre-UFC differentials are populated on 79.9% of all rows but only **57.8%**
of 2025-and-later rows.

**The question this document pre-registers a decision procedure for:** does the
`external` block — and the snapshot-dependent part of `notice` and `context` —
still earn its place in the deployed model, judged on the folds that represent
the future rather than on the pooled average over folds whose coverage we will
never have again?

## 2. What "snapshot-dependent" means, named column by column

Three registered blocks read the same static snapshot. Two of their columns
groups are *already* held out of every model matrix (`mma.tensors.DROPPED`,
`mma.models.xgb.MODEL_EXCLUDED`) as leak guards, and stay there either way —
they are what makes the `external_missing` slice reportable.

**Snapshot-dependent AND currently in the model matrix — the columns this
decision is about (15):**

| group | columns | count |
|---|---|---|
| `EXTERNAL` | `pre_ufc_wins_diff`, `pre_ufc_losses_diff`, `pre_ufc_finish_rate_diff`, `pre_ufc_finish_loss_rate_diff`, `pre_ufc_avg_opp_wins_diff`, `days_since_pro_debut_diff` | 6 |
| `NOTICE` | `notice_shortfall_days_diff`, `missed_weight_over_lbs_diff`, `short_notice_7_a`, `short_notice_7_b`, `short_notice_30_a`, `short_notice_30_b`, `missed_weight_a`, `missed_weight_b` | 8 |
| `CONTEXT` | `home_country_unknown` | 1 |

`CONTEXT` is snapshot-dependent only through nationality: `home_country_*` is
True when a fighter's nationality equals the event country, and nationality
exists only for fighters the snapshot mapped. `bonus_rate_diff` and the three
`referee_*` columns come from our own processed tables and are **not** part of
this decision.

**Snapshot-dependent and already excluded from every model matrix (unchanged by
this decision):** `external_missing`, `same_country`, `notice_unknown`,
`home_country_a`, `home_country_b`.

## 3. The candidates

All candidates are the **deployed feature table**, unchanged:
`base,external,trajectory,notice,context,opponent_adjusted`, 11,238 × 87, with
the block registry untouched. Removal is done with
`scripts/run_walkforward.py --drop-columns` — the mechanism already used for
the five leak-guard flags — and **not** by unregistering blocks, so the table,
`mma.walkforward.slice_masks` and the `external_missing` slice are identical
across every candidate and the comparison is paired and clean.

| id | `--drop-columns` | meaning |
|---|---|---|
| **K** | the 5 leak-guard flags | the deployed model as-is (the committed `models/walkforward/hybrid_e2.json`) |
| **R_ext** | 5 flags + `EXTERNAL` | the primary question |
| **R_notice** | 5 flags + `NOTICE` | the secondary read on `notice` |
| **R_ctx** | 5 flags + `CONTEXT` | the secondary read on `context` |
| **R_all** | 5 flags + all 15 | the consolidated removal |

K is the committed `hybrid_e2.json` re-used rather than regenerated: it is the
deployed report, run on this exact table with this exact recipe, and the
project's locked rules forbid regenerating an incumbent's numbers. Its
`config.drop_columns` is asserted by the decision script to be exactly the five
flags, so a mismatch is an error rather than a silent unpaired comparison.

Candidate `--candidate hybrid --seeds 0,1,2,3,4`, matching K.

## 4. The metric

The deployed scorer is the **hybrid** (`mma.inference.SimulatorPredictor` over
the blend's winner marginal), so its primary metric is the one SP3 shipped it
on:

- **Primary: pooled joint-outcome log-loss** (`pooled.joint_log_loss`). Lower
  is better; deltas are `candidate − K`, so **negative = removal is better**.
- **Secondary: pooled winner log-loss** (`pooled.winner_log_loss`), which must
  not regress by more than **σ_seed = 0.000346**
  (`models/walkforward/noise_floor.json`).

Both are reported **pooled over all eight folds** and **restricted to the two
most recent folds (2024 and 2025)**, the latter row-weighted:

```
recent(m) = (n_2024 * m_2024 + n_2025 * m_2025) / (n_2024 + n_2025)
```

## 5. The rule, stated mechanically

Let, for a removal candidate R against K:

- `d_pooled_joint  = R.pooled.joint_log_loss  − K.pooled.joint_log_loss`
- `d_recent_joint  = recent(R.joint_log_loss) − recent(K.joint_log_loss)`
- `d_pooled_winner = R.pooled.winner_log_loss − K.pooled.winner_log_loss`

The **contribution of keeping** the columns on the recent folds is stated in
the direction English states it — how much worse the removed model is —

```
keep_gain_recent = recent(R.joint_log_loss) − recent(K.joint_log_loss) = d_recent_joint
```

so it is **positive when keeping them helps** the recent folds. Note that this
is the same number as `d_recent_joint` and therefore reads in the *opposite*
direction to the other two deltas, which are negative-is-better: those are
about the removal, this one is about keeping.

> **Sign correction, made 2026-09-09 before any candidate report existed.**
> As first committed this line read `keep_recent = −d_recent_joint`, which is
> the wrong sign: negating `removed − kept` makes "keeping helps" come out
> negative and would have inverted clause 2. The error was found while writing
> `tests/test_decay.py`, with all four walk-forward runs still in flight and
> no candidate number read — the correcting commit precedes the first report
> in the log. Nothing else in the rule changed. Both directions are now pinned
> by `tests/test_decay.py::test_a_candidate_that_is_worse_recently_means_keeping_pays_and_keeps`
> and `::test_a_candidate_that_is_better_recently_removes`.

> **REMOVE the columns iff all three hold:**
>
> 1. `d_pooled_joint < 0.01` — removing them does not worsen pooled joint
>    log-loss by more than the 0.01 bar; **and**
> 2. `keep_gain_recent < 0.001` — the recent-fold (2024–2025) contribution of
>    keeping them is not better than the bar's tenth; **and**
> 3. `d_pooled_winner <= 0.000346` — the winner marginal does not regress by
>    more than σ_seed.
>
> Otherwise **KEEP**.

Clause 3 is the secondary guard from §4, stated as a veto so it can be applied
without judgement. Clause 1's `<` and clause 2's `<` are strict; clause 3's
`<=` is inclusive because it is a tolerance rather than a bar. All three are
evaluated on values rounded to 6 dp, as `mma.walkforward.bar_check` does, so a
candidate sitting exactly on a threshold does not cross it on float noise.

**Why the burden is asymmetric.** SP1's rule for an *addition* is "improve
pooled log-loss by more than the bar". This is not an addition; it is a
**simplification**, and the two are not symmetric:

- A block that pays only on stale folds is a **liability** once its coverage
  decays. Its pooled number averages folds whose coverage (7–12% missing) we
  will never have again with folds whose coverage (36–53% missing) is what the
  deployed model actually faces. Requiring the *simpler* model to beat that
  average would be requiring it to beat a measurement of a world that no longer
  exists.
- The simpler model has **no decay to manage**: no snapshot to refresh, no
  coverage threshold to watch, no growing gap between the rows the model was
  fitted on and the rows it serves. That is a real, permanent reduction in
  maintenance risk that the pooled log-loss does not price.
- So the pooled clause is a **do-no-harm** clause (the removal must not cost
  more than the same 0.01 bar an addition would have to clear), while the clause
  that actually decides is the **recent-fold** one, at a tenth of the bar. The
  columns keep their place only by earning it on the folds that represent the
  future, and by a margin an order of magnitude below what a new block would
  need — because they are already here and removal has its own cost.

**Ambiguity.** If the three clauses do not agree, or if a candidate's
`d_pooled_joint` or `keep_gain_recent` lands within 1e-4 of its threshold, the
result is recorded as **AMBIGUOUS** and reported rather than resolved. No
tiebreak is invented after the fact.

**Scope of the decision.** The rule is applied to each of `R_ext`,
`R_notice` and `R_ctx` independently. The change that is *deployed* is the
union of the groups the rule says REMOVE, and that union must itself satisfy
the rule as candidate `R_all` before anything is retrained. If the union is
`EXTERNAL ∪ NOTICE ∪ CONTEXT` then `R_all` is that check; if the union is a
single group, that group's own candidate is the check and `R_all` is reported
as context.

## 6. Fresh-seed confirmation (mandatory)

As every prior decision in this project has required, and as SP2.1's false
positive earned: **no change reaches the deployed model without a fresh-seed
confirmation.** The winning removal candidate is re-run at seeds **5,6,7,8,9**
against a **fresh-seed paired incumbent** — K re-run at the same seeds, since
`hybrid_e2_seeds5.json` exists only if it was run with K's exact
`drop_columns`; the decision script asserts the pairing rather than assuming
it. The rule of §5 is re-applied to the fresh-seed pair, and **both** seed sets
must say REMOVE. If they disagree, the result is AMBIGUOUS and nothing is
deployed.

## 7. What is produced

- `models/walkforward/external_decay_decision.json`, written by
  `scripts/external_decay_decision.py`, carrying: the coverage table, every
  candidate's pooled and recent-fold joint and winner numbers, per-fold deltas,
  the `external_missing` slice for each, the three clauses evaluated per group,
  the verdict, and the fresh-seed confirmation.
- The arithmetic of §5 lives in a **pure helper**
  (`mma.decay.removal_verdict`) with its own unit tests, so the rule is
  executable and testable rather than prose applied by hand.

## 8. Making the decay visible, whatever the verdict

Independent of the decision, coverage decay must fail loudly rather than rot:

- A coverage check in the pattern of `scripts/check_display_calibration.py`
  measures `external_missing` over the **trailing 12 months of the feature
  table** and warns on stderr past a threshold.
- **Threshold: 0.40.** Justification is recorded with the check itself and in
  `data/external/README.md`: the 2025 fold, on which the block's recent-fold
  contribution was measured, sits at 0.359 missing. A trailing-12-month rate
  above 0.40 means the deployed model is serving a majority-uncovered
  population *worse* than the worst fold any of these numbers were measured on,
  so the measurement this decision rests on no longer describes the rows being
  served, and the decision must be re-taken. It is a decision-invalidation
  threshold, not a data-quality one.
- Wired into the weekly Action next to the existing display-calibration check,
  report-only (`continue-on-error`), and unit-tested.

## 9. If the verdict is KEEP

Leave the deployed model alone, and record what would have to be true to
revisit: the coverage threshold of §8 firing is exactly that trigger, and the
check is what makes it arrive as a warning rather than as a silent drift.
