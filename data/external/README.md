# `data/external/` — derived open-snapshot aggregates

| | |
|---|---|
| Source | <https://github.com/ehan03/jds-mma-data> |
| Licence | MIT (Copyright (c) 2025 Eugene Han) |
| Snapshot commit | `ec77f53737829c51fbf15ba140b46d36dee65d22` |
| Coverage end (source UFC events) | 2024-12-14 |
| Licence notice | vendored verbatim as [`LICENSE-jds-mma-data`](LICENSE-jds-mma-data) |
| Rows in `fighter_external.parquet` | 2553 as committed; **2551 usable** |
| Match rate vs `data/processed/fighters.parquet` | 2553 / 4581 (0.557) as committed; **2551 / 4581 (0.557) usable** |

The two rows differ by the 2 fighter(s) whose recorded pro debut
post-dates their first UFC bout. They are kept in the committed parquet so the
source disagreement stays visible, and dropped by
`mma.external.drop_source_errors` before anything reads it -- so the *usable*
figure is the one the feature table actually sees, and the one to quote.

**Only derived aggregates are committed here, never the raw snapshot** (~130 MB);
regenerate with `python scripts/build_external.py`, which clones the source into
a scratch directory.

The snapshot is static: its last commit is December 2025 and its UFC event
coverage ends 2024-12-14, so fighters who debuted after that are
unmatched by construction. That is what the row-level `external_missing` flag
and the walk-forward slice of the same name exist to measure.

Per-column non-null counts over the 2553 committed rows:

| column | non-null | share |
|---|---|---|
| `pre_ufc_wins` | 2553 | 1.000 |
| `pre_ufc_losses` | 2553 | 1.000 |
| `pre_ufc_finish_rate` | 2378 | 0.931 |
| `pre_ufc_finish_loss_rate` | 1580 | 0.619 |
| `pre_ufc_avg_opp_wins` | 2390 | 0.936 |
| `pro_debut_date` | 2553 | 1.000 |
| `first_ufc_date` | 2553 | 1.000 |
| `nationality` | 2550 | 0.999 |
| `gym_id` | 2024 | 0.793 |

## Membership of this table is not itself a feature

Being in the source's cross-source fighter mapping requires the fighter to be
findable on every one of its sites, which in practice means having had a UFC
career. Among fighters who debuted 2013-2022 -- well inside the coverage
window -- 24% of the one-and-done fighters are in the mapping against 100% of
those with 11 or more bouts. Membership is therefore a look-ahead variable:
whether a fighter is here depends on fights that had not happened yet.

Consequently the `external` feature block carries missingness only at fight
level, as the symmetric either-corner OR. A per-corner
`external_missing_a`/`_b` pair was measured first and leaks badly: the unmapped
corner loses 76% of the time overall and 90% of the time in the debut slice,
which bought a spurious -0.0143 pooled log-loss on the 2018-2023 folds and then
cost +0.0487 on 2025, where missingness means "debuted after the snapshot"
instead. See the SP2 plan's Task 11 notes.

The shipped block goes further: both fight-level flags stay in the feature
table (so the `external_missing` walk-forward slice is reportable) and are
excluded from every model matrix, because modelling them makes the block decay
as the snapshot ages. What makes the selection effect *unreachable* rather
than merely unmodelled is that on the rows where exactly one corner is
unmapped, every one of the block's columns takes the same value -- one
distinct value tuple over all 1,445 such rows, asserted by
`tests/test_external.py::test_a_half_matched_fight_carries_exactly_one_value_tuple`.

## Point-in-time

Every `pre_ufc_*` column is cut at `history["date"] < first_ufc_date`, so it
summarises a window that closed before the fighter's UFC career began and is
therefore constant across all of their UFC fights. That constancy -- not the
truncation-invariance test, which cannot see a static artifact at all -- is
the checkable form of the claim, and
`tests/test_external.py::test_pre_ufc_values_are_constant_across_a_fighters_ufc_career`
asserts it on the real table. `scripts/build_external.py` carries the full
argument, including the zero-wins convention for an unrecorded opponent;
`src/mma/external.py` carries the join (by fighter id only, never by name) and
the missingness rules.

# `fight_notice.parquet` — short notice and missed weight

One row per CORNER of every UFC bout the snapshot's Bet MMA tables cover:
11348 rows over 5674 fights, 0.496 of our
11441 fights (0.917 of the 2013-04-20 .. 2024-12-14 window the source spans).
517 of those corners were late replacements (notice
1-46 days) and 223 missed weight.

| | |
|---|---|
| Source tables | `Bet MMA/late_replacements.csv`, `Bet MMA/missed_weights.csv`, `Bet MMA/bouts.csv`, `bout_mapping.csv`, `Bet MMA/fighters.csv` |
| Licence | MIT, same snapshot and commit as above |
| Coverage | 2013-04-20 .. 2024-12-14 |

**Membership is the three-state boundary.** The two source lists name only the
fighters a thing happened to, so "no row" would otherwise be ambiguous between
"trained a full camp" and "nobody recorded it". Bet MMA's own bout list
resolves it: inside the bouts it covers, absence of a replacement row is an
observation; outside them, absence is ignorance. The derivation therefore emits
BOTH corners of a covered bout or neither, which also means the block carries
no per-corner missingness channel at all — the failure mode the `external`
section above describes. `src/mma/notice.py` carries the rule and
`tests/test_notice.py` checks it against the committed table.

**Corner assignment never uses names.** `bout_mapping.csv` gives our
`fight_id`, both Bet MMA fighter ids resolve to ufcstats ids through
`fighter_mapping.csv` and `Bet MMA/fighters.csv`, and the derivation asserts
the resulting pair is exactly the pair our own fights table records for that
fight; a bout where it is not is dropped whole.

**Point-in-time.** A replacement is booked and a weigh-in happens before the
bout, so both facts are known on fight morning, and neither is accumulated
across fights.
