# `data/external/` — derived open-snapshot aggregates

| | |
|---|---|
| Source | <https://github.com/ehan03/jds-mma-data> |
| Licence | MIT (Copyright (c) 2025 Eugene Han) |
| Snapshot commit | `ec77f53737829c51fbf15ba140b46d36dee65d22` |
| Coverage end (source UFC events) | 2024-12-14 |
| Rows in `fighter_external.parquet` | 2553 |
| Match rate vs `data/processed/fighters.parquet` | 2553 / 4581 (0.557) |

**Only derived aggregates are committed here, never the raw snapshot** (~130 MB);
regenerate with `python scripts/build_external.py`, which clones the source into
a scratch directory.

The snapshot is static: its last commit is December 2025 and its UFC event
coverage ends 2024-12-14, so fighters who debuted after that are
unmatched by construction. That is what the row-level `external_missing` flag
and the walk-forward slice of the same name exist to measure.

Per-column non-null counts over the 2553 matched fighters:

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

See `scripts/build_external.py` for the point-in-time argument behind every
column, and `src/mma/external.py` for the join (by fighter id only, never by
name) and the missingness rules.
