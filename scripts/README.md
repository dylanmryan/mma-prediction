# scripts/

Thirty-eight files, and only twelve of them run on a schedule. This says which
is which, because "what actually runs?" should not require reading a workflow
file.

## The weekly pipeline

`.github/workflows/refresh-data.yml` runs these, in this order. Everything
after the rebuild is conditional on new fights having arrived.

| | |
|---|---|
| `refresh_data.py` | pull the maintained Kaggle mirror into `data/raw/` |
| `make_dataset.py` | raw CSVs → `data/processed/*.parquet`, with the regression guard and the standing daily-scrape merge |
| `build_ratings.py` | Elo and Glicko-2, one chronological pass |
| `build_features.py` | the model-ready table, from the committed block set |
| `train_xgb.py` · `train_torch.py` · `train_hazard.py` | retrain the deployed members on everything through the newest event |
| `predict_upcoming.py` | predict scheduled events **before** they happen, into `predictions/` |
| `grade_predictions.py` | grade past predictions; rewrite `predictions/track_record.json` |
| `revalidate_recipe.py --check-staleness` | the cheap weekly read: is the recipe's evidence older than the data it trains on? |
| `check_snapshot_coverage.py` · `check_display_calibration.py` | weekly warnings, not tests |

## Run by hand, occasionally

| | |
|---|---|
| `download_data.py` | first-time bootstrap of `data/raw/` |
| `build_external.py` · `build_rankings.py` | rebuild the `data/external/` tables from their upstream snapshots |
| `build_blend_config.py` · `derive_blend_temperature.py` | derive `models/blend.json` from the walk-forward report rather than letting the constants drift |
| `revalidate_recipe.py` | the full re-validation: re-run the harness and re-apply every recorded bar to the current table |
| `reconcile_sources.py` · `refresh_secondary.py` | imported by `make_dataset.py`; runnable alone to audit a source swap |
| `migrate_model_versions.py` | a one-time re-key of prediction records, kept for the record |

## The instruments

Reusable measurement, not one-off answers.

| | |
|---|---|
| `run_walkforward.py` | score **any** candidate on the expanding-window harness; every report in `models/walkforward/` came from here |
| `noise_floor.py` | measure a scorer's seed noise, so a bar can be `max(0.003, 2σ)` rather than a guess |
| `residual_probe.py` | *"does this column group add anything on top of what we deploy?"* — pinned offset, shuffled null, reported detection floor |
| `blend_check.py` | compose two finished reports post hoc (a diagnostic; the real blend is a candidate) |
| `config_search.py` | the SP2.1 architecture space, **frozen** — re-sampling it would turn its two arms into two unrelated searches |

## Recorded experiments

Each produced a decision artifact in `models/walkforward/` and a section in
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md). They are kept so every claim
in the README can be re-derived rather than taken on trust.

| | what it decided |
|---|---|
| `block_decision.py` | which of the eight feature blocks cleared their bar (one did) |
| `refit_decision.py` | whether to refit through the latest event, or hold out a split |
| `sp2_1_decision.py` | were the rejected blocks capacity-limited? (no — and its winner failed fresh seeds) |
| `sp2_2_decision.py` | the two-family blend — the project's one real win |
| `sp3_decision.py` | the Monte Carlo simulator, and the joint it produces |
| `sp5_decision.py` | the scorecard margin as a training label (fails; mechanism backwards) |
| `sp6_decision.py` | a third blend member (fails; 2 → 3 is worth nothing) |
| `calibration_audit.py` | is the model's confidence honest, and is fixing it worth anything? (yes; no) |
| `external_decay_decision.py` | whether the decaying `external` block still earns its place |
| `build_odds_benchmark.py` · `market_edge_analysis.py` | the market comparison, and ten pre-registered looks for an edge |
| `final_test_eval.py` | the one-time held-out 2024+ evaluation; refuses to re-run |
| `roll_window.py` | **retired** — points at `revalidate_recipe.py`, which replaced it |

Odds appear in exactly two of these and never as a model feature.
