"""Provenance checks on the committed XGBoost metrics file (refit mode)."""
import json
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "models" / "xgb_metrics_val.json"
# The decision the DEPLOYED budget came from. It is per feature table AND per
# fit shape -- refit_decision.json is the SP1 46-column table's,
# refit_decision_v3.json the SP2 `base,external` table's single fit,
# refit_decision_b1.json the SP2.2 S1 table's five-seed ensemble -- so this
# tracks whichever set the train scripts default to (their REPORT is that
# set's B report).
REFIT_DECISION = ROOT / "models" / "walkforward" / "refit_decision_v3.json"

pytestmark = pytest.mark.skipif(
    not METRICS.exists(),
    reason="xgb models not trained (run scripts/train_xgb.py)",
)


def test_the_seed_ensemble_artifacts_exist():
    """Since SP2.2 each head is a five-seed ensemble and all fifteen boosters
    are half of what serves -- a missing one is half a member of the blend."""
    from scripts.train_xgb import SEEDS

    for head in ("winner", "method", "round"):
        for seed in SEEDS:
            assert (ROOT / "models" / f"xgb_{head}_seed{seed}.json").exists()
    # the pre-SP2.2 single-fit artifacts are gone, not merely unused
    for head in ("winner", "method", "round"):
        assert not (ROOT / "models" / f"xgb_{head}.json").exists()


def test_the_metrics_file_records_the_seed_ensemble():
    from scripts.train_xgb import SEEDS

    metrics = json.loads(METRICS.read_text())
    assert metrics["seeds"] == list(SEEDS)


def test_refit_metrics_provenance_matches_harness_and_decision():
    """The evidence a refit metrics file quotes must be the committed harness
    report, and its per-head budget must be the committed refit decision's."""
    metrics = json.loads(METRICS.read_text())
    if metrics.get("mode") != "refit_through":
        pytest.skip("split-mode metrics carry their own held-out numbers")
    report = json.loads((ROOT / metrics["harness_report"]).read_text())
    assert metrics["walkforward_pooled"] == report["pooled"]
    assert metrics["harness_features_max_date"] == report["config"]["features_max_date"]
    assert metrics["harness_fold_years"] == report["fold_years"]
    # after a weekly retrain on newer data this goes stale until the harness
    # is re-run, so it must not fail the suite (and thus the Action's commit/
    # predict steps) every week.
    if metrics["train_through"] > metrics["harness_features_max_date"]:
        warnings.warn(
            "harness evidence predates training data; re-run scripts/run_walkforward.py"
        )
    assert metrics["winner"]["log_loss"] == report["pooled"]["winner_log_loss"]
    assert metrics["winner"]["log_loss"] < 0.6827  # must at least beat pooled Elo

    decision = json.loads(REFIT_DECISION.read_text())["xgb"]["budget"]["fixed_rounds"]
    assert metrics["budget"] == decision
    assert metrics["winner"]["best_iteration"] == decision["winner"]
