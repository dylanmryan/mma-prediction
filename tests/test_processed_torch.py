import json
import warnings
from pathlib import Path

import pytest

from mma.staleness import (
    load_revalidation,
    revalidation_cover,
    stale_harness_warning,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "torch"
# The decision the DEPLOYED budget came from. It is per feature table --
# refit_decision.json is the SP1 46-column table's, refit_decision_v3.json the
# SP2 `base,external` table's, refit_decision_b1.json the SP2.2 S1 table's --
# so this tracks whichever set the train scripts default to (their REPORT is
# that set's B report).
REFIT_DECISION = ROOT / "models" / "walkforward" / "refit_decision_b1.json"

pytestmark = pytest.mark.skipif(
    not (OUT / "metrics_val.json").exists(),
    reason="torch ensemble not trained (run scripts/train_torch.py)",
)


def test_five_seed_artifacts_exist():
    for seed in range(5):
        assert (OUT / f"net_seed{seed}.pt").exists()
    assert (OUT / "preprocess.json").exists()


def test_metrics_sane_and_honest():
    metrics = json.loads((OUT / "metrics_val.json").read_text())
    winner = metrics["winner_ensemble"]
    assert 0.55 < winner["accuracy"] < 0.70   # >0.70 would smell like leakage
    assert winner["log_loss"] < 0.6777        # must at least beat Elo
    assert len(metrics["per_seed"]) == 5
    temperatures = [seed["temperature"] for seed in metrics["per_seed"]]
    assert all(0.5 <= t <= 3.0 for t in temperatures)
    if metrics.get("mode") == "refit_through":
        # No held-out slice: the numbers above are the harness's pooled
        # walk-forward metrics and the per-seed spread has no equivalent.
        assert winner["mean_seed_spread"] is None
        assert metrics["walkforward_pooled"]["winner_log_loss"] == winner["log_loss"]
        assert metrics["n_train"] > 10_000  # every decisive fight through train_through
        assert all(seed["epochs_run"] == metrics["budget"] for seed in metrics["per_seed"])
        assert all(seed["best_val_log_loss"] is None for seed in metrics["per_seed"])
    else:
        assert 0 < winner["mean_seed_spread"] < 0.5


def test_refit_metrics_provenance_matches_harness_and_decision():
    """The evidence a refit metrics file quotes must be the committed harness
    report, and its budget/temperature must be the committed refit decision's
    -- otherwise the deployed model and its stated evidence have drifted."""
    metrics = json.loads((OUT / "metrics_val.json").read_text())
    if metrics.get("mode") != "refit_through":
        pytest.skip("split-mode metrics carry their own held-out numbers")
    report = json.loads((ROOT / metrics["harness_report"]).read_text())
    assert metrics["walkforward_pooled"] == report["pooled"]
    assert metrics["harness_features_max_date"] == report["config"]["features_max_date"]
    assert metrics["harness_fold_years"] == report["fold_years"]
    # The harness saw at least the data the deployed model trained on -- after
    # a weekly retrain on newer data this goes stale until something
    # re-measures, so it must not fail the suite (and thus the Action's commit
    # and predict steps) every week. It must not warn every week either: a
    # passing re-validation over the training data IS that measurement, so the
    # shared read in `mma.staleness` -- the same one the train scripts and the
    # weekly `--check-staleness` use -- decides whether there is anything to say.
    warning = stale_harness_warning(
        metrics["train_through"],
        metrics["harness_features_max_date"],
        revalidation_cover(load_revalidation(ROOT)),
    )
    if warning:
        warnings.warn(warning)

    decision = json.loads(REFIT_DECISION.read_text())["torch"]["budget"]
    assert metrics["budget"] == decision["fixed_epochs"]
    assert metrics["temperature"] == decision["temperature"]
    assert all(seed["temperature"] == decision["temperature"] for seed in metrics["per_seed"])
