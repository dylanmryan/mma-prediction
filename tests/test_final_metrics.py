"""Structural checks on the one-time held-out test metrics artifact.

These tests only validate the shape and plausibility of the committed
models/final_test_metrics.json -- they never recompute anything on the
test years, and they must never trigger a rerun of the evaluation.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "final_test_metrics.json"

pytestmark = pytest.mark.skipif(
    not METRICS_PATH.exists(),
    reason="one-time final test evaluation not run",
)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_PATH.read_text())


def test_top_level_keys_present(metrics):
    assert {"computed_once_on", "test_split", "n_test", "winner",
            "method", "finish_round"} <= set(metrics)
    assert metrics["test_split"] == "date >= 2024-01-01"


def test_all_winner_models_reported(metrics):
    assert set(metrics["winner"]) == {
        "coin_flip", "higher_elo_dummy", "elo", "xgboost", "torch_ensemble"
    }


def test_n_test_counts_plausible(metrics):
    for key in ("winner_features", "winner_elo"):
        assert 700 <= metrics["n_test"][key] <= 1200
    assert 0 < metrics["n_test"]["finish_round"] <= metrics["n_test"]["method"]
    assert metrics["n_test"]["method"] <= metrics["n_test"]["winner_features"]


def test_winner_metrics_in_plausible_ranges(metrics):
    for name, block in metrics["winner"].items():
        assert 0.4 < block["accuracy"] < 0.75, name
        if block["log_loss"] is not None:  # dummy reports accuracy only
            assert 0.5 < block["log_loss"] < 0.8, name
            assert 0.0 < block["brier"] < 0.3, name


def test_multiclass_blocks_have_baselines(metrics):
    for task, models in (
        ("method", ("xgboost", "torch_ensemble")),
        ("finish_round", ("xgboost",)),
    ):
        for model in models:
            block = metrics[task][model]
            assert 0.0 <= block["accuracy"] <= 1.0
            assert 0.0 <= block["macro_f1"] <= 1.0
            assert 0.0 <= block["majority_baseline"]["accuracy"] <= 1.0
