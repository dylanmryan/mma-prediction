import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "models" / "torch"

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
