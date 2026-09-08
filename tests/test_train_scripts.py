"""Pure-function tests for the train scripts' refit-through mode (budget
parsing, mode resolution, and the metrics-file assembly). Importing the
scripts has no side effects: they define main() under __main__."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma.models.xgb import feature_frame

import scripts.train_torch as train_torch
import scripts.train_xgb as train_xgb

POOLED = {
    "n": 4804, "winner_log_loss": 0.6512, "accuracy": 0.6184, "brier": 0.23,
    "ece": 0.0121, "joint_log_loss": 2.314, "method_macro_f1": 0.4036,
    "round_macro_f1": 0.3146, "n_method": 4792, "n_round": 2446,
}
HARNESS_MAX_DATE = "2026-08-08"
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def _args(**overrides):
    base = {"train_end": None, "val_start": None, "val_end": None, "refit_through": None}
    base.update(overrides)
    return argparse.Namespace(**base)


# --- XGB -------------------------------------------------------------------

def test_xgb_parse_budget_json_dict():
    assert train_xgb.parse_budget('{"winner": 81, "method": 79, "round": 75}') == {
        "winner": 81, "method": 79, "round": 75,
    }


def test_xgb_parse_budget_int_applies_to_all_heads():
    assert train_xgb.parse_budget("81") == {"winner": 81, "method": 81, "round": 81}
    assert train_xgb.parse_budget(81) == {"winner": 81, "method": 81, "round": 81}


def test_xgb_parse_budget_rejects_bad_shapes():
    with pytest.raises(ValueError, match="missing"):
        train_xgb.parse_budget('{"winner": 81}')
    with pytest.raises(ValueError, match="extra"):
        train_xgb.parse_budget('{"winner": 81, "method": 79, "round": 75, "extra": 1}')
    with pytest.raises(ValueError, match="positive"):
        train_xgb.parse_budget("0")
    with pytest.raises(ValueError):
        train_xgb.parse_budget("1.5")


def test_xgb_seed_ensemble_is_the_same_five_seeds_the_harness_measured():
    """The deployed XGB member has to be the construction B1 was scored with,
    not a lookalike: same seeds, same per-seed params."""
    from mma.candidates import XGBCandidate
    from scripts.run_walkforward import DEFAULT_SEEDS

    assert train_xgb.SEEDS == DEFAULT_SEEDS == (0, 1, 2, 3, 4)
    candidate = XGBCandidate(seeds=train_xgb.SEEDS)
    for seed in train_xgb.SEEDS:
        assert candidate._seed_params(seed) == {"random_state": int(seed)}


def test_xgb_head_path_is_one_artifact_per_head_per_seed():
    from mma.versioning import MODEL_ARTIFACT_GLOBS

    import fnmatch
    for head in ("winner", "method", "round"):
        for seed in train_xgb.SEEDS:
            path = train_xgb.head_path(Path("models"), head, seed)
            assert path.name == f"xgb_{head}_seed{seed}.json"
            # every artifact the trainer writes must be one the model hash covers
            assert any(fnmatch.fnmatch(f"models/{path.name}", glob)
                       for glob in MODEL_ARTIFACT_GLOBS), path


def test_xgb_mean_proba_averages_members_and_leaves_a_lone_member_untouched():
    class _Fake:
        def __init__(self, value):
            self.value = value

        def predict_proba(self, x):
            return np.full((len(x), 2), self.value)

    x = [0, 1, 2]
    only = _Fake(0.25)
    assert train_xgb.mean_proba([only], x) is not None
    np.testing.assert_array_equal(train_xgb.mean_proba([only], x), only.predict_proba(x))
    averaged = train_xgb.mean_proba([_Fake(0.2), _Fake(0.4)], x)
    np.testing.assert_allclose(averaged, np.full((3, 2), 0.3))


def test_xgb_refit_writes_a_booster_per_head_per_seed_and_they_differ(tmp_path):
    """End-to-end on a tiny frame: the refit path must leave fifteen artifacts,
    and the five winner boosters must not be five copies of one fit -- that
    would be a seed ensemble in name only."""
    rng = np.random.default_rng(0)
    n = 260
    features = pd.DataFrame({
        "fight_id": [f"f{i}" for i in range(n)],
        "date": pd.date_range("2015-01-01", periods=n, freq="7D"),
        "swapped": False,
        "weight_class": "Lightweight",
        "elo_diff": rng.normal(size=n),
        "reach_diff": rng.normal(size=n),
        "y_winner": rng.integers(0, 2, size=n).astype(float),
        "y_method": [["ko_tko", "submission", "decision"][i % 3] for i in range(n)],
        "y_finish_round": [["1", "2", "3", "45"][i % 4] for i in range(n)],
    })
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "pooled": POOLED, "fold_years": FOLD_YEARS,
        "config": {"features_max_date": HARNESS_MAX_DATE},
    }))
    args = argparse.Namespace(refit_through="latest", budget={"winner": 4, "method": 4, "round": 4},
                              report=report)
    x = feature_frame(features)
    metrics, winners = train_xgb.run_refit(features, x, args, tmp_path)

    assert len(winners) == len(train_xgb.SEEDS) == 5
    assert metrics["seeds"] == list(train_xgb.SEEDS)
    written = sorted(q.name for q in tmp_path.glob("xgb_*_seed*.json"))
    assert written == sorted(
        f"xgb_{head}_seed{seed}.json"
        for head in ("winner", "method", "round") for seed in train_xgb.SEEDS
    )
    probs = [model.predict_proba(x)[:, 1] for model in winners]
    assert not all(np.array_equal(probs[0], other) for other in probs[1:])
    ensembled = train_xgb.mean_proba(winners, x)[:, 1]
    np.testing.assert_allclose(ensembled, np.mean(probs, axis=0))


def test_xgb_refit_metrics_shape():
    budget = {"winner": 81, "method": 79, "round": 75}
    out = train_xgb.refit_metrics("2026-08-08", 11238, budget, "models/walkforward/xgb_refit.json", POOLED,
                                  HARNESS_MAX_DATE, FOLD_YEARS)
    assert out["mode"] == "refit_through"
    assert out["train_through"] == "2026-08-08" and out["n_train"] == 11238
    assert out["budget"] == budget and out["walkforward_pooled"] == POOLED
    assert out["harness_report"] == "models/walkforward/xgb_refit.json"
    assert out["harness_features_max_date"] == HARNESS_MAX_DATE
    assert out["harness_fold_years"] == FOLD_YEARS
    assert out["seeds"] == list(train_xgb.SEEDS)
    # README-facing blocks keep their key names
    assert out["winner"] == {
        "n_val": 4804, "accuracy": 0.6184, "log_loss": 0.6512, "brier": 0.23,
        "best_iteration": 81, "source": out["winner"]["source"],
    }
    assert "models/walkforward/xgb_refit.json" in out["winner"]["source"]
    for head, f1, n in (("method", 0.4036, 4792), ("finish_round", 0.3146, 2446)):
        block = out[head]
        assert block["n_val"] == n and block["macro_f1"] == f1
        assert block["accuracy"] is None and block["majority_baseline_accuracy"] is None
        assert "null" in block["source"]


# --- torch -----------------------------------------------------------------

def test_torch_parse_budget():
    assert train_torch.parse_budget("14") == 14
    assert train_torch.parse_budget(14) == 14
    with pytest.raises(ValueError, match="positive"):
        train_torch.parse_budget("0")
    with pytest.raises(ValueError):
        train_torch.parse_budget("fourteen")
    with pytest.raises(ValueError):
        train_torch.parse_budget(14.5)


def test_torch_refit_metrics_shape():
    per_seed = [
        {"seed": s, "best_epoch": 13, "best_val_log_loss": None, "epochs_run": 14, "temperature": 1.1}
        for s in range(5)
    ]
    out = train_torch.refit_metrics("2026-08-08", 11238, 14, 1.1,
                                    "models/walkforward/torch_refit.json", POOLED, per_seed,
                                    HARNESS_MAX_DATE, FOLD_YEARS)
    assert out["mode"] == "refit_through"
    assert out["train_through"] == "2026-08-08" and out["n_train"] == 11238
    assert out["budget"] == 14 and out["temperature"] == 1.1
    assert out["walkforward_pooled"] == POOLED
    assert out["harness_features_max_date"] == HARNESS_MAX_DATE
    assert out["harness_fold_years"] == FOLD_YEARS
    winner = out["winner_ensemble"]
    assert winner["n_val"] == 4804 and winner["accuracy"] == 0.6184
    assert winner["log_loss"] == 0.6512 and winner["brier"] == 0.23
    assert winner["mean_seed_spread"] is None
    assert len(out["per_seed"]) == 5
    for seed, entry in enumerate(out["per_seed"]):
        assert entry == {"seed": seed, "best_epoch": 13, "best_val_log_loss": None,
                         "temperature": 1.1, "epochs_run": 14}
    assert out["method_ensemble"] == {
        "n_val": 4792, "accuracy": None, "macro_f1": 0.4036,
        "source": out["method_ensemble"]["source"],
    }


# --- shared plumbing -------------------------------------------------------

@pytest.mark.parametrize("module", [train_xgb, train_torch])
def test_resolve_mode(module):
    assert module.resolve_mode(_args()) == module.DEFAULT_MODE
    assert module.resolve_mode(_args(refit_through="latest")) == module.MODE_REFIT
    assert module.resolve_mode(_args(train_end="2021-01-01")) == module.MODE_SPLIT
    assert module.resolve_mode(_args(val_end="2023-12-31")) == module.MODE_SPLIT
    # roll_window.py --execute passes all three explicitly
    assert module.resolve_mode(_args(train_end="2024-08-08", val_start="2024-08-08",
                                     val_end="2026-08-08")) == module.MODE_SPLIT
    with pytest.raises(SystemExit, match="cannot be combined"):
        module.resolve_mode(_args(refit_through="latest", train_end="2021-01-01"))


@pytest.mark.parametrize("module", [train_xgb, train_torch])
def test_stale_harness_warning(module):
    # training data no newer than the harness saw -> no warning
    assert module.stale_harness_warning("2026-08-08", "2026-08-08") is None
    assert module.stale_harness_warning("2026-08-01", "2026-08-08") is None
    # a weekly refresh moved train_through past the harness's data
    warning = module.stale_harness_warning("2026-08-15", "2026-08-08")
    assert warning is not None
    assert "harness evidence predates this training data" in warning
    assert "scripts/run_walkforward.py" in warning


@pytest.mark.parametrize("module", [train_xgb, train_torch])
def test_refit_cutoff(module):
    features = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2026-08-08", "2024-05-05"])})
    assert module.refit_cutoff(features, "latest") == pd.Timestamp("2026-08-08")
    assert module.refit_cutoff(features, "2024-12-31") == pd.Timestamp("2024-12-31")
