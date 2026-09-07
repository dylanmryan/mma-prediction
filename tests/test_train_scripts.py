"""Pure-function tests for the train scripts' refit-through mode (budget
parsing, mode resolution, and the metrics-file assembly). Importing the
scripts has no side effects: they define main() under __main__."""
from __future__ import annotations

import argparse

import pandas as pd
import pytest

import scripts.train_torch as train_torch
import scripts.train_xgb as train_xgb

POOLED = {
    "n": 4804, "winner_log_loss": 0.6512, "accuracy": 0.6184, "brier": 0.23,
    "ece": 0.0121, "joint_log_loss": 2.314, "method_macro_f1": 0.4036,
    "round_macro_f1": 0.3146, "n_method": 4792, "n_round": 2446,
}


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


def test_xgb_refit_metrics_shape():
    budget = {"winner": 81, "method": 79, "round": 75}
    out = train_xgb.refit_metrics("2026-08-08", 11238, budget, "models/walkforward/xgb_refit.json", POOLED)
    assert out["mode"] == "refit_through"
    assert out["train_through"] == "2026-08-08" and out["n_train"] == 11238
    assert out["budget"] == budget and out["walkforward_pooled"] == POOLED
    assert out["harness_report"] == "models/walkforward/xgb_refit.json"
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
                                    "models/walkforward/torch_refit.json", POOLED, per_seed)
    assert out["mode"] == "refit_through"
    assert out["train_through"] == "2026-08-08" and out["n_train"] == 11238
    assert out["budget"] == 14 and out["temperature"] == 1.1
    assert out["walkforward_pooled"] == POOLED
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
def test_refit_cutoff(module):
    features = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2026-08-08", "2024-05-05"])})
    assert module.refit_cutoff(features, "latest") == pd.Timestamp("2026-08-08")
    assert module.refit_cutoff(features, "2024-12-31") == pd.Timestamp("2024-12-31")
