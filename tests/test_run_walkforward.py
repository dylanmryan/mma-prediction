"""Argument handling for scripts/run_walkforward.py's fit-budget flags.

A budget is either derived from a reference report (--fixed-budget-from) or
stated outright (--fixed-epochs/--temperature, added for the SP2 recency
task's recent-fold temperature experiment). These are pure-function tests of
resolve_budget: importing the script has no side effects, and none of these
cases run the harness.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

import scripts.run_walkforward as rwf


def _args(**overrides) -> argparse.Namespace:
    base = {
        "candidate": "torch", "fixed_budget_from": None,
        "fixed_epochs": None, "temperature": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_no_flags_means_early_stopping():
    assert rwf.resolve_budget(_args()) is None


def test_fixed_epochs_alone_defaults_the_temperature():
    assert rwf.resolve_budget(_args(fixed_epochs=18)) == {"fixed_epochs": 18}


def test_fixed_epochs_with_temperature():
    assert rwf.resolve_budget(_args(fixed_epochs=18, temperature=0.98)) == {
        "fixed_epochs": 18, "temperature": 0.98,
    }


def test_reference_report_still_works(tmp_path):
    report = tmp_path / "ref.json"
    report.write_text(json.dumps({
        "fit_info": {"best_epoch": [[6, 8], [13, 15]], "temperature": [[1.0, 1.2], [0.8, 1.0]]}
    }))
    assert rwf.resolve_budget(_args(fixed_budget_from=report)) == {
        "fixed_epochs": 11, "temperature": 1.0,
    }


def test_reference_report_and_explicit_budget_conflict(tmp_path):
    report = tmp_path / "ref.json"
    report.write_text(json.dumps({"fit_info": {"best_epoch": [[6]]}}))
    with pytest.raises(SystemExit, match="mutually exclusive"):
        rwf.resolve_budget(_args(fixed_budget_from=report, fixed_epochs=18))
    with pytest.raises(SystemExit, match="mutually exclusive"):
        rwf.resolve_budget(_args(fixed_budget_from=report, temperature=0.98))


def test_temperature_without_a_budget_is_a_usage_error():
    with pytest.raises(SystemExit, match="fixed-budget mode only"):
        rwf.resolve_budget(_args(temperature=0.98))


@pytest.mark.parametrize("candidate", ["xgb", "elo"])
def test_explicit_budget_is_torch_only(candidate):
    with pytest.raises(SystemExit, match="torch candidate only"):
        rwf.resolve_budget(_args(candidate=candidate, fixed_epochs=18))


def test_nonsense_values_are_rejected():
    with pytest.raises(SystemExit, match="at least 1"):
        rwf.resolve_budget(_args(fixed_epochs=0))
    with pytest.raises(SystemExit, match="must be positive"):
        rwf.resolve_budget(_args(fixed_epochs=18, temperature=0.0))


# --- --drop-columns ---------------------------------------------------------
# The ablation path: hold columns out of the MODEL MATRIX while leaving them in
# the feature TABLE, so a paired incumbent can be computed on the same table as
# the candidate and `walkforward.slice_masks` still reports a slice keyed on a
# dropped column.


def _drop_args(**overrides) -> argparse.Namespace:
    base = {"candidate": "torch", "drop_columns": None}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_no_flag_drops_nothing():
    assert rwf.resolve_drop_columns(_drop_args()) == ()


def test_names_are_split_stripped_and_deduplicated_in_order():
    assert rwf.resolve_drop_columns(
        _drop_args(drop_columns=" external_missing , same_country ,external_missing,")
    ) == ("external_missing", "same_country")


def test_a_flag_that_names_nothing_is_a_usage_error():
    with pytest.raises(SystemExit, match="names no columns"):
        rwf.resolve_drop_columns(_drop_args(drop_columns=" , "))


def test_dropping_columns_from_the_elo_candidate_is_a_usage_error():
    with pytest.raises(SystemExit, match="fitted candidates only"):
        rwf.resolve_drop_columns(_drop_args(candidate="elo", drop_columns="elo_diff"))


def test_an_unknown_column_is_rejected_rather_than_silently_dropping_nothing():
    features = pd.DataFrame({"elo_diff": [1.0], "external_missing": [True]})
    rwf.check_drop_columns(("external_missing",), features)  # present: fine
    with pytest.raises(SystemExit, match="absent from the feature table"):
        rwf.check_drop_columns(("external_missing", "typo_diff"), features)


def _ablation_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "fight_id": ["f1", "f2", "f3", "f4"],
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"]),
        "swapped": [False, True, False, True],
        "y_winner": [1.0, 0.0, 1.0, 0.0],
        "y_method": ["KO/TKO", "Decision", "KO/TKO", "Decision"],
        "y_finish_round": ["1", None, "2", None],
        "weight_class": ["Lightweight"] * 4,
        "elo_diff": [10.0, -20.0, 30.0, -40.0],
        "age_diff": [1.5, -2.5, 3.5, -4.5],
        "career_fights_diff": [2.0, -1.0, 0.0, 3.0],
    })


def test_a_dropped_column_is_absent_from_both_learners_model_matrices():
    """The property the flag exists for, checked on the real builders."""
    from mma.models.xgb import feature_frame
    from mma.tensors import Preprocessor

    features = _ablation_frame()
    dropped = ("age_diff", "career_fights_diff")
    train = np.array([True, True, False, False])

    kept = feature_frame(features)
    assert {"elo_diff", *dropped} <= set(kept.columns)
    ablated = feature_frame(features, dropped)
    assert set(ablated.columns) == set(kept.columns) - set(dropped)
    assert "elo_diff" in ablated.columns

    full = Preprocessor.fit(features, train_mask=train)
    assert {"elo_diff", *dropped} <= set(full.numeric_columns)
    thin = Preprocessor.fit(features, train_mask=train, drop_columns=dropped)
    assert set(thin.numeric_columns) == set(full.numeric_columns) - set(dropped)
    assert thin.transform(features)[0].shape[1] == len(thin.numeric_columns)


def test_drop_columns_reaches_the_candidates_the_cli_builds():
    xgb = rwf.build_candidate("xgb", "x", "0", {}, None, ("age_diff",))
    torch_candidate = rwf.build_candidate("torch", "t", "0,1", {}, None, ("age_diff",))
    assert xgb.drop_columns == ("age_diff",)
    assert torch_candidate.drop_columns == ("age_diff",)
