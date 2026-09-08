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
