"""Report-set selection in scripts/refit_decision.py.

The refit A/B comparison is only meaningful within one feature table: a
budget derived from reports computed on the SP1 46-column table says nothing
about the SP2 `base,external` table the model is actually deployed on. The
script therefore names its inputs in `REPORT_SETS` rather than hard-coding
one set of report paths, and each set writes its own decision file. These
tests pin that wiring (and the rule that reads it) without running the
harness -- `build_decision` only reads committed JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.refit_decision as rd

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def noise_floor() -> dict:
    return json.loads(rd.NOISE_FLOOR.read_text())


def _report(pooled_ll: float, folds: dict, budget: dict) -> dict:
    return {
        "name": "fake",
        "config": {"budget": budget},
        "fold_years": sorted(int(y) for y in folds),
        "pooled": {"n": 100, "winner_log_loss": pooled_ll},
        "folds": {y: {"winner_log_loss": v, "n": 10} for y, v in folds.items()},
    }


def _write_set(tmp_path: Path, a_ll: float, b_ll: float) -> dict:
    """A complete report set on disk, with B's pooled log-loss under our control."""
    folds_a = {"2018": 0.66, "2019": 0.65}
    folds_b = {"2018": 0.66 + (b_ll - a_ll), "2019": 0.65 + (b_ll - a_ll)}
    paths = {}
    for name, (ll, folds, budget) in {
        "xgb_a": (a_ll, folds_a, {}),
        "xgb_b": (b_ll, folds_b, {"fixed_rounds": {"winner": 7, "method": 7, "round": 7}}),
        "torch_a": (a_ll, folds_a, {}),
        "torch_b": (b_ll, folds_b, {"fixed_epochs": 7, "temperature": 1.0}),
        "fresh_a": (a_ll, folds_a, {}),
        "fresh_b": (b_ll, folds_b, {"fixed_epochs": 7, "temperature": 1.0}),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_report(ll, folds, budget)))
        paths[name] = path
    return {
        "table": "a synthetic table",
        "out": tmp_path / "decision.json",
        "reports": {
            "xgb": {"A": paths["xgb_a"], "B": paths["xgb_b"]},
            "torch": {"A": paths["torch_a"], "B": paths["torch_b"]},
        },
        "fresh": {"A": paths["fresh_a"], "B": paths["fresh_b"]},
        "note": "a synthetic note",
    }


def test_every_report_set_names_a_distinct_output_file():
    outs = [entry["out"] for entry in rd.REPORT_SETS.values()]
    assert len(set(outs)) == len(outs)


def test_every_report_set_points_at_reports_that_exist():
    for name, entry in rd.REPORT_SETS.items():
        paths = [p for pair in entry["reports"].values() for p in pair.values()]
        paths += list(entry["fresh"].values())
        missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
        assert not missing, f"report set {name} names missing reports: {missing}"


def test_the_committed_decision_files_are_reproducible(noise_floor):
    """Each committed decision file is exactly what its report set produces --
    so the deployed budget can always be traced back to the reports."""
    for name, entry in rd.REPORT_SETS.items():
        if not entry["out"].exists():
            pytest.skip(f"{name} decision file not built yet")
        assert rd.build_decision(entry, noise_floor) == json.loads(entry["out"].read_text()), name


def test_the_decision_records_which_feature_table_it_was_taken_on(noise_floor, tmp_path):
    entry = _write_set(tmp_path, a_ll=0.65, b_ll=0.65)
    assert rd.build_decision(entry, noise_floor)["feature_table"] == "a synthetic table"


def test_recipe_is_refit_when_torch_b_is_not_worse_by_sigma(noise_floor, tmp_path):
    sigma = noise_floor["sigma_seed"]
    entry = _write_set(tmp_path, a_ll=0.65, b_ll=0.65 + sigma / 2)
    decision = rd.build_decision(entry, noise_floor)
    assert decision["torch"]["B_not_worse_than_A_by_sigma"]
    assert decision["deployment_recipe"] == rd.DEPLOYMENT_RECIPE


def test_recipe_falls_back_to_protocol_a_when_torch_b_loses(noise_floor, tmp_path):
    """The file must not record a recipe its own numbers do not support."""
    entry = _write_set(tmp_path, a_ll=0.65, b_ll=0.66)
    decision = rd.build_decision(entry, noise_floor)
    assert not decision["torch"]["B_not_worse_than_A_by_sigma"]
    assert decision["deployment_recipe"] == rd.SPLIT_RECIPE
