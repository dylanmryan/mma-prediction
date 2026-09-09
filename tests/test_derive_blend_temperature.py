"""The deployed blend temperature must be derived the way deployment gets it.

The harness fits one temperature per fold on that fold's inner-validation
year; deployment applies a single fixed value with no held-out year at all, so
the two are different forms of the same model and only one of them ships. The
number the project publishes has to describe the one that ships.

`scripts/derive_blend_temperature.py` re-scores the committed per-row dump
(`models/walkforward/preds/blend_b1.json`) under each candidate fixed-temperature
rule. Everything it does rests on one invertibility fact -- the dump's
probabilities are `sigmoid(logit(p_pre) / T_fold)` and nothing else, so
`p_pre` is recoverable exactly -- and on one leakage property: a fold's
walk-forward temperature is fitted on strictly earlier folds. Both are tested
here, because the whole derivation is worthless if either fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mma.blend import apply_temperature, logit  # noqa: E402
from mma.evaluate import expected_calibration_error, log_loss  # noqa: E402
from scripts.derive_blend_temperature import (  # noqa: E402
    ARTIFACT, DUMP, REPORT, apply_per_fold, build_record, deployed_temperature,
    fit_temperature_on, invert_per_fold, load_inputs, pooled_metrics,
    walkforward_temperatures,
)


@pytest.fixture(scope="module")
def inputs():
    return load_inputs(REPORT, DUMP)


# --- the inversion round-trip ----------------------------------------------


def test_inversion_then_reapplication_reproduces_the_committed_dump(inputs):
    """`invert_per_fold` is the exact inverse of the harness's own scaling."""
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    back = apply_per_fold(pre, inputs["years"], inputs["fold_temperatures"])
    assert np.abs(back - inputs["p_scored"]).max() < 1e-12


def test_the_round_trip_reproduces_the_reports_own_pooled_metrics(inputs):
    """The recovered pre-temperature probabilities, re-scaled per fold, score
    the report's published pooled log-loss and ECE at the report's own 4 dp.
    If this ever fails the dump and the report describe different runs and no
    number derived from either can be trusted."""
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    back = apply_per_fold(pre, inputs["years"], inputs["fold_temperatures"])
    pooled = inputs["report"]["pooled"]
    assert round(log_loss(inputs["y"], back), 4) == pooled["winner_log_loss"]
    assert round(expected_calibration_error(inputs["y"], back), 4) == pooled["ece"]


def test_the_recovered_probabilities_are_not_simply_the_dumped_ones(inputs):
    """Guards a no-op inversion passing the round-trip test vacuously."""
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    assert np.abs(pre - inputs["p_scored"]).max() > 0.01


def test_inversion_is_per_fold_and_uses_that_folds_own_temperature(inputs):
    """Every fold's rows invert under that fold's recorded T, not a shared one."""
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    for year, temperature in inputs["fold_temperatures"].items():
        mask = inputs["years"] == year
        expected = 1.0 / (1.0 + np.exp(-logit(inputs["p_scored"][mask]) * temperature))
        assert np.abs(pre[mask] - expected).max() < 1e-12


# --- the walk-forward construction -----------------------------------------


def _toy():
    rng = np.random.default_rng(0)
    years = np.repeat([2018, 2019, 2020, 2021], 400)
    pre = rng.uniform(0.05, 0.95, size=years.size)
    y = (rng.uniform(size=years.size) < pre).astype(float)
    return pre, y, years


def test_the_first_fold_uses_its_own_inner_val_fit():
    """There is no earlier fold to fit on, so the harness's own fitted value
    stands -- and the derivation says so rather than silently inventing one."""
    pre, y, years = _toy()
    temps = walkforward_temperatures(pre, y, years, first_fold_temperature=1.23)
    assert temps[2018] == 1.23


def test_a_folds_temperature_never_sees_that_fold_or_any_later_one():
    """Perturbing fold Y's labels must leave fold Y's temperature -- and every
    earlier fold's -- untouched. This is the leakage property the whole
    derivation rests on."""
    pre, y, years = _toy()
    base = walkforward_temperatures(pre, y, years, first_fold_temperature=1.0)
    for target in (2019, 2020, 2021):
        perturbed = y.copy()
        perturbed[years == target] = 1.0 - perturbed[years == target]
        after = walkforward_temperatures(pre, perturbed, years, first_fold_temperature=1.0)
        for year in (2018, 2019, 2020, 2021):
            if year <= target:
                assert after[year] == base[year], f"fold {year} moved when {target} changed"
        if target != 2021:
            assert any(after[year] != base[year] for year in (2018, 2019, 2020, 2021)
                       if year > target), f"nothing after {target} used it"


def test_an_earlier_folds_labels_do_move_the_later_temperatures():
    """The complement of the leakage test: prior folds are actually used."""
    pre, y, years = _toy()
    base = walkforward_temperatures(pre, y, years, first_fold_temperature=1.0)
    perturbed = y.copy()
    perturbed[years == 2018] = 1.0 - perturbed[years == 2018]
    after = walkforward_temperatures(pre, perturbed, years, first_fold_temperature=1.0)
    assert after[2018] == base[2018] == 1.0  # given, not fitted
    assert [after[y_] for y_ in (2019, 2020, 2021)] != [base[y_] for y_ in (2019, 2020, 2021)]


def test_each_walkforward_temperature_equals_a_fit_on_exactly_the_prior_folds():
    """Recomputed independently of `walkforward_temperatures`' own loop."""
    pre, y, years = _toy()
    temps = walkforward_temperatures(pre, y, years, first_fold_temperature=1.0)
    order = sorted(set(int(v) for v in years))
    for i, year in enumerate(order[1:], start=1):
        prior = np.isin(years, order[:i])
        assert temps[year] == fit_temperature_on(pre[prior], y[prior])


def test_the_deployed_temperature_is_the_rule_run_one_fold_past_the_last(inputs):
    """Deployment is the fold after the last one: every fold is 'before' it, so
    the walk-forward rule fits on all of them. That is the artifact's value."""
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    assert deployed_temperature(pre, inputs["y"]) == fit_temperature_on(pre, inputs["y"])


# --- the committed artifact -------------------------------------------------


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_committed_artifact_matches_a_fresh_derivation():
    assert json.loads(ARTIFACT.read_text()) == build_record()


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_the_artifact_deploys_the_walkforward_rule():
    record = json.loads(ARTIFACT.read_text())
    assert record["deployed"]["rule"] == "R2"
    assert record["deployed"]["temperature"] == record["rules"]["R2"]["deployment_temperature"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_blend_json_serves_the_derived_temperature():
    record = json.loads(ARTIFACT.read_text())
    config = json.loads((ROOT / "models" / "blend.json").read_text())
    assert config["temperature"] == record["deployed"]["temperature"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_the_published_pooled_numbers_are_recomputable_from_the_dump(inputs):
    """The deployed form's published log-loss and ECE are the walk-forward
    re-score of the committed dump, not a number typed into the artifact."""
    record = json.loads(ARTIFACT.read_text())
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    temps = walkforward_temperatures(
        pre, inputs["y"], inputs["years"],
        first_fold_temperature=record["rules"]["R2"]["per_fold_temperature"][
            str(min(inputs["fold_temperatures"]))],
    )
    scored = apply_per_fold(pre, inputs["years"], temps)
    assert pooled_metrics(inputs["y"], scored) == record["rules"]["R2"]["pooled"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_the_harness_form_row_reproduces_the_source_report(inputs):
    record = json.loads(ARTIFACT.read_text())
    pooled = inputs["report"]["pooled"]
    harness = record["rules"]["per_fold_fitted"]["pooled"]
    assert harness["winner_log_loss"] == pooled["winner_log_loss"]
    assert harness["ece"]["10"] == pooled["ece"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_the_superseded_median_rule_is_recorded_with_its_own_numbers():
    """R1 is the rule that shipped before this derivation. Keeping its measured
    numbers in the artifact is how the correction stays auditable."""
    record = json.loads(ARTIFACT.read_text())
    assert record["rules"]["R1"]["deployment_temperature"] == 0.8
    assert record["rules"]["R1"]["pooled"]["ece"]["10"] > record["rules"]["R2"]["pooled"]["ece"]["10"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_the_pooled_fit_is_labelled_as_in_sample_and_not_a_candidate():
    record = json.loads(ARTIFACT.read_text())
    assert record["rules"]["R3"]["candidate"] is False
    assert "in-sample" in record["rules"]["R3"]["note"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/derive_blend_temperature.py")
def test_every_rule_reports_all_four_bin_counts_and_every_fold():
    record = json.loads(ARTIFACT.read_text())
    years = [str(y) for y in json.loads(REPORT.read_text())["fold_years"]]
    for name, rule in record["rules"].items():
        assert sorted(rule["pooled"]["ece"]) == ["10", "15", "20", "5"], name
        assert sorted(rule["folds"]) == sorted(years), name
        for fold in rule["folds"].values():
            assert sorted(fold["ece"]) == ["10", "15", "20", "5"], name
