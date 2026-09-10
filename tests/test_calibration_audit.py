"""The calibration audit: what its bound covers, and what it does not.

The audit's value is not its reliability table -- that only says a band looks
off. It is the pair of answers to "what would fixing it be worth?": an oracle
that bounds BAND-LEVEL repair, and a direct trial of real calibrators fitted
out of sample.

The bound is the part that can do harm, because a bound that could be beaten
would rule remedies out wrongly. So these tests pin exactly where it holds
(no constant beats the realised rate) and exactly where it does not (a fitted
temperature beats it outright, by keeping the ordering the constant destroys).
That second test is why `recalibrator_trial` exists rather than being
redundant, and it failed when first written -- the bound had been described as
covering every calibrator, which it does not.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibration_audit import (
    BAND_EDGES,
    MIN_BAND_N,
    PROJECT_BAR,
    audit,
    band_masks,
    favoured_view,
    log_loss,
    oracle_gain,
    oracle_gain_all,
    verdict,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "calibration_audit.json"


# --- reading reliability from the favourite's side ---------------------------

def test_favoured_view_folds_both_corners_onto_one_axis():
    p = np.array([0.85, 0.15, 0.60])
    y = np.array([1.0, 0.0, 0.0])
    fav_p, fav_win = favoured_view(p, y)
    # 0.15 on A is 0.85 on B, and B won, so the favourite was right both times
    assert fav_p.tolist() == pytest.approx([0.85, 0.85, 0.60])
    assert fav_win.tolist() == [1.0, 1.0, 0.0]


def test_a_symmetric_model_hides_its_own_error_when_read_per_corner():
    """Why `favoured_view` exists: mirrored rows average the error away.

    Every over-confident 0.9 is paired with its mirror 0.1. Read per corner
    the mean prediction is 0.5 and the mean outcome is 0.5, so the model looks
    perfectly calibrated. Read from the favourite's side it is 0.9 promised
    against 0.6 delivered -- which is the error that is actually there.
    """
    p = np.array([0.9, 0.1] * 5)
    #                A won      B won      B won      A won      A won
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    assert p.mean() == pytest.approx(0.5)
    assert y.mean() == pytest.approx(0.7), "read per corner: 0.5 predicted, 0.7 seen"
    fav_p, fav_win = favoured_view(p, y)
    assert fav_p.mean() == pytest.approx(0.9), "0.9 promised"
    assert fav_win.mean() == pytest.approx(0.6), "0.6 delivered -- the real error"


# --- the bands ---------------------------------------------------------------

def test_bands_partition_every_favoured_probability_exactly_once():
    fav_p = np.linspace(0.5, 1.0, 501)
    covered = np.zeros(len(fav_p), dtype=int)
    for _, _, mask in band_masks(fav_p):
        covered += mask.astype(int)
    assert covered.tolist() == [1] * len(fav_p), "bands must tile [0.5, 1.0]"


def test_the_top_band_is_closed_so_a_certainty_is_not_dropped():
    fav_p = np.array([1.0])
    assert sum(int(m.sum()) for _, _, m in band_masks(fav_p)) == 1


# --- what the oracle bounds, and what escapes it -----------------------------

def _band_only(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.80, 0.90, n)
    y = (rng.uniform(size=n) < 0.70).astype(float)  # really 0.70, not 0.85
    return p, y


def test_the_oracle_beats_every_other_constant_for_that_band():
    """The realised rate is the best constant a band can be repaired with.

    That is what makes it a bound: any calibrator maps the band to SOME value,
    and none of them can beat the one fitted on the answer.
    """
    p, y = _band_only()
    mask = np.ones(len(p), dtype=bool)
    best = oracle_gain(p, y, mask)
    for candidate in (0.5, 0.6, 0.65, 0.75, 0.8, 0.85, 0.9):
        repaired = np.full_like(p, candidate)
        assert log_loss(y, p) - log_loss(y, repaired) <= best + 1e-9


def test_a_fitted_temperature_CAN_beat_the_oracle_which_is_the_bounds_limit():
    """The oracle bounds BAND-LEVEL repair, not every calibrator.

    Collapsing a band to one constant discards the model's ordering inside it.
    A temperature keeps that ordering and only moves the level, so on a band
    with real internal discrimination it beats the "oracle" outright. Pinned
    here because the script's conclusion would be unsound if it rested on the
    bound alone -- `recalibrator_trial` is what closes that gap, and this is
    the test that says why it has to exist.
    """
    p, y = _band_only(seed=3)
    best_constant = oracle_gain(p, y, np.ones(len(p), dtype=bool))
    logit = np.log(p / (1 - p))
    gains = [log_loss(y, p) - log_loss(y, 1 / (1 + np.exp(-logit * t)))
             for t in np.linspace(0.3, 2.0, 30)]
    assert max(gains) > best_constant, (
        "if no temperature beat the constant, the oracle would bound every "
        "monotone remedy and the direct trial would be redundant"
    )


def test_the_oracle_is_zero_when_a_band_is_already_perfect():
    p = np.full(200, 0.75)
    y = np.zeros(200)
    y[:150] = 1.0  # exactly 0.75
    assert oracle_gain(p, y, np.ones(200, dtype=bool)) == pytest.approx(0.0, abs=1e-9)


def test_the_oracle_repairs_both_orientations_of_a_band():
    """A band holds rows the model favoured A in and rows it favoured B in.
    Repairing only one orientation would understate the bound."""
    p = np.array([0.85, 0.15, 0.85, 0.15])
    y = np.array([1.0, 0.0, 0.0, 1.0])  # favourite went 2-2, not 0.85
    gain = oracle_gain(p, y, np.ones(4, dtype=bool))
    repaired = np.array([0.5, 0.5, 0.5, 0.5])
    assert gain == pytest.approx(log_loss(y, p) - log_loss(y, repaired), abs=1e-9)


def test_repairing_every_band_can_lose_to_repairing_the_best_one():
    """Collapsing a band to one number throws away the ordering inside it.

    This is not a quirk -- it is why SP2.2's isotonic remediation scored 0.7049
    against the temperature form's 0.6437. A coarse recalibrator buys
    calibration by spending discrimination.
    """
    rng = np.random.default_rng(1)
    p = rng.uniform(0.5, 1.0, 3000)
    y = (rng.uniform(size=3000) < p).astype(float)  # perfectly calibrated
    tested = [(lo, hi, m) for lo, hi, m in band_masks(favoured_view(p, y)[0])
              if int(m.sum()) >= MIN_BAND_N]
    assert oracle_gain_all(p, y, tested) < 0, (
        "flattening a calibrated model inside each band must cost log-loss"
    )


# --- the multiplicity arithmetic --------------------------------------------

def test_the_threshold_is_the_band_count_not_a_choice():
    dump = _synthetic_dump()
    report = audit(dump)
    n_looks = report["method"]["n_looks"]
    assert report["method"]["bonferroni_threshold"] == round(0.05 / n_looks, 5)
    assert n_looks == len(report["bands"])


def test_a_band_below_the_minimum_is_not_tested():
    assert MIN_BAND_N >= 20
    dump = _synthetic_dump(n=60)
    report = audit(dump)
    assert all(r["n"] >= MIN_BAND_N for r in report["bands"])


def test_verdict_separates_nominal_from_corrected_and_from_worth_fixing():
    rows = [
        {"band": "[0.80,1.00]", "nominally_off": True,
         "survives_bonferroni": False, "oracle_clears_bar": False},
        {"band": "[0.50,0.55)", "nominally_off": False,
         "survives_bonferroni": False, "oracle_clears_bar": False},
    ]
    v = verdict(rows, 0.00714, PROJECT_BAR)
    assert v["miscalibrated_bands"] == []
    assert v["nominally_off_but_not_past_correction"] == ["[0.80,1.00]"]
    assert v["calibration_is_sound"] is True
    assert v["a_remedy_is_worth_building"] is False


def test_a_band_that_clears_the_bar_under_an_oracle_is_reported_as_worth_fixing():
    rows = [{"band": "[0.80,1.00]", "nominally_off": True,
             "survives_bonferroni": True, "oracle_clears_bar": True}]
    v = verdict(rows, 0.05, PROJECT_BAR)
    assert v["a_remedy_is_worth_building"] is True
    assert v["calibration_is_sound"] is False


def _synthetic_dump(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < p).astype(float)
    return {"p_winner": p.tolist(), "y_winner": y.tolist(),
            "fold_year": rng.integers(2018, 2026, n).tolist()}


# --- the committed artifact --------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit not run")
def test_the_committed_audit_found_no_remedy_worth_building():
    """If this ever fails, a calibration remedy has become worth specifying
    and the README paragraph saying otherwise is out of date."""
    report = json.loads(ARTIFACT.read_text())
    assert report["oracle_ceiling"]["bar"] == PROJECT_BAR
    assert report["oracle_ceiling"]["best_single_band"] < PROJECT_BAR
    assert report["verdict"]["a_remedy_is_worth_building"] is False


@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit not run")
def test_the_committed_audit_tested_every_band_it_says_it_did():
    report = json.loads(ARTIFACT.read_text())
    assert report["method"]["bands"] == [
        f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:])
    ]
    assert report["provenance"]["n"] == sum(r["n"] for r in report["bands"])
