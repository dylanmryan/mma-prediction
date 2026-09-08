"""Pure helpers behind the SP2.1 decision artifact.

The artifact itself is regenerated from committed reports by
`scripts/sp2_1_decision.py`; what is worth testing is the arithmetic and the
branch table it applies, because those encode the pre-registered rule. None
of these tests read a report or run the harness.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import scripts.sp2_1_decision as dec
from scripts.blend_check import average_predictions
from scripts.noise_floor import seed_label

# --- spread -----------------------------------------------------------------


def test_spread_is_the_sample_standard_deviation():
    out = dec.spread([0.6467, 0.6472, 0.6488, 0.6493])
    assert out["n"] == 4
    assert out["mean"] == pytest.approx(0.648)
    assert out["sd"] == pytest.approx(np.std([0.6467, 0.6472, 0.6488, 0.6493], ddof=1))
    assert (out["min"], out["max"]) == (0.6467, 0.6493)


def test_a_single_run_measures_no_spread_at_all():
    assert math.isnan(dec.spread([0.65])["sd"])


# --- fresh_seed_confirmation ------------------------------------------------
# The confirmation is a PAIRED MEAN over seeds: the point is that one lucky
# seed cannot carry a result that re-seeding does not reproduce.


def test_confirmation_passes_only_when_the_mean_delta_beats_the_bar():
    out = dec.fresh_seed_confirmation([0.640, 0.641, 0.642], [0.650, 0.651, 0.652], 0.001)
    assert out["bar"] == 0.003
    assert out["mean_delta"] == pytest.approx(-0.010)
    assert out["passes"] is True


def test_a_delta_short_of_the_bar_fails_even_when_every_seed_improves():
    """The measured A1-XGB case: all four seeds better, mean delta inside the bar."""
    out = dec.fresh_seed_confirmation(
        [0.6467, 0.6472, 0.6488, 0.6493], [0.6506, 0.6496, 0.6512, 0.6516], 0.0012463,
    )
    assert out["mean_delta"] == pytest.approx(-0.00275)
    assert out["passes"] is False


def test_a_delta_exactly_on_the_bar_does_not_clear_it():
    out = dec.fresh_seed_confirmation([0.647], [0.650], 0.0001)
    assert (out["bar"], out["mean_delta"]) == (0.003, -0.003)
    assert out["passes"] is False


def test_the_bar_widens_with_a_noisy_seed_floor():
    assert dec.fresh_seed_confirmation([0.64], [0.65], 0.002)["bar"] == 0.004


def test_unpaired_or_empty_seed_sets_are_rejected():
    with pytest.raises(ValueError, match="at least one run"):
        dec.fresh_seed_confirmation([], [0.65], 0.001)
    with pytest.raises(ValueError, match="paired"):
        dec.fresh_seed_confirmation([0.64, 0.64], [0.65], 0.001)


def test_a_non_finite_sigma_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        dec.fresh_seed_confirmation([0.64], [0.65], float("nan"))


# --- attribution_branch -----------------------------------------------------
# Rule 3 of docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md.
# t_minus_c is a LOG-LOSS difference, so negative means the treatment is better.

SIGMA = 0.000346


def test_treatment_alone_clearing_supports_h1():
    out = dec.attribution_branch(c_ships=False, t_ships=True, t_minus_c=-0.004, sigma_seed=SIGMA)
    assert out["verdict"] == "H1 supported"
    assert out["action"] == "ship S1 with the searched config"


def test_both_clearing_with_a_real_gap_credits_both():
    out = dec.attribution_branch(c_ships=True, t_ships=True, t_minus_c=-0.002, sigma_seed=SIGMA)
    assert out["branch"] == "both clear and T - C > sigma_seed"


def test_both_clearing_within_seed_noise_is_the_architecture_null():
    for gap in (0.0, -SIGMA, 0.0001):
        out = dec.attribution_branch(c_ships=True, t_ships=True, t_minus_c=gap, sigma_seed=SIGMA)
        assert out["branch"] == "both clear and T - C <= sigma_seed"
        assert out["verdict"].startswith("H0")


def test_a_treatment_that_is_worse_never_reaches_the_both_matter_branch():
    """The literal sign of 'T - C > sigma_seed' would fire here; it must not."""
    out = dec.attribution_branch(c_ships=True, t_ships=True, t_minus_c=+0.01, sigma_seed=SIGMA)
    assert out["branch"] == "both clear and T - C <= sigma_seed"


def test_neither_clearing_is_the_measured_sp2_1_outcome():
    out = dec.attribution_branch(c_ships=False, t_ships=False, t_minus_c=0.0001, sigma_seed=SIGMA)
    assert out["branch"] == "neither clears"
    assert out["action"] == "record and revert"


def test_the_branch_the_rule_does_not_name_is_reported_as_uncovered():
    out = dec.attribution_branch(c_ships=True, t_ships=False, t_minus_c=0.004, sigma_seed=SIGMA)
    assert out["verdict"] == "not covered by the pre-registered rule"
    assert out["action"] == "stop for a human call"


# --- noise_floor.seed_label -------------------------------------------------
# XGB has no seed ensemble, so its seed is random_state, not config.seeds.


def test_torch_reports_are_labelled_by_their_ensemble():
    assert seed_label({"config": {"candidate": "torch", "seeds": "0,1,2,3,4"}}) == "0,1,2,3,4"


def test_xgb_model_seed_is_the_seed_label():
    assert seed_label({"config": {"candidate": "xgb", "seeds": None, "model_seed": 3}}) == "3"


def test_an_xgb_report_with_no_model_seed_ran_at_the_base_params_default():
    from mma.models.xgb import BASE_PARAMS

    assert seed_label({"config": {"candidate": "xgb", "seeds": None}}) == str(
        BASE_PARAMS["random_state"]
    )


def test_a_random_state_set_through_config_json_wins():
    label = seed_label({"config": {"candidate": "xgb", "seeds": None, "model_seed": None,
                                   "config": {"random_state": 7}}})
    assert label == "7"


def test_a_candidate_with_no_seed_has_no_label():
    assert seed_label({"config": {"candidate": "elo", "seeds": None}}) is None


# --- blend_check.average_predictions ----------------------------------------


def test_the_blend_is_a_row_wise_equal_weight_average():
    out = average_predictions([
        {"winner": np.array([0.2, 0.8]), "method": np.array([[0.5, 0.5]]), "round": None},
        {"winner": np.array([0.4, 0.6]), "method": np.array([[0.1, 0.9]]), "round": None},
    ])
    assert out["winner"] == pytest.approx([0.3, 0.7])
    assert out["method"] == pytest.approx(np.array([[0.3, 0.7]]))
    assert out["round"] is None
