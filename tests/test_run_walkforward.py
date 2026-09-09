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
    xgb = rwf.build_candidate("xgb", "x", None, {}, None, ("age_diff",))
    torch_candidate = rwf.build_candidate("torch", "t", (0, 1), {}, None, ("age_diff",))
    blend = rwf.build_candidate("blend", "b", (0, 1), {}, None, ("age_diff",),
                                {"blend_weight": 0.5, "blend_calibrated": True})
    assert xgb.drop_columns == ("age_diff",)
    assert torch_candidate.drop_columns == ("age_diff",)
    assert blend.drop_columns == ("age_diff",)
    assert all(m.drop_columns == ("age_diff",) for m in blend.members())


# --- --model-seed -----------------------------------------------------------
# XGBoost's stochasticity is subsample/colsample under one random_state, so a
# re-run with a different value is the XGB analogue of torch's fresh seed set
# (SP2.1 Task 4's fresh-seed confirmation). Torch is seeded with --seeds and
# elo fits nothing, so pointing the flag at either is a usage error.


def _seed_args(**overrides) -> argparse.Namespace:
    base = {"candidate": "xgb", "model_seed": None}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_no_flag_leaves_the_param_override_alone():
    assert rwf.apply_model_seed(_seed_args(), {}) == {}
    assert rwf.apply_model_seed(_seed_args(), {"max_depth": 5}) == {"max_depth": 5}


def test_model_seed_becomes_the_xgb_random_state():
    assert rwf.apply_model_seed(_seed_args(model_seed=3), {"max_depth": 5}) == {
        "max_depth": 5, "random_state": 3,
    }


def test_the_caller_s_config_is_not_mutated():
    config = {"max_depth": 5}
    rwf.apply_model_seed(_seed_args(model_seed=3), config)
    assert config == {"max_depth": 5}


@pytest.mark.parametrize("candidate", ["torch", "elo"])
def test_model_seed_is_xgb_only(candidate):
    with pytest.raises(SystemExit, match="xgb candidate only"):
        rwf.apply_model_seed(_seed_args(candidate=candidate, model_seed=1), {})


def test_two_sources_for_random_state_is_a_usage_error():
    with pytest.raises(SystemExit, match="both set random_state"):
        rwf.apply_model_seed(_seed_args(model_seed=1), {"random_state": 7})


def test_random_state_reaches_the_fitted_estimator():
    """The property the flag exists for, on the real xgboost builder."""
    from mma.models.xgb import _classifier

    config = rwf.apply_model_seed(_seed_args(model_seed=3), {})
    assert _classifier("binary:logistic", config, 10).get_params()["random_state"] == 3
    assert _classifier("binary:logistic", {}, 10).get_params()["random_state"] == 0


# --- --seeds / --blend-weight (SP2.2) ---------------------------------------
# torch and blend run a 5-seed ensemble by default. XGBoost does not: SP2.2
# gave it seed ensembling for the blend's sake, and defaulting it on would
# silently change what every plain `--candidate xgb` run means.


def _ens_args(**overrides) -> argparse.Namespace:
    base = {"candidate": "torch", "seeds": None, "model_seed": None,
            "blend_weight": None, "no_blend_calibration": False,
            "blend_calibrator": "temperature", "hazard_calibrate": False,
            "blend_joint_cells": False}
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("candidate", ["torch", "blend"])
def test_seeded_candidates_default_to_the_five_seed_ensemble(candidate):
    assert rwf.resolve_seeds(_ens_args(candidate=candidate)) == (0, 1, 2, 3, 4)


def test_xgb_stays_a_single_fit_unless_seeds_are_named():
    assert rwf.resolve_seeds(_ens_args(candidate="xgb")) is None
    assert rwf.resolve_seeds(_ens_args(candidate="xgb", seeds="0")) == (0,)
    assert rwf.resolve_seeds(_ens_args(candidate="elo")) is None


def test_seeds_are_parsed_stripped_and_deduplicated_in_order():
    assert rwf.resolve_seeds(_ens_args(seeds=" 5, 6 ,5,7,")) == (5, 6, 7)


def test_seeds_and_model_seed_together_are_a_usage_error():
    with pytest.raises(SystemExit, match="both set the xgb random_state"):
        rwf.resolve_seeds(_ens_args(candidate="xgb", seeds="0,1", model_seed=2))


def test_seeding_the_elo_floor_is_a_usage_error():
    with pytest.raises(SystemExit, match="fitted candidates only"):
        rwf.resolve_seeds(_ens_args(candidate="elo", seeds="0,1"))


def test_a_seeds_flag_that_names_nothing_is_a_usage_error():
    with pytest.raises(SystemExit, match="names no seeds"):
        rwf.resolve_seeds(_ens_args(seeds=" , "))


def test_blend_weight_defaults_to_the_pre_registered_half():
    assert rwf.resolve_blend(_ens_args(candidate="blend")) == {
        "blend_weight": 0.5, "blend_calibrated": True,
    }
    assert rwf.resolve_blend(_ens_args(candidate="blend", blend_weight=0.3,
                                       no_blend_calibration=True)) == {
        "blend_weight": 0.3, "blend_calibrated": False,
    }


def test_blend_flags_are_blend_only_and_bounded():
    assert rwf.resolve_blend(_ens_args(candidate="torch")) is None
    with pytest.raises(SystemExit, match="blend candidate only"):
        rwf.resolve_blend(_ens_args(candidate="xgb", blend_weight=0.5))
    with pytest.raises(SystemExit, match="blend candidate only"):
        rwf.resolve_blend(_ens_args(candidate="torch", no_blend_calibration=True))
    with pytest.raises(SystemExit, match=r"must be in \[0, 1\]"):
        rwf.resolve_blend(_ens_args(candidate="blend", blend_weight=1.5))


def test_the_blend_rejects_flags_it_cannot_honour():
    blend = {"blend_weight": 0.5, "blend_calibrated": True}
    with pytest.raises(SystemExit, match="fixed-budget mode does not apply"):
        rwf.build_candidate("blend", "b", (0,), {}, {"fixed_epochs": 4}, (), blend)
    with pytest.raises(SystemExit, match="two members"):
        rwf.build_candidate("blend", "b", (0,), {"lr": 1e-3}, None, (), blend)


def test_the_blend_hands_both_members_the_same_seeds():
    blend = rwf.build_candidate("blend", "b", (0, 1, 2), {}, None, (),
                                {"blend_weight": 0.3, "blend_calibrated": False})
    xgb, torch_member = blend.members()
    assert blend.seeds == (0, 1, 2)
    assert xgb.seeds == (0, 1, 2) and torch_member.seeds == (0, 1, 2)
    assert blend.weight == 0.3 and blend.calibrate is False


def test_the_default_calibrator_leaves_the_committed_blend_config_shape_alone():
    """`blend_calibrator` appears in the run config only when it is NOT the
    pre-registered temperature, so a default blend report is shaped exactly
    like the committed B0/B1 ones."""
    assert "blend_calibrator" not in rwf.resolve_blend(_ens_args(candidate="blend"))
    assert rwf.resolve_blend(_ens_args(candidate="blend", blend_calibrator="isotonic")) == {
        "blend_weight": 0.5, "blend_calibrated": True, "blend_calibrator": "isotonic",
    }


def test_an_unknown_or_contradictory_calibrator_is_a_usage_error():
    with pytest.raises(SystemExit, match="--blend-calibrator must be one of"):
        rwf.resolve_blend(_ens_args(candidate="blend", blend_calibrator="platt"))
    with pytest.raises(SystemExit, match="cannot be combined"):
        rwf.resolve_blend(_ens_args(candidate="blend", blend_calibrator="isotonic",
                                    no_blend_calibration=True))


def test_build_candidate_passes_the_calibrator_through_to_the_blend():
    default = rwf.build_candidate("blend", "b", (0,), {}, None, (),
                                  {"blend_weight": 0.5, "blend_calibrated": True})
    assert default.calibrator == "temperature"
    iso = rwf.build_candidate("blend", "b", (0,), {}, None, (),
                              {"blend_weight": 0.5, "blend_calibrated": True,
                               "blend_calibrator": "isotonic"})
    assert iso.calibrator == "isotonic"


# --- prediction_dump --------------------------------------------------------


def _dump_frame():
    return pd.DataFrame({
        "y_winner": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "date": pd.to_datetime(["2018-01-01"] * 3 + ["2019-01-01"] * 3),
    })


def test_prediction_dump_is_row_aligned_with_the_pooled_metrics():
    """The dump's row order must be `pool`'s -- the fold concatenation order --
    so an ECE recomputed from it is the report's ECE and not a lookalike over
    a differently ordered row set."""
    frame = _dump_frame()
    m18 = np.array([True, True, True, False, False, False])
    m19 = ~m18
    fold_results = [
        (2018, m18, {"winner": np.array([0.7, 0.2, 0.9]), "method": None, "round": None}, {}),
        (2019, m19, {"winner": np.array([0.1, 0.6, 0.55]), "method": None, "round": None}, {}),
    ]
    dump = rwf.prediction_dump("d", frame, fold_results)
    assert dump == {
        "name": "d", "n": 6,
        "fold_year": [2018, 2018, 2018, 2019, 2019, 2019],
        "y_winner": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "p_winner": [0.7, 0.2, 0.9, 0.1, 0.6, 0.55],
    }


def test_prediction_dump_follows_the_fold_order_it_is_given_not_the_table_order():
    frame = _dump_frame()
    m18 = np.array([True, True, True, False, False, False])
    fold_results = [
        (2019, ~m18, {"winner": np.array([0.1, 0.6, 0.55]), "method": None, "round": None}, {}),
        (2018, m18, {"winner": np.array([0.7, 0.2, 0.9]), "method": None, "round": None}, {}),
    ]
    dump = rwf.prediction_dump("d", frame, fold_results)
    assert dump["fold_year"] == [2019, 2019, 2019, 2018, 2018, 2018]
    assert dump["y_winner"] == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert dump["p_winner"] == [0.1, 0.6, 0.55, 0.7, 0.2, 0.9]


def test_prediction_dump_is_json_round_trippable_at_full_precision():
    frame = _dump_frame().iloc[:2]
    mask = np.array([True, True])
    p = np.array([0.6543210987654321, 0.1234567890123456])
    dump = rwf.prediction_dump("d", frame, [(2018, mask, {"winner": p, "method": None,
                                                          "round": None}, {})])
    assert json.loads(json.dumps(dump))["p_winner"] == [float(v) for v in p]


# --- the hazard (simulator) candidate ---------------------------------------


def test_the_hazard_candidate_is_seed_ensembled_by_default():
    assert rwf.resolve_seeds(_ens_args(candidate="hazard")) == (0, 1, 2, 3, 4)


def test_build_candidate_hands_the_hazard_candidate_its_fights_table():
    fights = pd.DataFrame({"fight_id": ["f0"], "finish_round": [1], "scheduled_rounds": [3]})
    cand = rwf.build_candidate("hazard", "h", (0, 1), {"max_depth": 3}, None,
                               ("age_diff",), None, fights)
    assert cand.seeds == (0, 1) and cand.drop_columns == ("age_diff",)
    assert cand.params == {"max_depth": 3}
    assert cand.fights is fights
    # the pre-registered simulation parameters, unchanged by the CLI
    assert cand.n_runs == 10_000 and cand.alpha == 1.0


def test_the_hazard_candidate_without_a_fights_table_is_a_usage_error():
    with pytest.raises(SystemExit, match="fights table"):
        rwf.build_candidate("hazard", "h", (0,), {}, None)


def test_fixed_budget_mode_does_not_apply_to_the_hazard_candidate():
    fights = pd.DataFrame({"fight_id": ["f0"]})
    with pytest.raises(SystemExit, match="hazard"):
        rwf.build_candidate("hazard", "h", (0,), {}, {"fixed_rounds": 50}, (), None, fights)


def test_hazard_calibration_is_off_unless_the_flag_asks_for_it():
    fights = pd.DataFrame({"fight_id": ["f0"], "finish_round": [1], "scheduled_rounds": [3]})
    plain = rwf.build_candidate("hazard", "h", (0,), {}, None, (), None, fights,
                                rwf.resolve_hazard(_ens_args(candidate="hazard",
                                                             hazard_calibrate=False)))
    assert plain.calibrate is False
    calibrated = rwf.build_candidate("hazard", "h", (0,), {}, None, (), None, fights,
                                     rwf.resolve_hazard(_ens_args(candidate="hazard",
                                                                  hazard_calibrate=True)))
    assert calibrated.calibrate is True


def test_hazard_calibration_flag_on_another_candidate_is_a_usage_error():
    assert rwf.resolve_hazard(_ens_args(candidate="blend", hazard_calibrate=False)) is None
    with pytest.raises(SystemExit, match="hazard candidate only"):
        rwf.resolve_hazard(_ens_args(candidate="blend", hazard_calibrate=True))


# --- the hybrid (blend winner x simulator conditional) candidate -------------


def test_the_hybrid_candidate_is_seed_ensembled_by_default():
    assert rwf.resolve_seeds(_ens_args(candidate="hybrid")) == (0, 1, 2, 3, 4)


def test_build_candidate_wires_the_hybrid_s_two_members():
    fights = pd.DataFrame({"fight_id": ["f0"], "finish_round": [1], "scheduled_rounds": [3]})
    cand = rwf.build_candidate("hybrid", "h", (0, 1), {}, None, ("age_diff",), None, fights)
    blend, hazard = cand.members()
    assert cand.fights is fights and cand.drop_columns == ("age_diff",)
    # the incumbent blend's pre-registered form, and the simulator's
    # pre-registered simulation parameters, unchanged by the CLI
    assert blend.weight == 0.5 and blend.calibrate is True
    assert hazard.n_runs == 10_000 and hazard.alpha == 1.0 and hazard.calibrate is False


def test_the_hybrid_candidate_without_a_fights_table_is_a_usage_error():
    with pytest.raises(SystemExit, match="fights table"):
        rwf.build_candidate("hybrid", "h", (0,), {}, None)


def test_the_hybrid_candidate_rejects_an_ambiguous_config_and_a_fixed_budget():
    fights = pd.DataFrame({"fight_id": ["f0"]})
    with pytest.raises(SystemExit, match="hybrid"):
        rwf.build_candidate("hybrid", "h", (0,), {"max_depth": 3}, None, (), None, fights)
    with pytest.raises(SystemExit, match="hybrid"):
        rwf.build_candidate("hybrid", "h", (0,), {}, {"fixed_rounds": 50}, (), None, fights)


def test_the_blend_joint_cell_control_is_off_by_default_and_blend_only():
    off = rwf.resolve_blend(_ens_args(candidate="blend"))
    assert "blend_joint_cells" not in off  # committed blend reports keep their config shape
    on = rwf.resolve_blend(_ens_args(candidate="blend", blend_joint_cells=True))
    assert on["blend_joint_cells"] is True
    assert rwf.build_candidate("blend", "b", (0,), {}, None, (), on).emit_joint_cells is True
    assert rwf.build_candidate("blend", "b", (0,), {}, None, (), off).emit_joint_cells is False
    with pytest.raises(SystemExit, match="blend candidate only"):
        rwf.resolve_blend(_ens_args(candidate="torch", blend_joint_cells=True))
