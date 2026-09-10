"""Periodic re-validation of the DEPLOYED recipe (scripts/revalidate_recipe.py).

This replaces the promotion gate `scripts/roll_window.py` used to describe.
There is no candidate-versus-incumbent choice to make any more -- the weekly
Action already refits the deployed recipe through the latest event -- so the
open question is not "should something else ship?" but "does what already
ships still clear the bars it was justified by, on the table it now trains
on?".

What is worth pinning here is everything that could make the artifact say the
recipe is still justified when it is not:

* the run configurations are READ OUT of the committed reports' own `config`
  blocks, so a re-validation run cannot quietly be a different experiment
  wearing the deployed recipe's name;
* every threshold is READ OUT of the committed decision artifacts, so a bar
  edited without touching this script changes what the script applies -- and a
  bar this script can no longer find is a loud failure, not a fallback to a
  retyped constant;
* the refit B run's budget comes from the FRESHLY re-run A, not the committed
  one, because a budget derived on a smaller table is a budget for a different
  table;
* a bar that is no longer met exits non-zero, because that is a finding a
  human has to act on rather than a line in a log.
"""
from __future__ import annotations

import json

import pytest

import scripts.revalidate_recipe as rr


# --- the deployed configuration is read, not retyped -------------------------

HYBRID_CONFIG = {
    "candidate": "hybrid",
    "seeds": "0,1,2,3,4",
    "model_seed": None,
    "config": {},
    "train_start": None,
    "half_life": None,
    "fixed_budget_from": None,
    "budget": {},
    "drop_columns": ["external_missing", "same_country", "notice_unknown",
                     "home_country_a", "home_country_b"],
    "n_feature_rows": 11238,
    "features_max_date": "2026-08-08",
    "runtime_sec": 390.3,
}
BLEND_CELLS_CONFIG = {
    **HYBRID_CONFIG,
    "candidate": "blend",
    "blend_weight": 0.5,
    "blend_calibrated": True,
    "blend_joint_cells": True,
}


def test_run_args_round_trip_the_hybrid_config():
    args = rr.run_args(HYBRID_CONFIG, name="x", out_dir="/tmp/o")
    assert "--candidate" in args and args[args.index("--candidate") + 1] == "hybrid"
    assert args[args.index("--seeds") + 1] == "0,1,2,3,4"
    assert args[args.index("--drop-columns") + 1] == (
        "external_missing,same_country,notice_unknown,home_country_a,home_country_b")
    assert args[args.index("--name") + 1] == "x"
    # nothing the deployed hybrid does not set
    assert "--blend-weight" not in args
    assert "--fixed-budget-from" not in args
    assert "--config-json" not in args
    assert "--model-seed" not in args


def test_run_args_carry_the_blends_own_switches():
    args = rr.run_args(BLEND_CELLS_CONFIG, name="b", out_dir="/tmp/o")
    assert args[args.index("--blend-weight") + 1] == "0.5"
    assert "--blend-joint-cells" in args
    assert "--no-blend-calibration" not in args  # calibration is on


def test_run_args_uncalibrated_blend_passes_the_flag():
    args = rr.run_args({**BLEND_CELLS_CONFIG, "blend_calibrated": False},
                       name="b", out_dir="/tmp/o")
    assert "--no-blend-calibration" in args


def test_run_args_budget_from_points_at_the_fresh_A_not_the_committed_one():
    """A refit budget derived on a smaller table is a budget for a different
    table -- the whole reason this comparison is re-run rather than re-read."""
    config = {**HYBRID_CONFIG, "candidate": "torch",
              "fixed_budget_from": "models/walkforward/torch_a1_combined.json"}
    args = rr.run_args(config, name="torch_B", out_dir="/tmp/o",
                       budget_from="/tmp/o/torch_A.json")
    assert args[args.index("--fixed-budget-from") + 1] == "/tmp/o/torch_A.json"
    assert "models/walkforward/torch_a1_combined.json" not in args


def test_run_args_refuses_a_budget_config_with_no_fresh_source():
    """A committed --fixed-budget-from that this script does not redirect would
    silently re-validate against the OLD table's budget."""
    config = {**HYBRID_CONFIG, "candidate": "torch",
              "fixed_budget_from": "models/walkforward/torch_a1_combined.json"}
    with pytest.raises(SystemExit, match="fixed_budget_from"):
        rr.run_args(config, name="torch_B", out_dir="/tmp/o")


def test_every_deployed_run_names_a_committed_report():
    for run in rr.RUNS:
        assert run.source.exists(), run.source


def test_the_deployed_runs_agree_on_the_drop_columns():
    """One recipe, one model matrix. Two members judged on different held-out
    columns would not be the deployed configuration at all."""
    seen = {run.key: tuple(rr.load(run.source)["config"]["drop_columns"])
            for run in rr.RUNS}
    assert len(set(seen.values())) == 1, seen


def test_deployed_configuration_reads_blocks_from_the_sidecar():
    config = rr.deployed_configuration()
    assert config["feature_blocks"] == list(rr.table_blocks())
    assert config["drop_columns"] == HYBRID_CONFIG["drop_columns"]
    assert len(config["deployed_hash"]) == 12


# --- the bars are read out of the committed decision artifacts ---------------

def test_sp3_thresholds_come_from_the_sp3_decision_artifact():
    thresholds = rr.sp3_thresholds(json.loads(rr.SP3_DECISION.read_text()))
    assert thresholds["joint_bar"] == 0.01
    assert thresholds["sigma_seed"] == pytest.approx(0.000346, abs=5e-7)


def test_sp3_thresholds_fail_loudly_when_the_artifact_has_no_bar():
    with pytest.raises(SystemExit, match="joint_bar"):
        rr.sp3_thresholds({"bar": {}})


def test_refit_rule_comes_from_the_refit_decision_artifact():
    rule = rr.refit_rule(json.loads(rr.REFIT_DECISION.read_text()))
    assert "sigma_seed" in rule["text"]
    assert rule["sigma_seed"] == pytest.approx(0.0003464101615137373)
    assert rule["gated_member"] == "torch"


def test_refit_rule_fails_loudly_without_a_rule():
    with pytest.raises(SystemExit, match="rule"):
        rr.refit_rule({"torch": {}})


def test_ece_gate_threshold_is_the_amended_mean_plus_its_own_tolerance():
    gate = rr.ece_gate(json.loads(rr.SP2_2_DECISION.read_text()))
    assert gate["threshold"] == pytest.approx(0.011633 + 0.005608327142146162, abs=1e-6)
    assert "three disjoint seed sets" in gate["form"]


def test_ece_gate_fails_loudly_when_the_amendment_is_gone():
    with pytest.raises(SystemExit, match="ece_gate"):
        rr.ece_gate({"ece_gate": {}})


def test_the_sigma_the_bars_apply_is_the_measured_noise_floor():
    """Two artifacts quote sigma_seed; if they ever disagree with the floor
    that was measured, the bars are applying a tolerance nobody measured."""
    floor = json.loads(rr.NOISE_FLOOR.read_text())["sigma_seed"]
    rr.check_sigma_agreement(
        rr.sp3_thresholds(json.loads(rr.SP3_DECISION.read_text()))["sigma_seed"],
        rr.refit_rule(json.loads(rr.REFIT_DECISION.read_text()))["sigma_seed"],
        floor,
    )
    with pytest.raises(SystemExit, match="sigma"):
        rr.check_sigma_agreement(0.01, floor, floor)


# --- the bars, applied ------------------------------------------------------

def _report(name, *, winner, joint, ece=0.012, n=4804):
    folds = {str(y): {"winner_log_loss": winner, "joint_log_loss": joint, "ece": ece}
             for y in range(2018, 2026)}
    return {
        "name": name,
        "config": dict(HYBRID_CONFIG),
        "fold_years": list(range(2018, 2026)),
        "folds": folds,
        "pooled": {"n": n, "winner_log_loss": winner, "joint_log_loss": joint,
                   "ece": ece, "accuracy": 0.62, "brier": 0.226},
        "slices": {}, "fit_info": {},
    }


SP3 = {"joint_bar": 0.01, "sigma_seed": 0.000346}


def test_sp3_bar_still_clears_when_the_joint_gain_is_wide():
    check = rr.check_sp3_bar(_report("h", winner=0.6437, joint=2.1432),
                             _report("i", winner=0.6437, joint=2.2013), SP3)
    assert check["still_clears"] is True
    assert check["joint_delta"] == pytest.approx(-0.0581)


def test_sp3_bar_no_longer_met_when_the_joint_gain_collapses():
    check = rr.check_sp3_bar(_report("h", winner=0.6437, joint=2.1950),
                             _report("i", winner=0.6437, joint=2.2013), SP3)
    assert check["still_clears"] is False


def test_sp3_bar_no_longer_met_when_the_winner_marginal_regresses():
    """The hybrid takes the blend's winner array verbatim, so a nonzero winner
    delta is itself the finding."""
    check = rr.check_sp3_bar(_report("h", winner=0.6500, joint=2.1432),
                             _report("i", winner=0.6437, joint=2.2013), SP3)
    assert check["still_clears"] is False
    assert check["winner_clause_satisfied"] is False


def test_refit_check_passes_when_B_is_not_worse_by_more_than_sigma():
    rule = {"text": "...", "sigma_seed": 0.000346, "gated_member": "torch"}
    check = rr.check_refit_rule(_report("A", winner=0.6475, joint=2.31),
                                _report("B", winner=0.6473, joint=2.31), rule)
    assert check["still_clears"] is True
    assert check["delta_B_minus_A"] == pytest.approx(-0.0002)


def test_refit_check_fails_when_B_loses_to_A_by_more_than_sigma():
    rule = {"text": "...", "sigma_seed": 0.000346, "gated_member": "torch"}
    check = rr.check_refit_rule(_report("A", winner=0.6475, joint=2.31),
                                _report("B", winner=0.6500, joint=2.31), rule)
    assert check["still_clears"] is False


def test_ece_check_uses_the_recorded_threshold():
    gate = {"form": "f", "threshold": 0.017241, "incumbent_mean": 0.011633,
            "tolerance_2sigma": 0.005608}
    assert rr.check_ece_gate(_report("b", winner=0.64, joint=2.2, ece=0.0124),
                             gate)["still_clears"] is True
    assert rr.check_ece_gate(_report("b", winner=0.64, joint=2.2, ece=0.0190),
                             gate)["still_clears"] is False


# --- the verdict and the exit code ------------------------------------------

def test_verdict_is_still_justified_when_every_gated_bar_clears():
    bars = {
        "sp3_joint_bar": {"gated": True, "still_clears": True},
        "refit_rule_torch": {"gated": True, "still_clears": True},
        "refit_rule_xgb": {"gated": False, "still_clears": False},
        "blend_ece_gate": {"gated": True, "still_clears": True},
    }
    out = rr.verdict(bars)
    assert out["still_justified"] is True
    assert out["bars_no_longer_met"] == []
    assert rr.exit_code(out) == 0
    # the ungated xgb read is surfaced, not swallowed
    assert "refit_rule_xgb" in out["ungated_bars_not_met"]


def test_verdict_names_every_gated_bar_that_is_no_longer_met():
    bars = {
        "sp3_joint_bar": {"gated": True, "still_clears": False},
        "refit_rule_torch": {"gated": True, "still_clears": True},
        "blend_ece_gate": {"gated": True, "still_clears": False},
    }
    out = rr.verdict(bars)
    assert out["still_justified"] is False
    assert out["bars_no_longer_met"] == ["blend_ece_gate", "sp3_joint_bar"]
    assert rr.exit_code(out) == 1
    assert "no longer" in out["outcome"].lower()


# --- staleness: the cheap weekly read ---------------------------------------

def test_staleness_measures_the_gap_between_the_harness_and_the_training_data():
    report = rr.staleness(
        table_max_date="2026-09-05", n_table_rows=11290,
        members={"torch": {"train_through": "2026-09-05",
                           "harness_features_max_date": "2026-08-08"}},
    )
    assert report["stale"] is True
    assert report["members"]["torch"]["days_behind"] == 28
    assert "revalidate_recipe.py" in report["what_to_do"]


def test_staleness_is_quiet_when_the_harness_has_seen_the_training_data():
    report = rr.staleness(
        table_max_date="2026-08-08", n_table_rows=11238,
        members={"torch": {"train_through": "2026-08-08",
                           "harness_features_max_date": "2026-08-08"}},
    )
    assert report["stale"] is False
    assert report["members"]["torch"]["days_behind"] == 0


def test_staleness_needs_no_model_and_no_fit():
    """The weekly Action runs this every week; it must read files and nothing
    else, or it is not a cheap step."""
    report = rr.staleness_from_disk()
    assert report["table_max_date"] == "2026-09-05"
    assert set(report["members"]) == {"torch", "xgb", "hazard"}


def test_staleness_warns_on_stderr_when_the_evidence_is_older_than_the_data():
    """The weekly Action's step is report-only; the warning is how a stale
    justification reaches a reader, the way check_snapshot_coverage.py warns
    about the other measurement that ages rather than breaks."""
    report = rr.staleness(
        table_max_date="2026-09-05", n_table_rows=11290,
        members={"torch": {"train_through": "2026-09-05",
                           "harness_features_max_date": "2026-08-08"},
                 "xgb": {"train_through": "2026-09-05",
                         "harness_features_max_date": "2026-08-08"}},
    )
    warning = rr.staleness_warning(report)
    assert "28 day(s)" in warning
    assert "revalidate_recipe.py" in warning


def test_staleness_is_silent_when_the_evidence_covers_the_data():
    report = rr.staleness(
        table_max_date="2026-08-08", n_table_rows=11238,
        members={"torch": {"train_through": "2026-08-08",
                           "harness_features_max_date": "2026-08-08"}},
    )
    assert rr.staleness_warning(report) is None


# --- then vs now: has the margin moved? -------------------------------------

def test_recorded_margins_come_from_the_committed_decision_artifacts():
    """A re-validation that only reports today's number cannot say whether a
    margin has moved. The 'then' side is read out of the artifacts, so it is
    the number the decision was actually taken on."""
    then = rr.recorded_margins()
    assert then["sp3_joint_delta"] == pytest.approx(-0.0581, abs=1e-6)
    assert then["sp3_winner_delta"] == 0.0
    assert then["refit_torch_delta"] == pytest.approx(-0.0002)
    assert then["refit_xgb_delta"] == pytest.approx(-0.0007)
    assert then["blend_pooled_ece"] == pytest.approx(0.0124)


def test_margin_movement_lines_the_two_up_and_names_the_direction():
    then = {"sp3_joint_delta": -0.0581, "sp3_winner_delta": 0.0,
            "refit_torch_delta": -0.0002, "refit_xgb_delta": -0.0007,
            "blend_pooled_ece": 0.0124}
    bars = {
        "sp3_joint_bar": {"joint_delta": -0.0580, "winner_delta": 0.0,
                          "joint_margin_past_the_bar": 0.048},
        "refit_rule_torch": {"delta_B_minus_A": -0.0002},
        "refit_rule_xgb": {"delta_B_minus_A": -0.0008},
        "blend_ece_gate": {"pooled_ece": 0.0146, "headroom": 0.002641,
                           "thresholds": {"threshold": 0.017241}},
    }
    moved = rr.margin_movement(then, bars)
    assert moved["sp3_joint_delta"]["then"] == -0.0581
    assert moved["sp3_joint_delta"]["now"] == -0.0580
    # ECE moved toward the threshold; the block has to make that visible
    assert moved["blend_pooled_ece"]["change"] == pytest.approx(0.0022, abs=1e-6)
    assert moved["blend_pooled_ece"]["closer_to_the_bar"] is True
    assert moved["refit_xgb_delta"]["closer_to_the_bar"] is False


def test_margin_movement_reads_a_worse_sp3_joint_as_closer_to_the_bar():
    """The joint delta is negative-is-better, so 'closer to the bar' is the
    delta rising -- the opposite direction from the ECE's."""
    then = {"sp3_joint_delta": -0.0581, "sp3_winner_delta": 0.0,
            "refit_torch_delta": -0.0002, "refit_xgb_delta": -0.0007,
            "blend_pooled_ece": 0.0124}
    bars = {
        "sp3_joint_bar": {"joint_delta": -0.0300, "winner_delta": 0.0,
                          "joint_margin_past_the_bar": 0.020},
        "refit_rule_torch": {"delta_B_minus_A": -0.0002},
        "refit_rule_xgb": {"delta_B_minus_A": -0.0007},
        "blend_ece_gate": {"pooled_ece": 0.0124, "headroom": 0.0048,
                           "thresholds": {"threshold": 0.017241}},
    }
    moved = rr.margin_movement(then, bars)
    assert moved["sp3_joint_delta"]["closer_to_the_bar"] is True
    assert moved["blend_pooled_ece"]["closer_to_the_bar"] is False


# --- rebuilding the artifact without re-fitting ------------------------------

def test_reusing_reports_refuses_a_report_from_a_different_table(tmp_path):
    """--from-reports must not quietly assemble a verdict out of runs made on
    two different tables, or on a table that is no longer the current one."""
    for run in rr.RUNS:
        path = tmp_path / f"revalidation_{run.key}.json"
        config = dict(rr.load(run.source)["config"])
        config["features_max_date"] = "2026-08-08"
        config["n_feature_rows"] = 11238
        path.write_text(json.dumps({"name": run.key, "config": config,
                                    "folds": {}, "pooled": {}, "slices": {},
                                    "fit_info": {}}))
    with pytest.raises(SystemExit, match="2026-08-08"):
        rr.reuse_reports(tmp_path, table_max_date="2026-09-05", n_feature_rows=11290)


def test_reusing_reports_refuses_a_missing_run(tmp_path):
    with pytest.raises(SystemExit, match="revalidation_hybrid"):
        rr.reuse_reports(tmp_path, table_max_date="2026-09-05", n_feature_rows=11290)


# --- a re-validation answers the question the staleness read asks ------------
#
# The warning asks one thing: has the recipe's justification been re-measured
# on the data the deployed models train on? Each member's own harness report is
# older than its training cutoff and always will be -- that report ran on the
# table it ran on -- so the member gap alone cannot answer it. This script's own
# artifact can, and these pin the four ways it can answer.

PASSED_REVALIDATION = {
    "date": "2026-09-09",
    "table": {"features_max_date": "2026-09-05", "n_feature_rows": 11290},
    "verdict": {"still_justified": True, "bars_no_longer_met": []},
}
BEHIND = {"train_through": "2026-09-05", "harness_features_max_date": "2026-08-08"}


def test_revalidation_cover_reads_the_committed_artifact():
    cover = rr.revalidation_cover(PASSED_REVALIDATION)
    assert cover["date"] == "2026-09-09"
    assert cover["features_max_date"] == "2026-09-05"
    assert cover["still_justified"] is True
    assert cover["bars_no_longer_met"] == []
    assert cover["read_from"].endswith("recipe_revalidation.json")


def test_revalidation_cover_is_absent_when_nothing_has_been_revalidated():
    assert rr.revalidation_cover(None) is None


# case 1: no re-validation artifact at all -- the warning is the whole point
def test_with_no_revalidation_the_gap_is_actionable_and_warns():
    report = rr.staleness(
        table_max_date="2026-09-05", n_table_rows=11290,
        members={"torch": dict(BEHIND)},
    )
    assert report["members"]["torch"]["harness_stale"] is True
    assert report["members"]["torch"]["revalidated"] is False
    assert report["stale"] is True
    assert report["revalidation"] is None
    warning = rr.staleness_warning(report)
    assert "28 day(s)" in warning
    assert "has not been re-measured" in warning
    assert "No re-validation artifact" in warning


# case 2: a passing re-validation that reaches the training cutoff -- clears
def test_a_passing_revalidation_over_the_training_data_clears_the_flag():
    """A passing re-validation on a table that reaches the training cutoff IS
    the measurement the warning asks for. The member's own harness report is
    still older, and stays honestly recorded as such -- rewriting
    `harness_features_max_date` would be a lie about which table it ran on."""
    report = rr.staleness(
        table_max_date="2026-09-05", n_table_rows=11290,
        members={"torch": dict(BEHIND)}, revalidation=PASSED_REVALIDATION,
    )
    assert report["members"]["torch"]["harness_stale"] is True
    assert report["members"]["torch"]["days_behind"] == 28
    assert report["members"]["torch"]["harness_features_max_date"] == "2026-08-08"
    assert report["members"]["torch"]["revalidated"] is True
    assert report["stale"] is False
    assert "2026-09-05" in report["what_it_means"]
    assert rr.staleness_warning(report) is None


# case 3: a passing re-validation that predates the training cutoff -- does not
def test_a_passing_revalidation_on_an_older_table_does_not_clear_the_flag():
    """It measured the recipe on a table the models have since trained past, so
    it does not answer the question for the data they train on now."""
    older = {**PASSED_REVALIDATION,
             "table": {"features_max_date": "2026-08-20", "n_feature_rows": 11260}}
    report = rr.staleness(
        table_max_date="2026-09-05", n_table_rows=11290,
        members={"torch": dict(BEHIND)}, revalidation=older,
    )
    assert report["members"]["torch"]["revalidated"] is False
    assert report["stale"] is True
    warning = rr.staleness_warning(report)
    # check_snapshot_coverage.py's convention: name the standing decision, the
    # table it was taken on, and how far things have drifted since.
    assert "2026-08-20" in warning
    assert "still cleared" in warning
    assert "16 day(s)" in warning and "30 row(s)" in warning


# case 4: a re-validation that found a bar no longer met -- never clears, louder
def test_a_failing_revalidation_never_clears_the_flag_and_warns_louder():
    """Worse news than an unmeasured recipe, so it must not read as coverage --
    and it must reach the weekly reader in its own right."""
    failed = {**PASSED_REVALIDATION,
              "verdict": {"still_justified": False,
                          "bars_no_longer_met": ["blend_ece_gate", "sp3_joint_bar"]}}
    report = rr.staleness(
        table_max_date="2026-09-05", n_table_rows=11290,
        members={"torch": dict(BEHIND)}, revalidation=failed,
    )
    assert report["members"]["torch"]["revalidated"] is False
    assert report["stale"] is True
    warning = rr.staleness_warning(report)
    assert "blend_ece_gate" in warning and "sp3_joint_bar" in warning
    assert "no longer met" in warning
    assert "models/walkforward/recipe_revalidation.json" in warning
    # and it must not also claim the data was never re-measured -- it was, and
    # it failed. That claim would be this false positive's mirror image.
    assert "has not been re-measured" not in warning
    assert "HAS been re-measured" in warning


def test_drift_since_the_revalidation_is_measured_against_the_current_table():
    report = rr.staleness(
        table_max_date="2026-09-19", n_table_rows=11310,
        members={"torch": dict(BEHIND)}, revalidation=PASSED_REVALIDATION,
    )
    assert report["drift_since_revalidation"] == {"days": 14, "rows": 20}


def test_the_committed_tree_is_revalidated_so_the_weekly_read_is_quiet():
    """The live false positive this fixed: the committed re-validation ran on
    the 2026-09-05 table with every bar clearing, which is exactly the data the
    deployed models train on."""
    report = rr.staleness_from_disk()
    assert report["revalidation"]["still_justified"] is True
    assert all(m["harness_stale"] for m in report["members"].values())
    assert all(m["revalidated"] for m in report["members"].values())
    assert report["stale"] is False
    assert rr.staleness_warning(report) is None
