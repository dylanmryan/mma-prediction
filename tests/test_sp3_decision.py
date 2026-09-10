"""Pure helpers behind SP3's decision artifact (scripts/sp3_decision.py).

The script reads committed reports and applies the pre-registered bar. What is
worth pinning is:

* that the bar it quotes is the plan file's own text, and that the two NUMBERS
  it applies are read out of that text rather than retyped beside it -- a
  constant next to a quote is exactly how an artifact comes to state one bar
  and apply another;
* the arithmetic of the bar itself, whose two clauses are deliberately
  asymmetric (the joint clause is strict, the winner clause is a tolerance on
  a regression), because getting that backwards would ship on a tie;
* the element-wise winner check, which is the difference between verifying the
  hybrid's winner clause and assuming it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.sp3_decision as d


PLAN_STUB = """# A plan

## Locked evaluation rules

Inherited from somewhere and unchanged by this plan:

- **Incumbent:** the deployed blend. Winner marginal pooled **0.6437** (harness form).
- **Simulator ships** iff joint-outcome log-loss beats the composed baseline by
  **more than 0.01** *and* its winner marginal is not worse than the incumbent's
  by more than σ_seed (0.000346).
- **Fresh-seed confirmation is mandatory** before anything ships (seeds 5-9).
- Negative results are deliverables.

---

## Scoring the simulator fairly

- This bullet is under a different heading and must not be quoted as a rule.
"""


# --- quoted_rules -----------------------------------------------------------


def test_quoted_rules_takes_the_rules_from_the_plan_verbatim():
    rules = d.quoted_rules(PLAN_STUB)
    assert len(rules) == 4
    assert rules[0].startswith("- **Incumbent:**")
    assert rules[3] == "- Negative results are deliverables."
    # Markdown emphasis survives untouched -- a paraphrase is not a quote.
    assert "**more than 0.01**" in rules[1]


def test_quoted_rules_folds_a_wrapped_bullet_into_one_rule():
    """The bar spans three source lines; quoted as three fragments it would
    parse as three rules and `joint_bar_from` would see a bar with no number."""
    rules = d.quoted_rules(PLAN_STUB)
    bar = rules[1]
    assert "**more than 0.01**" in bar and "σ_seed (0.000346)" in bar
    assert "\n" not in bar


def test_quoted_rules_stops_at_the_end_of_the_section():
    assert all("different heading" not in rule for rule in d.quoted_rules(PLAN_STUB))


@pytest.mark.parametrize("text", ["# no heading at all\n\n- a bullet\n",
                                  PLAN_STUB + "\n## Locked evaluation rules\n\n- again\n"])
def test_quoted_rules_fails_loudly_rather_than_guessing_the_section(text):
    with pytest.raises(ValueError, match="exactly one"):
        d.quoted_rules(text)


def test_quoted_rules_fails_when_the_section_holds_no_bullets():
    with pytest.raises(ValueError, match="no bullets"):
        d.quoted_rules("## Locked evaluation rules\n\nprose only, no list.\n")


def test_quoted_rules_reads_the_real_pre_registration():
    rules = d.quoted_rules(d.PLAN.read_text())
    assert any(d.BAR_MARKER in rule for rule in rules)
    assert any(d.FRESH_SEED_MARKER in rule for rule in rules)


# --- quoted_bar / the numbers inside it -------------------------------------


def test_quoted_bar_picks_the_one_rule_that_is_the_bar():
    bar = d.quoted_bar(PLAN_STUB)
    assert bar.startswith("- **Simulator ships**")


def test_quoted_bar_refuses_a_marker_that_matches_none_or_several():
    with pytest.raises(ValueError, match="exactly one"):
        d.quoted_bar(PLAN_STUB, "**No such marker**")
    with pytest.raises(ValueError, match="exactly one"):
        d.quoted_bar(PLAN_STUB, "-")  # matches every bullet


def test_joint_bar_and_sigma_are_read_out_of_the_quoted_text():
    bar = d.quoted_bar(PLAN_STUB)
    assert d.joint_bar_from(bar) == 0.01
    assert d.sigma_from(bar) == 0.000346


def test_the_numbers_come_from_the_real_plan_too():
    bar = d.quoted_bar(d.PLAN.read_text())
    assert d.joint_bar_from(bar) == 0.01
    assert d.sigma_from(bar) == 0.000346


def test_joint_bar_from_ignores_the_sigma_clauses_own_more_than():
    """'not worse ... by more than σ_seed' carries no digit, so exactly one
    numeric 'more than' must be found even though the phrase appears twice."""
    bar = d.quoted_bar(PLAN_STUB)
    assert bar.count("more than") == 2
    assert d.joint_bar_from(bar) == 0.01


@pytest.mark.parametrize("text,fn", [
    ("- a bar with no number at all", d.joint_bar_from),
    ("- beats by more than **0.01** and more than **0.02**", d.joint_bar_from),
    ("- a bar naming no sigma", d.sigma_from),
    ("- σ_seed (0.1) and σ_seed (0.2)", d.sigma_from),
])
def test_the_number_readers_fail_loudly_rather_than_guessing(text, fn):
    with pytest.raises(ValueError, match="exactly one"):
        fn(text)


# --- simulator_bar_check ----------------------------------------------------


SIGMA = 0.000346
BAR = 0.01


def _report(joint, winner, folds=None, n=100):
    folds = folds if folds is not None else {"2018": (winner, joint)}
    return {
        "pooled": {"joint_log_loss": joint, "winner_log_loss": winner, "n": n},
        "folds": {year: {"winner_log_loss": w, "joint_log_loss": j}
                  for year, (w, j) in folds.items()},
    }


def _check(cand, inc, sigma=SIGMA, bar=BAR):
    return d.simulator_bar_check(cand, inc, sigma, bar)


def test_a_candidate_clearing_both_clauses_ships():
    out = _check(_report(2.14, 0.6437), _report(2.20, 0.6437))
    assert out["clears_joint_bar"] is True
    assert out["winner_clause_satisfied"] is True
    assert out["ships"] is True
    assert out["joint_delta"] == -0.06
    assert out["winner_delta"] == 0.0
    assert out["joint_margin_past_the_bar"] == pytest.approx(0.05)


def test_the_joint_clause_is_strict_so_a_candidate_exactly_on_the_bar_does_not_ship():
    """'beats ... by MORE than 0.01'. A candidate exactly 0.01 better has not
    beaten it by more than 0.01."""
    out = _check(_report(2.19, 0.6437), _report(2.20, 0.6437))
    assert out["joint_delta"] == -0.01
    assert out["clears_joint_bar"] is False
    assert out["ships"] is False


def test_the_winner_clause_is_a_tolerance_so_a_regression_of_exactly_sigma_passes():
    """'not WORSE than the incumbent's by MORE than sigma_seed' -- equality is
    not worse by more than sigma, so it passes."""
    out = _check(_report(2.14, 0.6437 + SIGMA), _report(2.14 + 0.02, 0.6437))
    assert out["winner_delta"] == SIGMA
    assert out["winner_clause_satisfied"] is True
    assert out["winner_headroom"] == 0.0
    assert out["ships"] is True


def test_a_winner_regression_past_sigma_sinks_an_otherwise_clearing_candidate():
    out = _check(_report(2.10, 0.6437 + SIGMA + 1e-6), _report(2.20, 0.6437))
    assert out["clears_joint_bar"] is True
    assert out["winner_clause_satisfied"] is False
    assert out["winner_headroom"] < 0
    assert out["ships"] is False


def test_a_better_winner_marginal_passes_the_winner_clause_outright():
    out = _check(_report(2.10, 0.6000), _report(2.20, 0.6437))
    assert out["winner_delta"] < 0
    assert out["winner_clause_satisfied"] is True


@pytest.mark.parametrize("candidate,incumbent", [
    (_report(2.10, 0.64, folds={"2018": (0.64, 2.10)}),
     _report(2.20, 0.64, folds={"2018": (0.64, 2.20), "2019": (0.64, 2.20)})),
    (_report(2.10, 0.64, n=100), _report(2.20, 0.64, n=101)),
])
def test_a_non_comparable_pair_never_ships(candidate, incumbent):
    out = _check(candidate, incumbent)
    assert out["clears_joint_bar"] is True
    assert out["comparable"] is False
    assert out["ships"] is False


def test_fold_deltas_are_reported_for_both_metrics_but_do_not_gate():
    """SP3's bar has no per-fold no-regression clause. A candidate with one
    badly regressing fold and a clearing pooled joint still ships -- pinning
    this so nobody silently tightens the pre-registered bar."""
    candidate = _report(2.10, 0.64, folds={"2018": (0.64, 1.90), "2019": (0.64, 2.60)})
    incumbent = _report(2.20, 0.64, folds={"2018": (0.64, 2.20), "2019": (0.64, 2.20)})
    out = _check(candidate, incumbent)
    assert out["joint_fold_deltas"] == {"2018": -0.3, "2019": 0.4}
    assert out["worst_joint_fold_delta"] == 0.4
    assert out["ships"] is True


def test_a_nan_metric_raises_rather_than_reading_as_a_failed_bar():
    with pytest.raises(ValueError, match="not finite"):
        _check(_report(float("nan"), 0.64), _report(2.20, 0.64))


@pytest.mark.parametrize("sigma,bar,match", [
    (float("nan"), BAR, "sigma_seed"),
    (None, BAR, "sigma_seed"),
    (SIGMA, 0.0, "joint_bar"),
    (SIGMA, -0.01, "joint_bar"),
])
def test_the_bar_refuses_a_tolerance_or_margin_it_cannot_apply(sigma, bar, match):
    with pytest.raises(ValueError, match=match):
        d.simulator_bar_check(_report(2.1, 0.64), _report(2.2, 0.64), sigma, bar)


# --- identical_winner_arrays ------------------------------------------------


def _dump(p_winner, y=None, years=None):
    n = len(p_winner)
    return {
        "n": n,
        "p_winner": list(p_winner),
        "y_winner": list(y if y is not None else [1.0] * n),
        "fold_year": list(years if years is not None else [2018] * n),
    }


def test_identical_arrays_are_reported_identical():
    probs = [0.61, 0.4, 0.5231234567890123]
    out = d.identical_winner_arrays(_dump(probs), _dump(probs))
    assert out["identical"] is True
    assert out["same_rows"] is True
    assert out["n_elements_differing"] == 0
    assert out["max_absolute_difference"] == 0.0
    assert out["n_rows_compared"] == 3


def test_the_comparison_is_exact_not_a_tolerance():
    """A re-derivation that merely agrees to within floating-point slop is not
    the same array, and the whole claim is that it IS the same array."""
    a = [0.61, 0.4, 0.5]
    b = [0.61, 0.4, 0.5 + 1e-16]
    out = d.identical_winner_arrays(_dump(a), _dump(b))
    assert out["identical"] is False
    assert out["n_elements_differing"] == 1
    assert 0 < out["max_absolute_difference"] < 1e-15


def test_the_rows_are_checked_before_the_probabilities_are():
    """Two arrays can agree element-wise while describing different fights.
    Equal fold_year and y_winner sequences are what rules that out."""
    probs = [0.61, 0.4]
    out = d.identical_winner_arrays(_dump(probs, y=[1.0, 0.0]), _dump(probs, y=[0.0, 1.0]))
    assert out["same_rows"] is False
    assert out["identical"] is False

    out = d.identical_winner_arrays(_dump(probs, years=[2018, 2018]),
                                    _dump(probs, years=[2018, 2019]))
    assert out["same_rows"] is False
    assert out["identical"] is False


def test_mismatched_lengths_do_not_raise_and_never_read_as_identical():
    out = d.identical_winner_arrays(_dump([0.6, 0.4]), _dump([0.6]))
    assert out["identical"] is False
    assert out["same_rows"] is False
    assert out["n_elements_differing"] is None
    assert out["max_absolute_difference"] is None


# --- finish_round_shares ----------------------------------------------------


def _fights(rows):
    return pd.DataFrame(rows, columns=["method", "finish_round"])


def test_finish_round_shares_counts_only_finishes():
    fights = _fights([
        ("ko_tko", 1), ("ko_tko", 1), ("ko_tko", 1),
        ("submission", 2), ("decision", None), ("decision", None),
    ])
    out = d.finish_round_shares(fights)
    assert out["n_finishes_with_a_round"] == 4
    assert out["counts"] == {"1": 3, "2": 1}
    assert out["shares"] == {"1": 0.75, "2": 0.25}
    assert out["modal_round"] == "1"
    assert out["modal_share"] == 0.75


def test_finish_round_shares_ignores_a_finish_with_no_recorded_round():
    fights = _fights([("ko_tko", 1), ("submission", None)])
    assert d.finish_round_shares(fights)["n_finishes_with_a_round"] == 1


def test_finish_round_shares_raises_when_nothing_was_finished():
    with pytest.raises(ValueError, match="no finishes"):
        d.finish_round_shares(_fights([("decision", None)]))


def test_round_one_is_the_modal_finish_in_the_committed_fights_table():
    """Finding 1 rests on this being true of the real data, not the fixture."""
    out = d.finish_round_shares(pd.read_parquet(d.FIGHTS))
    assert out["modal_round"] == "1"
    assert out["modal_share"] > 0.5


# --- config_match -----------------------------------------------------------


def test_config_match_ignores_the_seeds_and_the_runtime(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(d, "ROOT", tmp_path)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    base = {"candidate": "blend", "drop_columns": ["x"], "half_life": None}
    a.write_text(json.dumps({"config": {**base, "seeds": "5,6,7,8,9", "runtime_sec": 1.0}}))
    b.write_text(json.dumps({"config": {**base, "seeds": "0,1,2,3,4", "runtime_sec": 2.0}}))
    out = d.config_match(a, b)
    assert out["matches"] is True
    assert out["differing_fields"] == []
    assert out["seeds"] == {"candidate": "5,6,7,8,9", "reference": "0,1,2,3,4"}


def test_config_match_reports_a_recipe_that_actually_differs(tmp_path, monkeypatch):
    """A fresh-seed incumbent differing in anything but its seeds is not a
    paired incumbent, and reusing one would defeat the confirmation."""
    import json

    monkeypatch.setattr(d, "ROOT", tmp_path)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"config": {"seeds": "5,6,7,8,9", "drop_columns": ["x"]}}))
    b.write_text(json.dumps({"config": {"seeds": "0,1,2,3,4", "drop_columns": []}}))
    out = d.config_match(a, b)
    assert out["matches"] is False
    assert out["differing_fields"] == ["drop_columns"]


# --- the committed artifact -------------------------------------------------


def test_the_committed_decision_file_is_reproducible():
    """It reads only committed reports, dumps and data, so regenerating it
    must give back exactly what is in the tree."""
    import json

    assert d.build() == json.loads(d.OUT.read_text())


def test_the_artifact_applies_the_bar_at_the_mandatory_fresh_seeds():
    import json

    artifact = json.loads(d.OUT.read_text())
    fresh = artifact["experiments"]["fresh_seed"]
    assert fresh["seeds"] == "5,6,7,8,9"
    assert fresh["bar_check"]["ships"] is True
    assert fresh["bar_check"]["clears_joint_bar"] is True
    assert fresh["bar_check"]["winner_clause_satisfied"] is True
    # The paired incumbent must be the same recipe at a disjoint seed set.
    assert fresh["incumbent_config_matches_on_every_field_but_seeds"]["matches"] is True


def test_the_artifact_verifies_the_winner_clause_rather_than_asserting_it():
    import json

    block = json.loads(d.OUT.read_text())["winner_clause_by_construction"]
    assert block["verification"]["identical"] is True
    assert block["verification"]["same_rows"] is True
    assert block["verification"]["n_elements_differing"] == 0
    assert block["verification"]["n_rows_compared"] > 0


def test_the_artifact_records_the_three_findings_with_their_evidence():
    import json

    findings = json.loads(d.OUT.read_text())["findings"]
    assert set(findings) == {
        "1_macro_f1_was_the_wrong_yardstick",
        "2_the_hybrid_beat_the_pure_simulator",
        "3_the_d3_control_rules_out_the_arithmetic",
    }
    for block in findings.values():
        assert block["finding"] and block["why_it_matters"] and block["evidence"]

    macro = findings["1_macro_f1_was_the_wrong_yardstick"]["evidence"]
    assert macro["macro_f1_delta"] < 0        # the simulator looks worse on macro-F1
    assert macro["joint_log_loss_delta"] < 0  # and is better on the joint

    hybrid = findings["2_the_hybrid_beat_the_pure_simulator"]["evidence"]
    assert hybrid["delta_hybrid_minus_pure"] < 0

    control = findings["3_the_d3_control_rules_out_the_arithmetic"]["evidence"]
    assert control["difference"] == 0.0
    assert control["every_pooled_metric_identical"] is True


def test_the_artifact_reports_the_monte_carlo_diagnostics_the_plan_required():
    import json

    mc = json.loads(d.OUT.read_text())["monte_carlo"]
    assert mc["n_runs"] == 10000 and mc["alpha"] == 1.0
    for run in mc["standard_error"].values():
        assert run["max"] <= 0.005
    for run in mc["zero_mass_cell_fraction"].values():
        assert run["max"] == 0.0


def test_the_artifact_quotes_the_bar_from_the_plan_and_applies_those_numbers():
    import json

    bar = json.loads(d.OUT.read_text())["bar"]
    plan_text = d.PLAN.read_text()
    assert bar["quoted_from_the_plan"] == d.quoted_bar(plan_text)
    assert bar["locked_evaluation_rules"] == d.quoted_rules(plan_text)
    assert bar["joint_bar"] == d.joint_bar_from(bar["quoted_from_the_plan"])
    assert bar["sigma_seed_quoted_in_the_plan"] == d.sigma_from(bar["quoted_from_the_plan"])
    assert bar["sigma_seed_agrees_with_the_measured_floor"] is True
    assert bar["applied_at"] == bar["sigma_seed_quoted_in_the_plan"]


def test_the_artifact_does_not_claim_to_have_deployed_anything():
    import json

    decision = json.loads(d.OUT.read_text())["decision"]
    assert decision["outcome"] == "ship the hybrid (D2)"
    assert "NOT performed by this script" in decision["deployment"]
