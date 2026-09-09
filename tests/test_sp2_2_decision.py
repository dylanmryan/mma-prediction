"""Pure helpers behind SP2.2's decision artifact (scripts/sp2_2_decision.py).

The script reads committed reports and applies the pre-registration's rules;
what is worth pinning is the arithmetic it uses to describe the ECE gate, the
guarantee that the rules it quotes are the pre-registration's own text rather
than a paraphrase that could drift from it, and -- since rule 4's gate was
AMENDED after the numbers were seen -- that the amendment is quoted from the
plan file too, so the artifact cannot state a gate the plan does not.
"""
from __future__ import annotations

import numpy as np
import pytest

import scripts.sp2_2_decision as d


# --- quoted_rules -----------------------------------------------------------


PLAN_STUB = """# A plan

### Candidates

- **B0** something.

### Decision rule (mechanical, fixed now)

1. Build the fresh-seed paired incumbent.
2. A candidate **ships** only if it clears the bar at seeds 0-4 **and** again at 5-9.
3. **Choice between candidates:** ship B0 unless B1 beats it.
4. **Calibration is a gate, not a tiebreak.** Stop for a human call.
5. If nothing clears, revert and record.

<!-- AMENDMENT: rule 4 -->
**AMENDMENT** the gate is re-specified to three disjoint seed sets.
<!-- END AMENDMENT -->

### What would make this experiment wrong

- Dropping the ECE gate because log-loss looks good.
"""


def test_quoted_rules_takes_the_rules_from_the_plan_verbatim():
    rules = d.quoted_rules(PLAN_STUB)
    assert sorted(rules) == ["1", "2", "3", "4", "5"]
    assert rules["4"] == ("4. **Calibration is a gate, not a tiebreak.** Stop for a human call.")
    # markdown emphasis is preserved -- "verbatim" means verbatim
    assert "**ships**" in rules["2"]
    # the list that follows the next heading is not swept in
    assert not any("Dropping the ECE gate" in text for text in rules.values())


def test_quoted_rules_reads_the_real_pre_registration():
    rules = d.quoted_rules(d.PLAN.read_text())
    assert sorted(rules) == ["1", "2", "3", "4", "5"]
    assert rules["4"].startswith("4. **Calibration is a gate, not a tiebreak.**")
    assert "stop for a human call" in rules["4"]
    assert "fresh-seed re-score at seeds 5" in rules["2"]


def test_quoted_rules_fails_loudly_rather_than_quoting_a_partial_list():
    with pytest.raises(ValueError, match="expected rules 1-5"):
        d.quoted_rules("### Decision rule\n\n1. only one rule.\n")
    with pytest.raises(ValueError, match="exactly one"):
        d.quoted_rules("no heading here")


def test_the_amendment_does_not_disturb_the_original_rules():
    """The amendment sits BELOW the numbered list and rule 4's original
    wording is left in place, so the quoted rules are still the ones written
    before anything was run."""
    rules = d.quoted_rules(PLAN_STUB)
    assert sorted(rules) == ["1", "2", "3", "4", "5"]
    assert "AMENDMENT" not in " ".join(rules.values())
    real = d.quoted_rules(d.PLAN.read_text())
    assert "(0.0088)" in real["4"]  # the original single-value gate, unedited


# --- quoted_amendment -------------------------------------------------------


def test_quoted_amendment_takes_the_block_from_the_plan_verbatim():
    text = d.quoted_amendment(PLAN_STUB)
    assert text == "**AMENDMENT** the gate is re-specified to three disjoint seed sets."
    assert d.AMENDMENT_START not in text and d.AMENDMENT_END not in text


def test_quoted_amendment_reads_the_real_pre_registration():
    text = d.quoted_amendment(d.PLAN.read_text())
    assert "amended 2026-09-08" in text
    assert "three disjoint seed sets" in text
    assert "is a real weakness and is recorded as one" in text


def test_quoted_amendment_fails_loudly_when_the_block_is_missing_or_doubled():
    with pytest.raises(ValueError, match="exactly one"):
        d.quoted_amendment("no markers here")
    doubled = PLAN_STUB + PLAN_STUB
    with pytest.raises(ValueError, match="exactly one"):
        d.quoted_amendment(doubled)


# --- amended_ece_gate -------------------------------------------------------


def test_amended_gate_pools_the_two_arms_standard_deviations():
    """The tolerance is 2 sigma of the POOLED spread of the two arms, not of
    either one alone -- the incumbent's own sd is four times the candidate's,
    and reading the gate off whichever arm suits is the failure mode the
    amendment exists to close."""
    import math

    incumbent = {"mean": 0.011633333333333334, "sigma_seed": 0.0038370995990895693,
                 "n_reports": 3}
    candidate = {"mean": 0.013366666666666666, "sigma_seed": 0.0010016652800877814,
                 "n_reports": 3}
    out = d.amended_ece_gate(incumbent, candidate)
    expected = math.sqrt((incumbent["sigma_seed"] ** 2 + candidate["sigma_seed"] ** 2) / 2)
    assert out["pooled_sd"] == pytest.approx(expected, abs=1e-12)
    assert out["tolerance_2sigma"] == pytest.approx(2 * expected, abs=1e-12)
    assert out["difference_candidate_minus_incumbent"] == pytest.approx(0.0017333, abs=1e-6)
    assert out["passes"] is True


def test_amended_gate_fails_when_the_candidate_is_worse_by_more_than_2_sigma():
    incumbent = {"mean": 0.010, "sigma_seed": 0.001, "n_reports": 3}
    candidate = {"mean": 0.020, "sigma_seed": 0.001, "n_reports": 3}
    out = d.amended_ece_gate(incumbent, candidate)
    assert out["passes"] is False
    assert out["difference_candidate_minus_incumbent"] == pytest.approx(0.010)


def test_amended_gate_passes_a_better_calibrated_candidate_outright():
    incumbent = {"mean": 0.020, "sigma_seed": 0.004, "n_reports": 3}
    candidate = {"mean": 0.010, "sigma_seed": 0.001, "n_reports": 3}
    assert d.amended_ece_gate(incumbent, candidate)["passes"] is True


# --- spread -----------------------------------------------------------------


def test_spread_is_the_sample_standard_deviation():
    out = d.spread([0.0088, 0.0160, 0.0113])
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(np.mean([0.0088, 0.0160, 0.0113]), abs=1e-6)
    assert out["sd"] == pytest.approx(np.std([0.0088, 0.0160, 0.0113], ddof=1))
    assert (out["min"], out["max"]) == (0.0088, 0.0160)


def test_spread_of_one_value_measures_no_spread_at_all():
    assert np.isnan(d.spread([0.0088])["sd"])
    with pytest.raises(ValueError, match="at least one value"):
        d.spread([])


# --- binning_sensitivity ----------------------------------------------------


def _dump(y, p):
    return {"y_winner": list(map(float, y)), "p_winner": list(map(float, p))}


def test_binning_sensitivity_reports_the_gap_at_every_bin_count():
    rng = np.random.default_rng(0)
    p_inc = rng.uniform(0.3, 0.7, size=400)
    y = (rng.uniform(size=400) < p_inc).astype(float)
    p_cand = np.clip(p_inc + 0.05, 0.01, 0.99)  # deliberately over-confident
    out = d.binning_sensitivity(_dump(y, p_inc), _dump(y, p_cand))
    assert [row["n_bins"] for row in out["rows"]] == [5, 10, 15, 20]
    for row in out["rows"]:
        assert row["gap"] == pytest.approx(row["candidate_ece"] - row["incumbent_ece"], abs=1e-6)
        assert row["candidate_worse"] is True
    assert out["sign_flips"] is False
    assert out["candidate_worse_at_every_bin_count"] is True


def test_binning_sensitivity_detects_a_sign_flip():
    """The whole point of the sweep: if the sign of the gap depends on the bin
    count, rule 4's gate is being decided by a binning artifact."""
    y = np.array([1.0, 0.0] * 20)
    # incumbent: two tight clusters that a 5-bin grid merges and a 20-bin grid
    # separates; candidate: one spread-out band. Which looks better depends on
    # where the edges fall.
    p_inc = np.array([0.42, 0.58] * 20)
    p_cand = np.linspace(0.05, 0.95, 40)
    out = d.binning_sensitivity(_dump(y, p_inc), _dump(y, p_cand), bin_counts=(2, 5, 10, 20))
    signs = {int(np.sign(row["gap"])) for row in out["rows"]}
    assert out["sign_flips"] is (len(signs) > 1)
    assert out["candidate_worse_at_every_bin_count"] is all(
        row["candidate_worse"] for row in out["rows"])


def test_binning_sensitivity_at_ten_bins_is_the_reports_own_number():
    """Row `n_bins=10` must reproduce what score_rows wrote into the report,
    or the sweep is describing a different quantity than the gate uses."""
    from mma.evaluate import expected_calibration_error

    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.8, size=300)
    y = (rng.uniform(size=300) < p).astype(float)
    out = d.binning_sensitivity(_dump(y, p), _dump(y, p))
    ten = next(row for row in out["rows"] if row["n_bins"] == 10)
    assert ten["incumbent_ece"] == pytest.approx(expected_calibration_error(y, p), abs=1e-6)
    assert ten["gap"] == 0.0
    assert out["sign_flips"] is False


# --- worst_bins -------------------------------------------------------------


def test_worst_bins_ranks_by_contribution_to_ece_not_by_raw_gap():
    """A big miscalibration in a nearly empty bin means something different
    from a small one in a heavily populated bin, and ECE already knows that;
    the ranking has to agree with ECE's own weighting."""
    from mma.evaluate import reliability_curve

    y = np.concatenate([np.ones(300), np.zeros(100), np.ones(4)])
    p = np.concatenate([np.full(300, 0.62), np.full(100, 0.66), np.full(4, 0.05)])
    curve = reliability_curve(y, p, n_bins=10)
    top = d.worst_bins(curve, k=2)
    assert top[0]["ece_contribution"] >= top[1]["ece_contribution"]
    # the 4-row bin at 0.05 has the largest RAW gap but a tiny weight
    sparse = next(row for row in curve if row["lo"] == 0.0)
    assert abs(sparse["gap"]) > abs(top[0]["gap"])
    assert sparse["weight"] * abs(sparse["gap"]) < top[0]["ece_contribution"]


def test_worst_bins_skips_empty_bins():
    from mma.evaluate import reliability_curve

    curve = reliability_curve([1.0, 0.0], [0.45, 0.45], n_bins=10)
    top = d.worst_bins(curve)
    assert len(top) == 1 and top[0]["n"] == 2


# --- fresh_seed_branch ------------------------------------------------------


@pytest.mark.parametrize("a,b,expected", [
    (True, True, True), (True, False, False), (False, True, False), (False, False, False),
])
def test_rule_2_needs_both_seed_sets(a, b, expected):
    out = d.fresh_seed_branch({"ships": a}, {"ships": b})
    assert out["rule_2_satisfied"] is expected
    assert (out["clears_at_seeds_0_4"], out["clears_at_seeds_5_9"]) == (a, b)


# --- the committed artifact -------------------------------------------------


def test_the_committed_decision_file_is_reproducible():
    """It reads only committed reports, so regenerating it must give back
    exactly what is in the tree."""
    import json

    assert d.build() == json.loads(d.OUT.read_text())


def test_the_artifact_records_the_human_call_rather_than_taking_it_itself():
    """Rule 4 reserved the call for a human; the human made it on 2026-09-08
    (B1 ships, ECE gate re-specified). The artifact must record THAT -- the
    outcome, the amended gate and the amendment's own text -- and must still
    not invent one: the recorded call names the human decision and the
    amendment it rests on."""
    import json

    decision = json.loads(d.OUT.read_text())
    assert decision["decided"] is True
    assert decision["decision"]["outcome"] == "ship B1"
    assert decision["decision"]["taken_by"] == "human call reserved by rule 4"
    assert decision["decision"]["date"] == "2026-09-08"
    assert decision["rules"]["status"]["4"]["resolved"] is True
    assert decision["rules"]["status"]["4"]["gate_applied"] == "amended"
    assert decision["remediation_isotonic"]["label"].startswith("POST-HOC VARIANT")
    assert decision["decision"]["ships"]["isotonic_remediation"] is False


def test_the_artifact_keeps_the_original_gate_visible_next_to_the_amended_one():
    """Amending a pre-registered rule after seeing results is only defensible
    if the original stays readable beside the replacement."""
    import json

    gate = json.loads(d.OUT.read_text())["ece_gate"]
    assert "(0.0088)" in gate["rule_as_written"]
    assert gate["B1_fails_the_gate_as_written"] is True
    amended = gate["amended"]
    assert amended["passes"] is True
    assert amended["text"] == d.quoted_amendment(d.PLAN.read_text())
    assert amended["weakness"].startswith("Amending a pre-registered gate")


def test_the_quoted_rules_match_the_pre_registration_on_disk():
    import json

    decision = json.loads(d.OUT.read_text())
    assert decision["rules"]["text"] == d.quoted_rules(d.PLAN.read_text())

    assert decision["rules"]["amendment"] == d.quoted_amendment(d.PLAN.read_text())


# --- the form that actually ships -------------------------------------------


def test_deployed_form_reports_the_fixed_temperature_scorer_not_the_harness_one():
    """The harness fits a temperature per fold; deployment applies one fixed
    value. Publishing the harness's calibration as the model's was the defect
    this block exists to close, so the block must carry the deployed
    temperature and the deployed form's own pooled metrics at all four bin
    counts, per fold as well as pooled."""
    import json

    from mma.inference import load_blend_config

    decision = json.loads(d.OUT.read_text())
    shipped = decision["deployed_form"]
    record = json.loads(d.BLEND_TEMPERATURE.read_text())

    assert shipped["temperature"] == load_blend_config()["temperature"]
    assert shipped["temperature"] == record["deployed"]["temperature"]
    assert shipped["pooled"] == record["deployed"]["honest_pooled"]
    assert sorted(shipped["pooled"]["ece"]) == ["10", "15", "20", "5"]
    assert sorted(shipped["folds"]) == [str(y) for y in record["fold_years"]]
    assert shipped["round_trip_verification"]["passed"] is True


def test_deployed_form_compares_against_both_the_harness_form_and_the_incumbent():
    import json

    decision = json.loads(d.OUT.read_text())
    shipped = decision["deployed_form"]
    b1 = json.loads(d.B1.read_text())["pooled"]
    incumbent = json.loads(d.INCUMBENT.read_text())["pooled"]

    assert shipped["vs_harness_form"]["harness_winner_log_loss"] == b1["winner_log_loss"]
    assert shipped["vs_harness_form"]["harness_ece_10_bins"] == b1["ece"]
    assert shipped["vs_incumbent_I"]["incumbent_ece_10_bins"] == incumbent["ece"]
    assert shipped["vs_incumbent_I"]["incumbent_winner_log_loss"] == incumbent["winner_log_loss"]


def test_the_artifact_says_which_form_the_ece_gate_was_evaluated_on():
    """Both forms of rule 4's gate compared numbers taken from walk-forward
    reports, and every walk-forward report scores the per-fold-fitted form. The
    artifact has to say so rather than let a reader assume the gate judged the
    shipped scorer."""
    import json

    decision = json.loads(d.OUT.read_text())
    assert "HARNESS form" in decision["ece_gate"]["evaluated_on"]
    against = decision["deployed_form"]["against_the_amended_ece_gate"]
    assert "HARNESS form" in against["gate_was_evaluated_on"]
    assert against["not_a_re_run_of_the_gate"].startswith("The gate is not re-applied")


def test_the_deployed_forms_standing_against_the_amended_threshold_is_stated():
    """The threshold is the amended gate's own: the incumbent mean plus its
    2-sigma tolerance. The deployed form is inside it; the median rule that
    shipped before the derivation is not, and both facts are recorded."""
    import json

    decision = json.loads(d.OUT.read_text())
    gate = decision["ece_gate"]["amended"]
    against = decision["deployed_form"]["against_the_amended_ece_gate"]

    assert against["threshold"] == round(gate["incumbent_mean"] + gate["tolerance_2sigma"], 6)
    assert against["deployed_ece_10_bins"] <= against["threshold"]
    assert against["deployed_within_the_threshold"] is True
    superseded = against["the_rule_that_shipped_before"]
    assert superseded["temperature"] == 0.8
    assert superseded["within_the_threshold"] is False
