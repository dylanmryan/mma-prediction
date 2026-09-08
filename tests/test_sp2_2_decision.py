"""Pure helpers behind SP2.2's decision artifact (scripts/sp2_2_decision.py).

The script itself reads committed reports and decides nothing; what is worth
pinning is the arithmetic it uses to describe the ECE gate, and the guarantee
that the rules it quotes are the pre-registration's own text rather than a
paraphrase that could drift from it.
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


def test_the_artifact_decides_nothing():
    """SP2.2's rule 4 reserves the call for a human. If this test ever fails
    because some field started asserting an outcome, that is the failure it
    exists to catch."""
    import json

    decision = json.loads(d.OUT.read_text())
    assert decision["decided"] is False
    assert decision["rules"]["status"]["4"]["resolved"] is False
    assert decision["awaiting_human_call"]["blocked_on"] == "rule 4"
    assert decision["remediation_isotonic"]["label"].startswith("POST-HOC VARIANT")


def test_the_quoted_rules_match_the_pre_registration_on_disk():
    import json

    decision = json.loads(d.OUT.read_text())
    assert decision["rules"]["text"] == d.quoted_rules(d.PLAN.read_text())
