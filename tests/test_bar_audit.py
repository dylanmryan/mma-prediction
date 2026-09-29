"""What the shipping bar should be, measured rather than assumed.

Every bar in this project is `max(0.003, 2 sigma_seed)`, and sigma_seed for
the deployed blend is 0.0001 -- so the operative bar is thirty times the only
noise anyone measured. But seed noise answers "would a refit land here again?",
not "will this gain hold on the next four thousand fights", and the second
question is the one a shipping decision actually asks. This instrument
measures the second: the sampling standard error of a PAIRED per-fight
log-loss difference.

The tests pin three things the audit is worthless without:

* the per-fight decomposition really does pool to the numbers in the committed
  reports, so the delta being bootstrapped is the delta that was decided on;
* two dumps are only ever paired after their outcomes are checked element-wise,
  because these dumps carry no `fight_id` and a positional join that silently
  mismatches is the exact bug class that broke the market benchmark once;
* a bar derived from it covers the project's own known-null pair (one recipe,
  two seed sets) while still clearing its one known-real gain (SP2.2's blend).

That last pair is why the audit does not stop at the bootstrap. A paired
bootstrap over fights asks "would this delta hold on new fights, given these
two fits" -- and two seed sets ARE two different fits, so the bootstrap is
entitled to resolve a swap between them. Seed variance is a second and
orthogonal source, already measured at sigma_seed = 0.0001, and an honest bar
has to carry both.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.bar_audit import (
    ARTIFACT,
    PAIRS,
    REFERENCE_PAIR,
    SIGMA_SEED,
    audit,
    bootstrap,
    combined_se,
    implied_bars,
    load_dump,
    paired_delta,
    per_fight_log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "models" / "walkforward" / "preds"

#: SP2.2's shipped pair: the two-family blend against the torch-alone
#: incumbent it replaced. The committed reports pool to these.
BLEND = PREDS / "blend_b1.json"
INCUMBENT = PREDS / "torch_external_diffsonly_extslice.json"
#: The same blend recipe at seeds 5-9: a pair whose true effect is zero.
BLEND_SEEDS5 = PREDS / "blend_b1_cells_seeds5.json"


# --- the delta being bootstrapped is the delta that was decided on -----------

def test_per_fight_log_loss_pools_to_the_committed_report():
    """If this drifts, every number below is measuring something else."""
    dump = load_dump(BLEND)
    pooled = float(np.mean(per_fight_log_loss(dump.y, dump.p)))
    assert round(pooled, 4) == 0.6437


def test_paired_delta_pools_to_the_difference_the_experiment_recorded():
    """SP2.2 recorded the blend at -0.0039 against its incumbent. The sign
    convention is the project's: candidate minus incumbent, so NEGATIVE is an
    improvement."""
    delta = paired_delta(load_dump(BLEND), load_dump(INCUMBENT))
    assert round(float(np.mean(delta)), 4) == -0.0039


# --- pairing is checked, never assumed ---------------------------------------

def test_dumps_of_different_lengths_are_refused():
    """`hybrid_e2_cells.json` covers the refreshed 4,856-row table; the rest
    cover the 4,804-row one. Pairing them positionally would compare fights
    to other fights."""
    with pytest.raises(ValueError, match="length"):
        paired_delta(load_dump(PREDS / "hybrid_e2_cells.json"), load_dump(BLEND))


def test_dumps_whose_outcomes_disagree_are_refused(tmp_path):
    """These dumps carry no `fight_id`, so alignment can only be argued from
    the outcome and fold-year columns -- which means it has to be CHECKED. A
    positional join that quietly mismatched is what broke the market benchmark
    once, and the guard has to exist before the instrument is trusted."""
    good = json.loads(BLEND.read_text())
    shuffled = dict(good, y_winner=list(reversed(good["y_winner"])))
    path = tmp_path / "shuffled.json"
    path.write_text(json.dumps(shuffled))
    with pytest.raises(ValueError, match="outcome"):
        paired_delta(load_dump(BLEND), load_dump(path))


# --- the bootstrap can tell a real gain from a null one ----------------------

def test_a_bar_from_this_instrument_covers_a_pure_seed_swap():
    """The requirement that makes a derived bar usable at all. One recipe at
    two seed sets is a true effect of zero; if the bar it derives does not
    cover that delta, the bar would ship noise."""
    years = load_dump(BLEND).fold_year
    delta = paired_delta(load_dump(BLEND), load_dump(BLEND_SEEDS5))
    se = combined_se(bootstrap(delta, years)["se"])
    assert implied_bars(se)["single_arm"] > abs(float(np.mean(delta)))


def test_an_effect_nobody_disputes_clears_the_bar_it_derives():
    """The instrument's other half: it has to SEE a real effect, or the bar it
    derives would reject everything.

    The effect used here is the isotonic diagnostic (0.7049 against 0.6437) and
    NOT SP2.2's shipped blend, deliberately. Whether that 0.0039 survives a
    sampling-error bar is the question this whole audit exists to answer, and a
    test that asserted the answer would decide it in advance."""
    years = load_dump(BLEND).fold_year
    delta = paired_delta(load_dump(PREDS / "blend_b1_isotonic.json"), load_dump(BLEND))
    se = combined_se(bootstrap(delta, years)["se"])
    assert abs(float(np.mean(delta))) > implied_bars(se)["single_arm"]


def test_seed_noise_is_carried_alongside_the_sampling_error_not_instead_of_it():
    """Both sources or neither: the combined error has to exceed each."""
    assert combined_se(0.0) == pytest.approx(SIGMA_SEED)
    assert combined_se(0.001) > 0.001


def test_a_search_of_many_arms_needs_a_higher_bar_than_one():
    """The project already judges a best-of-N against the sigma*sqrt(2 ln N)
    such a search produces from nothing; the derived bar keeps that clause."""
    bars = implied_bars(0.001)
    assert implied_bars(0.001, n_arms=25)["after_search"] > bars["after_search"]
    assert bars["after_search"] == pytest.approx(bars["single_arm"])


def test_the_bootstrap_reports_how_often_the_effect_reverses():
    """A two-standard-error bar is one reading; the share of resamples that
    land on the OTHER side of zero is the more interpretable one, and the two
    can disagree for an effect sitting near the boundary. Reporting only the
    bar would hide that."""
    years = load_dump(BLEND).fold_year
    real = bootstrap(paired_delta(load_dump(BLEND), load_dump(INCUMBENT)), years)
    null = bootstrap(paired_delta(load_dump(BLEND), load_dump(BLEND_SEEDS5)), years)
    assert real["p_wrong_sign"] < 0.05
    # a true zero effect reverses in something near half of all resamples
    assert null["p_wrong_sign"] > 0.2


def test_the_bootstrap_is_deterministic_under_its_fixed_seed():
    delta = paired_delta(load_dump(BLEND), load_dump(INCUMBENT))
    years = load_dump(BLEND).fold_year
    assert bootstrap(delta, years) == bootstrap(delta, years)


# --- the artifact ------------------------------------------------------------

def test_the_audit_reports_a_bar_for_every_pair_it_was_given():
    report = audit(PAIRS[:1])
    assert [p["pair"] for p in report["pairs"]] == [PAIRS[0].pair]
    assert report["pairs"][0]["bar_single_arm"] > 0.0


def test_the_audit_names_the_reference_pair_its_headline_floor_comes_from():
    """A single "detection floor" is only meaningful if the comparison it was
    measured on is named, because the standard error depends on the pair."""
    report = audit(PAIRS)
    assert report["harness_detection_floor"]["measured_on"] == REFERENCE_PAIR.pair
    floor = report["harness_detection_floor"]["value"]
    match = [p for p in report["pairs"] if p["pair"] == REFERENCE_PAIR.pair][0]
    assert floor == match["bar_single_arm"]


def test_the_audit_states_that_it_changes_no_bar():
    """`residual_probe.py` carries the same clause. Measuring a bar and
    adopting one are separate acts, and the second is pre-registered."""
    assert "decides nothing" in audit(PAIRS[:1])["note"].lower()


def test_the_committed_artifact_is_reproducible():
    """The audit is deterministic given its seed, so the file in the repo has
    to be the file the script writes. If this fails, re-run the script."""
    if not ARTIFACT.exists():
        pytest.skip("run scripts/bar_audit.py to write the artifact")
    committed = json.loads(ARTIFACT.read_text())
    fresh = audit(PAIRS)
    assert fresh["pairs"] == committed["pairs"]
    assert fresh["harness_detection_floor"] == committed["harness_detection_floor"]
