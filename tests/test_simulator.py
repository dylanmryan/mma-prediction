"""Monte Carlo fight simulator (SP3 Task 2).

Properties, not golden numbers: every expected value here is either
analytic (computed in the test from the hazard that produced it) or a
smoothing identity, so the file stays meaningful if the sampler is
rewritten.
"""
import numpy as np
import pytest

from mma.hazard import HAZARD_CLASSES
from mma.simulator import METHOD_ORDER, simulate

A_KO, A_SUB, B_KO, B_SUB, SURVIVE = range(5)
N = 10_000


def _rng(seed=0):
    return np.random.default_rng(seed)


def _uniform(rounds, row):
    return np.tile(np.asarray(row, dtype=float), (rounds, 1))


def _certain(rounds, index):
    hazard = np.zeros((rounds, 5))
    hazard[:, index] = 1.0
    return hazard


def _analytic_p_a_wins(hazard, decision_prob, n_rounds):
    """P(feature-corner A wins) under the same generative process."""
    survived, total = 1.0, 0.0
    for r in range(n_rounds):
        total += survived * (hazard[r, A_KO] + hazard[r, A_SUB])
        survived *= hazard[r, SURVIVE]
    return total + survived * decision_prob


def test_hazard_class_order_is_the_shared_contract():
    assert HAZARD_CLASSES == ("a_ko", "a_sub", "b_ko", "b_sub", "survive")
    assert METHOD_ORDER == ("ko_tko", "submission", "decision")


# --------------------------------------------------------------------------
# degenerate cases
# --------------------------------------------------------------------------

def test_certain_ko_in_round_one():
    out = simulate(_certain(3, A_KO), 0.5, 3, N, _rng())
    assert out.p_a_wins == pytest.approx(1.0, abs=2e-3)
    assert out.method_probs[0] == pytest.approx(1.0, abs=2e-3)
    assert out.round_probs[0] == pytest.approx(1.0, abs=2e-3)
    assert out.p_distance == pytest.approx(0.0, abs=2e-3)
    assert out.finish_counts[0, 0, 0] == N


def test_certain_survival_reproduces_the_decision_model():
    out = simulate(_certain(3, SURVIVE), 0.7, 3, N, _rng())
    assert out.p_distance == pytest.approx(1.0, abs=2e-3)
    assert out.method_probs[2] == pytest.approx(1.0, abs=2e-3)
    assert out.finish_counts.sum() == 0
    assert out.decision_counts.sum() == N
    se = np.sqrt(0.7 * 0.3 / N)
    assert out.p_a_wins == pytest.approx(0.7, abs=4 * se + 2e-3)


# --------------------------------------------------------------------------
# coherence
# --------------------------------------------------------------------------

def test_marginals_are_coherent():
    out = simulate(_uniform(5, [0.08, 0.04, 0.06, 0.03, 0.79]), 0.55, 5, N, _rng(1))
    assert out.p_a_wins + out.p_b_wins == pytest.approx(1.0)
    assert out.method_probs.sum() == pytest.approx(1.0)
    assert out.round_probs.sum() == pytest.approx(1.0)


def test_joint_cells_sum_to_one():
    out = simulate(_uniform(5, [0.08, 0.04, 0.06, 0.03, 0.79]), 0.55, 3, N, _rng(2))
    assert out.finish_probs.sum() + out.decision_probs.sum() == pytest.approx(1.0)


def test_p_distance_is_the_decision_cell_mass():
    out = simulate(_uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73]), 0.6, 3, N, _rng(3))
    assert out.p_distance == pytest.approx(out.decision_probs.sum())


def test_marginals_agree_with_the_joint_cells():
    out = simulate(_uniform(5, [0.09, 0.05, 0.07, 0.04, 0.75]), 0.4, 5, N, _rng(4))
    assert out.p_a_wins == pytest.approx(
        out.finish_probs[0].sum() + out.decision_probs[0]
    )
    assert out.method_probs[0] == pytest.approx(out.finish_probs[:, 0, :].sum())
    assert out.method_probs[1] == pytest.approx(out.finish_probs[:, 1, :].sum())
    finish_mass = out.finish_probs.sum()
    assert out.round_probs == pytest.approx(
        out.finish_probs.sum(axis=(0, 1)) / finish_mass
    )


# --------------------------------------------------------------------------
# round masking
# --------------------------------------------------------------------------

def test_three_round_fight_has_no_mass_beyond_round_three():
    hazard = _uniform(5, [0.10, 0.05, 0.08, 0.04, 0.73])
    out = simulate(hazard, 0.5, 3, N, _rng(5))
    assert out.finish_probs[:, :, 3:].sum() == 0.0
    assert out.finish_counts[:, :, 3:].sum() == 0
    assert out.round_probs[3:].sum() == 0.0


def test_five_round_fight_puts_mass_in_rounds_four_and_five():
    hazard = _uniform(5, [0.10, 0.05, 0.08, 0.04, 0.73])
    out = simulate(hazard, 0.5, 5, N, _rng(6))
    assert out.finish_counts[:, :, 3].sum() > 0
    assert out.finish_counts[:, :, 4].sum() > 0
    assert out.round_probs[4] > 0


@pytest.mark.parametrize("scheduled", [1, 2])
def test_short_scheduled_fights(scheduled):
    hazard = _uniform(5, [0.10, 0.05, 0.08, 0.04, 0.73])
    out = simulate(hazard, 0.5, scheduled, N, _rng(7))
    assert out.finish_probs[:, :, scheduled:].sum() == 0.0
    assert out.finish_probs.sum() + out.decision_probs.sum() == pytest.approx(1.0)
    expected = _analytic_p_a_wins(hazard, 0.5, scheduled)
    se = np.sqrt(expected * (1 - expected) / N)
    assert out.p_a_wins == pytest.approx(expected, abs=4 * se + 2e-3)


def test_scheduled_rounds_may_be_unknown():
    """NA scheduled rounds: every round the hazard supplies is reachable."""
    hazard = _uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73])
    out = simulate(hazard, 0.5, None, N, _rng(8))
    assert out.n_rounds == 3
    assert out.finish_probs.sum() + out.decision_probs.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Monte Carlo error
# --------------------------------------------------------------------------

def test_matches_the_analytic_p_a_wins_within_four_standard_errors():
    hazard = _uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73])
    expected = _analytic_p_a_wins(hazard, 0.75, 3)
    out = simulate(hazard, 0.75, 3, N, _rng(9))
    se = np.sqrt(expected * (1 - expected) / N)
    assert out.p_a_wins == pytest.approx(expected, abs=4 * se + 2e-3)


def test_standard_error_is_below_the_registered_threshold():
    hazard = _uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73])
    out = simulate(hazard, 0.75, 3, N, _rng(10))
    assert out.p_a_wins_standard_error < 0.005
    assert out.p_a_wins_standard_error == pytest.approx(
        np.sqrt(out.p_a_wins * (1 - out.p_a_wins) / N), rel=1e-6
    )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_output():
    hazard = _uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73])
    a = simulate(hazard, 0.6, 3, N, _rng(11))
    b = simulate(hazard, 0.6, 3, N, _rng(11))
    assert np.array_equal(a.finish_counts, b.finish_counts)
    assert np.array_equal(a.decision_counts, b.decision_counts)
    assert a.p_a_wins == b.p_a_wins


def test_different_seeds_differ_but_agree_within_error():
    hazard = _uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73])
    a = simulate(hazard, 0.6, 3, N, _rng(11))
    b = simulate(hazard, 0.6, 3, N, _rng(12))
    assert not np.array_equal(a.finish_counts, b.finish_counts)
    assert abs(a.p_a_wins - b.p_a_wins) < 8 * a.p_a_wins_standard_error


def test_input_arrays_are_not_mutated():
    hazard = _uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73])
    before = hazard.copy()
    simulate(hazard, 0.6, 3, N, _rng(13))
    assert np.array_equal(hazard, before)


# --------------------------------------------------------------------------
# Laplace smoothing
# --------------------------------------------------------------------------

def _valid_cells(n_rounds):
    return 4 * n_rounds + 2


def test_valid_cell_with_no_simulated_occurrences_gets_nonzero_probability():
    """No submission can ever be drawn, but a submission is still possible."""
    hazard = _uniform(3, [0.10, 0.0, 0.08, 0.0, 0.82])
    out = simulate(hazard, 0.6, 3, 1_000, _rng(14))
    assert out.finish_counts[0, 1].sum() == 0
    assert (out.finish_probs[0, 1] > 0).all()
    assert out.finish_probs[0, 1, 0] == pytest.approx(
        1.0 / (1_000 + _valid_cells(3))
    )


def test_invalid_cell_is_exactly_zero_and_out_of_the_denominator():
    hazard = _uniform(5, [0.10, 0.05, 0.08, 0.04, 0.73])
    out = simulate(hazard, 0.6, 3, 1_000, _rng(15))
    assert (out.finish_probs[:, :, 3:] == 0.0).all()
    denominator = 1_000 + _valid_cells(3)
    assert out.finish_probs[0, 0, 0] == pytest.approx(
        (out.finish_counts[0, 0, 0] + 1) / denominator
    )
    assert out.decision_probs[0] == pytest.approx(
        (out.decision_counts[0] + 1) / denominator
    )


def test_smoothing_strength_is_configurable_and_zero_is_raw():
    hazard = _uniform(3, [0.10, 0.0, 0.08, 0.0, 0.82])
    out = simulate(hazard, 0.6, 3, 1_000, _rng(16), alpha=0.0)
    assert out.finish_probs[0, 1, 0] == 0.0
    assert out.finish_probs.sum() + out.decision_probs.sum() == pytest.approx(1.0)


def test_zero_mass_valid_cells_are_reported():
    hazard = _uniform(3, [0.10, 0.0, 0.08, 0.0, 0.82])
    out = simulate(hazard, 0.6, 3, 1_000, _rng(17))
    # both submission corners, every one of the 3 rounds, are never drawn
    assert out.n_zero_mass_cells == 6
    assert out.n_valid_cells == _valid_cells(3)


def test_raw_counts_total_the_run_count():
    out = simulate(_uniform(3, [0.10, 0.05, 0.08, 0.04, 0.73]), 0.6, 3, N, _rng(18))
    assert out.finish_counts.sum() + out.decision_counts.sum() == N
    assert out.n_runs == N


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hazard, message",
    [
        (np.zeros((3, 4)), "shape"),
        (np.full((3, 5), 0.5), "sum to 1"),
        (np.zeros((0, 5)), "at least one round"),
    ],
)
def test_bad_hazard_is_rejected(hazard, message):
    with pytest.raises(ValueError, match=message):
        simulate(hazard, 0.5, 3, 100, _rng())


def test_bad_decision_probability_is_rejected():
    with pytest.raises(ValueError, match="decision_prob"):
        simulate(_certain(3, SURVIVE), 1.5, 3, 100, _rng())


def test_negative_hazard_is_rejected():
    hazard = _uniform(3, [0.10, -0.05, 0.13, 0.04, 0.78])
    with pytest.raises(ValueError, match="non-negative"):
        simulate(hazard, 0.5, 3, 100, _rng())


def test_zero_runs_is_rejected():
    with pytest.raises(ValueError, match="n_runs"):
        simulate(_certain(3, SURVIVE), 0.5, 3, 0, _rng())


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------

def test_ten_thousand_runs_stay_cheap_enough_for_a_walk_forward():
    """Guards the vectorisation. Measured ~1.4 ms for a five-round fight, so
    a whole walk-forward (4,804 evaluation fights x 5 members x both corner
    orderings) is ~1 minute of simulation. The 50 ms bar is deliberately
    loose -- it only catches a fall back to a per-run Python loop."""
    import time

    hazard = _uniform(5, [0.10, 0.05, 0.08, 0.04, 0.73])
    rng = _rng(19)
    simulate(hazard, 0.6, 5, N, rng)  # warm up numpy
    start = time.perf_counter()
    for _ in range(10):
        simulate(hazard, 0.6, 5, N, rng)
    assert (time.perf_counter() - start) / 10 < 0.05
