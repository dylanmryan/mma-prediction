import numpy as np
import pytest

from mma.evaluate import (
    accuracy,
    brier_score,
    expected_calibration_error,
    joint_cell_index,
    joint_cell_log_loss,
    joint_outcome_log_loss,
    log_loss,
    macro_f1,
    realised_zero_mass_fraction,
    reliability_curve,
)


def test_log_loss_perfect_and_uninformed():
    assert log_loss([1, 0], [0.999999, 0.000001]) == pytest.approx(0.0, abs=1e-4)
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))


def test_log_loss_clips_extremes():
    # p=0 or 1 must not produce inf
    assert np.isfinite(log_loss([1], [0.0]))


def test_accuracy_threshold():
    assert accuracy([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6]) == 0.5


def test_brier():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1], [0.5]) == 0.25


def test_macro_f1_perfect():
    assert macro_f1(["x", "y", "x"], ["x", "y", "x"]) == 1.0


def test_macro_f1_one_class_wrong():
    # x: tp=1 (idx0), fp=1 (idx1's "y" predicted as "x"), fn=0 -> f1_x = 2/3
    # y: tp=0, fp=0, fn=1 (idx1 never predicted as "y") -> f1_y = 0
    # macro = (2/3 + 0) / 2 = 1/3
    assert macro_f1(["x", "y"], ["x", "x"]) == pytest.approx(1 / 3)


def test_macro_f1_ignores_labels_missing_from_truth():
    assert macro_f1(["x", "x"], ["x", "y"]) == pytest.approx(1 / 3)


def test_ece_zero_when_perfectly_calibrated_bins():
    # bin [0.2,0.3): mean pred 0.25, empirical 0.25 (1 of 4); bin [0.7,0.8): 0.75, 3 of 4
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_weights_bins_by_count():
    p = np.array([0.9] * 3 + [0.1] * 1)   # bin 9: pred .9, empirical 2/3; bin 1: pred .1, empirical 0
    y = np.array([1, 1, 0, 0])
    expected = 0.75 * abs(0.9 - 2 / 3) + 0.25 * abs(0.1 - 0.0)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(expected)


def test_joint_outcome_log_loss_composes_winner_method_round():
    method_classes = ["ko_tko", "submission", "decision"]
    round_classes = ["1", "2", "3", "45"]
    p_winner = np.array([0.8, 0.4])
    method = np.array([[0.5, 0.2, 0.3], [0.1, 0.1, 0.8]])
    rounds = np.array([[0.6, 0.2, 0.1, 0.1], [0.25] * 4])
    y_winner = np.array([1, 0])
    y_method = np.array(["ko_tko", "decision"], dtype=object)
    y_round = np.array(["1", None], dtype=object)
    # row 0: A wins by KO in R1 -> 0.8 * 0.5 * 0.6 ; row 1: B wins by decision -> 0.6 * 0.8
    expected = -np.mean([np.log(0.8 * 0.5 * 0.6), np.log(0.6 * 0.8)])
    got = joint_outcome_log_loss(
        y_winner, y_method, y_round, p_winner, method, rounds, method_classes, round_classes
    )
    assert got == pytest.approx(expected)


def test_joint_outcome_log_loss_skips_unknown_method():
    got = joint_outcome_log_loss(
        np.array([1, 1]), np.array([None, "decision"], dtype=object), np.array([None, None], dtype=object),
        np.array([0.5, 0.5]), np.array([[0.2, 0.2, 0.6]] * 2), np.array([[0.25] * 4] * 2),
        ["ko_tko", "submission", "decision"], ["1", "2", "3", "45"],
    )
    assert got == pytest.approx(-np.log(0.5 * 0.6))


# --- reliability_curve ------------------------------------------------------


def test_reliability_curve_reconstructs_the_ece_it_summarises():
    """The gate SP2.2 applies compares two ECEs; this is the table each of them
    collapses, so it has to collapse back to exactly the same number."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=500)
    y = (rng.uniform(size=500) < p).astype(float)
    for n_bins in (5, 10, 15, 20):
        rows = reliability_curve(y, p, n_bins=n_bins)
        assert len(rows) == n_bins
        rebuilt = sum(r["weight"] * abs(r["gap"]) for r in rows if r["n"])
        assert rebuilt == pytest.approx(expected_calibration_error(y, p, n_bins=n_bins))
        assert sum(r["n"] for r in rows) == len(y)


def test_reliability_curve_keeps_empty_bins_so_two_curves_stay_row_comparable():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    p = np.array([0.42, 0.44, 0.46, 0.48])  # all in bin 4 of 10
    rows = reliability_curve(y, p, n_bins=10)
    assert [r["n"] for r in rows] == [0, 0, 0, 0, 4, 0, 0, 0, 0, 0]
    empty = rows[0]
    assert empty["mean_pred"] is None and empty["empirical_rate"] is None and empty["gap"] is None
    filled = rows[4]
    assert (filled["lo"], filled["hi"]) == (0.4, 0.5)
    assert filled["weight"] == pytest.approx(1.0)
    assert filled["mean_pred"] == pytest.approx(0.45)
    assert filled["empirical_rate"] == pytest.approx(0.75)
    assert filled["gap"] == pytest.approx(0.45 - 0.75)


def test_reliability_curve_bins_the_endpoints_the_way_ece_does():
    """0.0 lands in the first bin and 1.0 in the last -- `np.digitize` against
    the interior edges, clipped, exactly as expected_calibration_error does."""
    rows = reliability_curve([0.0, 1.0], [0.0, 1.0], n_bins=10)
    assert rows[0]["n"] == 1 and rows[-1]["n"] == 1
    assert sum(r["n"] for r in rows) == 2


def test_reliability_curve_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        reliability_curve([1.0, 0.0], [0.5])


# --------------------------------------------------------------------------
# joint cell scoring (SP3 Task 3)
# --------------------------------------------------------------------------

METHOD = ["ko_tko", "submission", "decision"]
ROUND = ["1", "2", "3", "45"]


def _outcomes():
    y_w = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    y_m = np.array(["ko_tko", "decision", "submission", "ko_tko", "decision"], dtype=object)
    y_r = np.array(["1", None, "45", "3", None], dtype=object)
    return y_w, y_m, y_r


def _marginals(seed=0):
    rng = np.random.default_rng(seed)
    p_w = rng.uniform(0.2, 0.8, size=5)
    method = rng.dirichlet(np.ones(3), size=5)
    rounds = rng.dirichlet(np.ones(4), size=5)
    return p_w, method, rounds


def _composed_cells(p_w, method, rounds):
    """The independence composition written out as a flat joint cell vector.

    Multiplication order matches `joint_outcome_log_loss` exactly (`p_w`,
    then the method term, then the round term), so the two paths agree bit
    for bit rather than merely to 4 dp.
    """
    n = len(p_w)
    cells = np.zeros((n, 18))
    for i in range(n):
        for w, p_corner in enumerate((p_w[i], 1.0 - p_w[i])):
            for m in range(2):
                for r in range(4):
                    cells[i, w * 8 + m * 4 + r] = p_corner * method[i, m] * rounds[i, r]
            cells[i, 16 + w] = p_corner * method[i, 2]
    return cells


def test_joint_cells_from_independent_marginals_score_exactly_the_composed_path():
    y_w, y_m, y_r = _outcomes()
    p_w, method, rounds = _marginals()
    composed = joint_outcome_log_loss(y_w, y_m, y_r, p_w, method, rounds, METHOD, ROUND)
    direct = joint_cell_log_loss(y_w, y_m, y_r, _composed_cells(p_w, method, rounds), METHOD, ROUND)
    assert direct == composed


def test_joint_cell_index_layout():
    # finishes: winner-major, then method, then round; decisions last
    assert joint_cell_index(0, 0, 0, METHOD, ROUND) == 0
    assert joint_cell_index(0, 1, 3, METHOD, ROUND) == 7
    assert joint_cell_index(1, 0, 0, METHOD, ROUND) == 8
    assert joint_cell_index(1, 1, 3, METHOD, ROUND) == 15
    assert joint_cell_index(0, None, None, METHOD, ROUND) == 16
    assert joint_cell_index(1, None, None, METHOD, ROUND) == 17


def test_joint_cell_log_loss_reads_the_realised_cell():
    y_w = np.array([1.0])
    y_m = np.array(["submission"], dtype=object)
    y_r = np.array(["2"], dtype=object)
    cells = np.full((1, 18), 0.01)
    cells[0, joint_cell_index(0, 1, 1, METHOD, ROUND)] = 0.4
    assert joint_cell_log_loss(y_w, y_m, y_r, cells, METHOD, ROUND) == pytest.approx(-np.log(0.4))


def test_joint_cell_log_loss_uses_the_corner_b_block_when_b_won():
    y_w = np.array([0.0])
    y_m = np.array(["ko_tko"], dtype=object)
    y_r = np.array(["3"], dtype=object)
    cells = np.full((1, 18), 0.01)
    cells[0, joint_cell_index(1, 0, 2, METHOD, ROUND)] = 0.3
    assert joint_cell_log_loss(y_w, y_m, y_r, cells, METHOD, ROUND) == pytest.approx(-np.log(0.3))


def test_joint_cell_log_loss_skips_unscorable_rows_like_the_composed_path():
    y_w = np.array([1.0, 1.0, 1.0])
    y_m = np.array([None, "ko_tko", "ko_tko"], dtype=object)
    y_r = np.array(["1", None, "1"], dtype=object)
    cells = np.full((3, 18), 0.05)
    cells[2, joint_cell_index(0, 0, 0, METHOD, ROUND)] = 0.5
    assert joint_cell_log_loss(y_w, y_m, y_r, cells, METHOD, ROUND) == pytest.approx(-np.log(0.5))


def test_joint_cell_log_loss_is_finite_on_a_zero_cell():
    y_w, y_m, y_r = np.array([1.0]), np.array(["ko_tko"], dtype=object), np.array(["1"], dtype=object)
    assert np.isfinite(joint_cell_log_loss(y_w, y_m, y_r, np.zeros((1, 18)), METHOD, ROUND))


def test_realised_zero_mass_fraction_counts_only_scorable_rows():
    y_w = np.array([1.0, 1.0, 1.0])
    y_m = np.array(["ko_tko", "ko_tko", None], dtype=object)
    y_r = np.array(["1", "2", None], dtype=object)
    zero = np.zeros((3, 18), dtype=bool)
    zero[0, joint_cell_index(0, 0, 0, METHOD, ROUND)] = True
    # the third row is unscorable, so the denominator is 2 and not 3
    assert realised_zero_mass_fraction(y_w, y_m, y_r, zero, METHOD, ROUND) == pytest.approx(0.5)
