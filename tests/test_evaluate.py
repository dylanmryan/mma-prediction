import numpy as np
import pytest

from mma.evaluate import (
    accuracy,
    brier_score,
    expected_calibration_error,
    joint_outcome_log_loss,
    log_loss,
    macro_f1,
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
