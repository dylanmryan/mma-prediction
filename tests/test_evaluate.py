import numpy as np
import pytest

from mma.evaluate import accuracy, brier_score, log_loss, macro_f1


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
