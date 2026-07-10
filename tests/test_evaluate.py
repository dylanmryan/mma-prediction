import numpy as np
import pytest

from mma.evaluate import accuracy, brier_score, log_loss


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
