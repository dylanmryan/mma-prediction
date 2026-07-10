import pytest

from mma.elo import EloParams, expected_score, striking_share


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)


def test_expected_score_400_points():
    # +400 rating difference -> 10:1 odds
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)


def test_params_defaults():
    params = EloParams()
    assert params.k_early == 40.0
    assert params.k_late == 24.0
    assert params.early_fights == 5
    assert params.finish_bonus == 1.2


def test_striking_share_pure_striking():
    share = striking_share(sig_landed=100, td_landed=0, sub_att=0, ctrl_sec=0)
    assert share == 1.0


def test_striking_share_balanced():
    # 50 sig strikes vs 5 takedowns * 5 + 25 ctrl minutes * 1 = 50 grappling units
    share = striking_share(sig_landed=50, td_landed=5, sub_att=0, ctrl_sec=25 * 60)
    assert share == pytest.approx(0.5)


def test_striking_share_missing_stats_neutral():
    assert striking_share(sig_landed=None, td_landed=None, sub_att=None, ctrl_sec=None) == 0.5
    assert striking_share(sig_landed=0, td_landed=0, sub_att=0, ctrl_sec=0) == 0.5
