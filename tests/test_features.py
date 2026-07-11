import hashlib

import pandas as pd

from mma.features import build_features, swap_corner


def test_swap_is_deterministic_md5_parity():
    for fight_id in ("abc", "f1", "20170729-x-vs-y"):
        expected = int(hashlib.md5(fight_id.encode()).hexdigest(), 16) % 2 == 1
        assert swap_corner(fight_id) is expected


def _tables():
    fights = pd.DataFrame(
        {
            "fight_id": ["f1"],
            "date": pd.to_datetime(["2021-06-01"]),
            "fighter_a_id": ["x"],
            "fighter_b_id": ["y"],
            "winner": ["a"],
            "method": ["ko_tko"],
            "finish_round": pd.array([2], dtype="Int64"),
            "scheduled_rounds": pd.array([3], dtype="Int64"),
            "weight_class": ["Lightweight"],
            "title_fight": [False],
            "duration_sec": [500.0],
        }
    )
    fighters = pd.DataFrame(
        {
            "fighter_id": ["x", "y"],
            "name": ["X", "Y"],
            "height_cm": [180.0, 175.0],
            "reach_cm": [183.0, None],
            "stance": ["Southpaw", "Orthodox"],
            "dob": pd.to_datetime(["1990-06-01", None]),
        }
    )
    ratings = pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "corner": ["a", "b"],
            "fighter_id": ["x", "y"],
            "pre_overall": [1550.0, 1500.0],
            "pre_striking": [1540.0, 1500.0],
            "pre_grappling": [1510.0, 1500.0],
            "pre_fights": [3, 0],
        }
    )
    history = pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "corner": ["a", "b"],
            "fighter_id": ["x", "y"],
            "career_fights": [3, 0],
            "career_wins": [2.0, 0.0],
            "career_win_rate": [2 / 3, None],
            "career_finish_rate": [0.5, None],
            "kd_pf": [0.3, None],
            "sub_att_pf": [0.7, None],
            "td_landed_pf": [1.0, None],
            "td_acc": [0.5, None],
            "td_def": [0.8, None],
            "sig_pm": [4.0, None],
            "sig_absorbed_pm": [3.0, None],
            "ctrl_share": [0.2, None],
            "streak": [2, 0],
            "days_since_last": [120.0, None],
            "last5_win_rate": [2 / 3, None],
            "last5_avg_opp_elo": [1510.0, None],
        }
    )
    return fights, fighters, ratings, history


def test_feature_row_shape_and_targets():
    features = build_features(*_tables())
    assert len(features) == 1
    row = features.iloc[0]
    swapped = row["swapped"]
    assert row["y_winner"] == (0 if swapped else 1)
    assert row["y_method"] == "ko_tko"
    assert row["y_finish_round"] == "2"
    assert row["weight_class"] == "Lightweight"
    assert row["scheduled_rounds"] == 3


def test_diffs_flip_sign_with_swap():
    features = build_features(*_tables()).iloc[0]
    sign = -1 if features["swapped"] else 1
    assert features["elo_diff"] == sign * 50.0
    assert features["height_diff"] == sign * 5.0
    assert features["career_fights_diff"] == sign * 3


def test_missing_flags_and_debut():
    features = build_features(*_tables()).iloc[0]
    # y has no reach and no dob; x has both
    if features["swapped"]:
        assert features["reach_missing_a"] and not features["reach_missing_b"]
        assert features["debut_a"] and not features["debut_b"]
    else:
        assert features["reach_missing_b"] and not features["reach_missing_a"]
        assert features["debut_b"] and not features["debut_a"]
    assert bool(features["debut_matchup"]) is True


def test_draws_and_nc_excluded():
    fights, fighters, ratings, history = _tables()
    fights.loc[0, "winner"] = "draw"
    assert len(build_features(fights, fighters, ratings, history)) == 0
