import pandas as pd
import pytest

from mma.history import build_history


def _fights():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2"],
            "date": pd.to_datetime(["2020-01-01", "2020-03-01"]),
            "fighter_a_id": ["x", "x"],
            "fighter_b_id": ["y", "z"],
            "winner": ["a", "a"],
            "method": ["ko_tko", "decision"],
            "match_time_sec": [300.0, 900.0],
        }
    )


def _stats():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "kd": [2, 0, 0, 0],
            "sig_landed": [30, 10, 50, 40],
            "td_landed": [1, 0, 2, 1],
            "td_attempted": [2, 1, 4, 2],
            "sub_att": [0, 1, 1, 0],
            "ctrl_sec": [60.0, 30.0, 300.0, 100.0],
        }
    )


def _ratings():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "fighter_id": ["x", "y", "x", "z"],
            "pre_overall": [1500.0, 1500.0, 1532.0, 1500.0],
        }
    )


def test_debut_row_is_empty_history():
    history = build_history(_fights(), _stats(), _ratings())
    x_f1 = history[(history["fight_id"] == "f1") & (history["corner"] == "a")].iloc[0]
    assert x_f1["career_fights"] == 0
    assert x_f1["streak"] == 0
    assert pd.isna(x_f1["career_win_rate"])
    assert pd.isna(x_f1["days_since_last"])
    assert pd.isna(x_f1["last5_avg_opp_elo"])


def test_second_fight_reflects_first():
    history = build_history(_fights(), _stats(), _ratings())
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_fights"] == 1
    assert x_f2["career_win_rate"] == 1.0
    assert x_f2["career_finish_rate"] == 1.0  # won by ko
    assert x_f2["streak"] == 1
    assert x_f2["days_since_last"] == 60
    assert x_f2["kd_pf"] == 2.0
    assert x_f2["td_acc"] == pytest.approx(0.5)          # 1 of 2
    assert x_f2["td_def"] == pytest.approx(1.0)          # opponent 0 of 1
    assert x_f2["sig_pm"] == pytest.approx(30 / 5.0)     # 30 landed in 5 min
    assert x_f2["sig_absorbed_pm"] == pytest.approx(10 / 5.0)
    assert x_f2["ctrl_share"] == pytest.approx(60 / 300.0)
    assert x_f2["last5_win_rate"] == 1.0
    assert x_f2["last5_avg_opp_elo"] == 1500.0


def test_loss_and_draw_semantics():
    fights = _fights()
    fights.loc[0, "winner"] = "b"     # x loses f1
    fights.loc[1, "winner"] = "draw"  # then draws f2 (still emitted to history)
    history = build_history(fights, _stats(), _ratings())
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_win_rate"] == 0.0
    assert x_f2["streak"] == -1


def test_nc_ignored():
    fights = _fights()
    fights.loc[0, "winner"] = "nc"
    history = build_history(fights, _stats(), _ratings())
    assert "f1" not in set(history["fight_id"])
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_fights"] == 0
