import pandas as pd
import pytest

from mma.dataset import build_fight_stats


def _raw_fights():
    return pd.DataFrame(
        {
            "fight_id": ["f1"],
            "r_id": ["jj"],
            "b_id": ["dc"],
            "r_kd": [1.0], "b_kd": [0.0],
            "r_sig_str_landed": [58.0], "b_sig_str_landed": [44.0],
            "r_sig_str_atmpted": [92.0], "b_sig_str_atmpted": [96.0],
            "r_total_str_landed": [70.0], "b_total_str_landed": [61.0],
            "r_total_str_atmpted": [105.0], "b_total_str_atmpted": [115.0],
            "r_td_landed": [1.0], "b_td_landed": [0.0],
            "r_td_atmpted": [2.0], "b_td_atmpted": [1.0],
            "r_sub_att": [0.0], "b_sub_att": [1.0],
            "r_ctrl": [130.0], "b_ctrl": [None],
        }
    )


def test_two_rows_per_fight_schema_and_order():
    stats = build_fight_stats(_raw_fights())
    assert list(stats.columns) == [
        "fight_id", "fighter_id", "corner", "kd", "sig_landed", "sig_attempted",
        "total_landed", "total_attempted", "td_landed", "td_attempted",
        "sub_att", "ctrl_sec",
    ]
    assert len(stats) == 2
    assert list(stats["corner"]) == ["a", "b"]


def test_values_unpivoted_to_correct_corner():
    stats = build_fight_stats(_raw_fights())
    a = stats[stats["corner"] == "a"].iloc[0]
    b = stats[stats["corner"] == "b"].iloc[0]
    assert a["fighter_id"] == "jj" and b["fighter_id"] == "dc"
    assert a["kd"] == 1 and b["kd"] == 0
    assert a["sig_landed"] == 58 and a["sig_attempted"] == 92
    assert b["sig_landed"] == 44 and b["sig_attempted"] == 96
    assert a["td_landed"] == 1 and a["td_attempted"] == 2
    assert b["sub_att"] == 1
    assert a["ctrl_sec"] == 130


def test_missing_stat_stays_missing():
    stats = build_fight_stats(_raw_fights())
    b = stats[stats["corner"] == "b"].iloc[0]
    assert pd.isna(b["ctrl_sec"])


def test_multiple_fights_sorted():
    raw = pd.concat(
        [_raw_fights(), _raw_fights().assign(fight_id="f0")], ignore_index=True
    )
    stats = build_fight_stats(raw)
    assert list(stats["fight_id"]) == ["f0", "f0", "f1", "f1"]
    assert list(stats["corner"]) == ["a", "b", "a", "b"]


def test_duplicate_fight_ids_rejected():
    raw = pd.concat([_raw_fights(), _raw_fights()], ignore_index=True)
    with pytest.raises(ValueError):
        build_fight_stats(raw)
