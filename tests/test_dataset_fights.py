import pandas as pd
import pytest

from mma.dataset import build_fights


def _raw_master():
    return pd.DataFrame(
        {
            "fight_id": ["f2", "f1", "f3", "f4"],
            "event_id": ["e2", "e1", "e3", "e3"],
            "event_date": ["2017-07-29", "2016-11-12", "2019-03-02", "2019-03-02"],
            "event_location": ["Anaheim, California, USA"] * 4,
            "weight_class": ["Light Heavyweight", "Lightweight", "Bout", "Women's Strawweight"],
            "title_fight": [1, 0, 0, 0],
            "r_fighter_id": ["jj", "cm", "aa", "ww"],
            "b_fighter_id": ["dc", "ed", "bb", "vv"],
            "winner_id": ["jj", None, None, "vv"],
            "result_status": ["win", "draw", "no_contest", "win"],
            "method": ["KO/TKO", "Decision - Majority", "Overturned", "Submission"],
            "finish_round": [3, 3, 2, 1],
            "finish_time": ["4:20", "5:00", "2:32", "0:45"],
            "time_format": ["5 Rnd (5-5-5-5-5)", "3 Rnd (5-5-5)", "No Time Limit", "3 Rnd (5-5-5)"],
            "referee": ["Herb Dean", "Marc Goddard", None, "Jason Herzog"],
        }
    )


def test_schema_order_and_sorting():
    fights = build_fights(_raw_master())
    assert list(fights.columns) == [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight", "duration_sec",
        "referee", "event_id", "location",
    ]
    assert list(fights["fight_id"]) == ["f1", "f2", "f3", "f4"]  # date-sorted, stable


def test_finish_fight_values():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f2"].iloc[0]
    assert row["fighter_a_id"] == "jj" and row["fighter_b_id"] == "dc"
    assert row["winner"] == "a"
    assert row["method"] == "ko_tko" and row["method_raw"] == "KO/TKO"
    assert row["finish_round"] == 3
    assert row["scheduled_rounds"] == 5
    assert row["weight_class"] == "Light Heavyweight"
    assert bool(row["title_fight"]) is True
    assert row["duration_sec"] == (3 - 1) * 300 + 260
    assert row["referee"] == "Herb Dean"
    assert row["event_id"] == "e2"
    assert row["location"] == "Anaheim, California, USA"


def test_draw_and_decision_have_no_finish_round():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f1"].iloc[0]
    assert row["winner"] == "draw"
    assert row["method"] == "decision" and row["decision_subtype"] == "majority"
    assert pd.isna(row["finish_round"])
    assert row["duration_sec"] == 3 * 300  # went the distance


def test_no_contest_no_time_limit_and_noise_weight_class():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f3"].iloc[0]
    assert row["winner"] == "nc"
    assert pd.isna(row["method"])
    assert pd.isna(row["scheduled_rounds"])
    assert pd.isna(row["weight_class"])  # "Bout" carries no class
    assert row["duration_sec"] == (2 - 1) * 300 + 152  # raw-round fallback
    assert pd.isna(row["referee"])


def test_winner_b():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f4"].iloc[0]
    assert row["winner"] == "b"
    assert row["method"] == "submission"
    assert row["finish_round"] == 1
    assert row["duration_sec"] == 45
    assert row["weight_class"] == "Women's Strawweight"


def test_win_status_with_unmatched_winner_id_is_nc():
    raw = _raw_master()
    raw.loc[0, "winner_id"] = "someone-else"
    fights = build_fights(raw)
    assert fights[fights["fight_id"] == "f2"].iloc[0]["winner"] == "nc"


def test_duplicate_fight_ids_rejected():
    raw = _raw_master()
    raw.loc[1, "fight_id"] = "f2"
    with pytest.raises(ValueError, match="fight_id"):
        build_fights(raw)


def test_missing_corner_id_rejected():
    raw = _raw_master()
    raw.loc[0, "b_fighter_id"] = None
    with pytest.raises(ValueError, match="missing corner"):
        build_fights(raw)


def test_string_dtypes():
    fights = build_fights(_raw_master())
    for column in ("winner", "method", "method_raw", "decision_subtype",
                   "weight_class", "referee", "event_id", "location"):
        assert fights[column].dtype == "string"
