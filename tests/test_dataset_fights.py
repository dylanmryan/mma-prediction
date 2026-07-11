import pandas as pd
import pytest

from mma.dataset import build_fights


def _raw_fights():
    return pd.DataFrame(
        {
            "fight_id": ["f2", "f1", "f3"],
            "date": ["2017/07/29", "2016/11/12", "2019/03/02"],
            "r_id": ["jj", "cm", "aa"],
            "b_id": ["dc", "ed", "bb"],
            "winner": ["Jon Jones", None, None],
            "winner_id": ["jj", None, None],
            "method": ["KO/TKO", "Decision - Majority", "Overturned"],
            "finish_round": [3, 3, 2],
            "total_rounds": [5.0, 3.0, None],
            "division": ["light heavyweight", "lightweight", "6 tournament"],
            "title_fight": [1, 0, 0],
            "match_time_sec": [260.0, 900.0, 452.0],
        }
    )


def test_schema_order_and_sorting():
    fights = build_fights(_raw_fights())
    assert list(fights.columns) == [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight", "duration_sec",
    ]
    assert list(fights["fight_id"]) == ["f1", "f2", "f3"]  # date-sorted


def test_finish_fight_values():
    fights = build_fights(_raw_fights())
    row = fights[fights["fight_id"] == "f2"].iloc[0]
    assert row["fighter_a_id"] == "jj" and row["fighter_b_id"] == "dc"
    assert row["winner"] == "a"
    assert row["method"] == "ko_tko"
    assert row["method_raw"] == "KO/TKO"
    assert row["finish_round"] == 3
    assert row["scheduled_rounds"] == 5
    assert row["weight_class"] == "Light Heavyweight"
    assert bool(row["title_fight"]) is True
    assert pd.Timestamp(row["date"]) == pd.Timestamp("2017-07-29")


def test_draw_and_decision_have_no_finish_round():
    fights = build_fights(_raw_fights())
    row = fights[fights["fight_id"] == "f1"].iloc[0]
    assert row["winner"] == "draw"
    assert row["method"] == "decision"
    assert row["decision_subtype"] == "majority"
    assert pd.isna(row["finish_round"])


def test_no_contest_and_noise_division():
    fights = build_fights(_raw_fights())
    row = fights[fights["fight_id"] == "f3"].iloc[0]
    assert row["winner"] == "nc"
    assert pd.isna(row["method"])
    assert pd.isna(row["weight_class"])
    assert pd.isna(row["scheduled_rounds"])
    assert pd.isna(row["finish_round"])


def test_winner_b_and_unmatched_winner_id():
    raw = _raw_fights()
    raw.loc[0, "winner_id"] = "dc"
    fights = build_fights(raw)
    assert fights[fights["fight_id"] == "f2"].iloc[0]["winner"] == "b"
    raw.loc[0, "winner_id"] = "zz"
    fights = build_fights(raw)
    assert fights[fights["fight_id"] == "f2"].iloc[0]["winner"] == "nc"


def test_other_method_null_winner_is_draw():
    raw = _raw_fights()
    raw.loc[0, "winner"] = None
    raw.loc[0, "winner_id"] = None
    raw.loc[0, "method"] = "Other"
    fights = build_fights(raw)
    assert fights[fights["fight_id"] == "f2"].iloc[0]["winner"] == "draw"


def test_duplicate_fight_ids_rejected():
    raw = pd.concat([_raw_fights(), _raw_fights().iloc[[0]]])
    with pytest.raises(ValueError):
        build_fights(raw)


def test_duration_derived():
    fights = build_fights(_raw_fights())
    assert "match_time_sec" not in fights.columns

    # f2: KO/TKO in round 3, last-round clock 260s -> (3-1)*300 + 260 = 860
    row_f2 = fights[fights["fight_id"] == "f2"].iloc[0]
    assert row_f2["duration_sec"] == 860.0

    # f1: 3-round decision (went the distance) -> 3 * 300 = 900
    row_f1 = fights[fights["fight_id"] == "f1"].iloc[0]
    assert row_f1["duration_sec"] == 900.0

    # f3: Overturned -> not a finish (finish_round output is NA) and no
    # scheduled_rounds on record, so falls back to the RAW finish_round
    # column (2) and raw match_time_sec (452.0): (2-1)*300 + 452 = 752
    row_f3 = fights[fights["fight_id"] == "f3"].iloc[0]
    assert row_f3["duration_sec"] == 752.0
