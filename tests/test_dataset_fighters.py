import pandas as pd
import pytest

from mma.dataset import build_fighters


def _raw():
    return pd.DataFrame(
        {
            "fighter_id": ["jj", "cm", "xx"],
            "fighter_name": ["Jon Jones", " Conor McGregor ", "No Data"],
            "fighter_nick_name": ["Bones", "Notorious", None],
            "height": ["6' 4\"", "5' 9\"", None],
            "weight_lbs": [205.0, 155.0, None],
            "reach_inches": [84.5, 74.0, None],
            "stance": ["Orthodox", "Southpaw", None],
            "dob": ["1987-07-19", "1988-07-14", None],
            "slpm": [4.3, 5.3, 0.0],
            "str_acc": [57, 49, 0],
            "sapm": [2.2, 4.0, 0.0],
            "str_def": [64, 54, 0],
            "td_avg": [1.9, 0.7, 0.0],
            "td_acc": [45, 55, 0],
            "td_def": [95, 67, 0],
            "sub_avg": [0.5, 0.2, 0.0],
        }
    )


def test_schema_and_values():
    fighters = build_fighters(_raw())
    assert list(fighters.columns) == [
        "fighter_id", "name", "height_cm", "reach_cm", "stance", "dob",
    ]
    jj = fighters[fighters["fighter_id"] == "jj"].iloc[0]
    assert jj["name"] == "Jon Jones"
    assert jj["height_cm"] == pytest.approx(76 * 2.54)
    assert jj["reach_cm"] == pytest.approx(84.5 * 2.54)
    assert jj["stance"] == "Orthodox"
    assert jj["dob"] == pd.Timestamp("1987-07-19")
    cm = fighters[fighters["fighter_id"] == "cm"].iloc[0]
    assert cm["name"] == "Conor McGregor"  # stripped


def test_missing_stance_dob_height_stay_missing():
    fighters = build_fighters(_raw())
    xx = fighters[fighters["fighter_id"] == "xx"].iloc[0]
    assert pd.isna(xx["stance"]) and pd.isna(xx["dob"])
    assert pd.isna(xx["height_cm"]) and pd.isna(xx["reach_cm"])


def test_unparseable_height_is_missing():
    raw = _raw()
    raw.loc[0, "height"] = "--"
    fighters = build_fighters(raw)
    assert pd.isna(fighters[fighters["fighter_id"] == "jj"].iloc[0]["height_cm"])


def test_duplicate_ids_rejected():
    raw = _raw()
    raw.loc[1, "fighter_id"] = "jj"
    with pytest.raises(ValueError, match="duplicate fighter ids"):
        build_fighters(raw)


def test_missing_id_rejected():
    raw = _raw()
    raw.loc[1, "fighter_id"] = None
    with pytest.raises(ValueError, match="missing ids"):
        build_fighters(raw)


def test_leaky_career_columns_dropped():
    fighters = build_fighters(_raw())
    for leaky in ("slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc",
                  "td_def", "sub_avg", "weight_lbs", "fighter_nick_name"):
        assert leaky not in fighters.columns


def test_sorted_by_id():
    fighters = build_fighters(_raw())
    assert list(fighters["fighter_id"]) == ["cm", "jj", "xx"]
