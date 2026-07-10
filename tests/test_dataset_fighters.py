import pandas as pd
import pytest

from mma.dataset import build_fighters


def _raw_fighters():
    return pd.DataFrame(
        {
            "id": ["07f72a2a7591b409", "8f382b3baa954d2a"],
            "name": ["Jon Jones", "Amanda Nunes"],
            "nick_name": ["Bones", "The Lioness"],
            "wins": [27, 23],
            "losses": [1, 5],
            "draws": [0, 0],
            "height": [193.04, 172.72],
            "weight": [113.4, 61.23],
            "reach": [215.9, 175.26],
            "stance": ["Orthodox", None],
            "dob": ["Jul 19, 1987", None],
            "splm": [4.29, 4.9],
            "str_acc": [57.0, 52.0],
            "sapm": [2.22, 2.77],
            "str_def": [64.0, 61.0],
            "td_avg": [1.93, 2.05],
            "td_avg_acc": [45.0, 51.0],
            "td_def": [95.0, 80.0],
            "sub_avg": [0.5, 0.3],
        }
    )


def test_schema_and_values():
    fighters = build_fighters(_raw_fighters())
    assert list(fighters.columns) == [
        "fighter_id", "name", "height_cm", "reach_cm", "stance", "dob",
    ]
    jones = fighters[fighters["name"] == "Jon Jones"].iloc[0]
    assert jones["fighter_id"] == "07f72a2a7591b409"
    assert jones["height_cm"] == 193.04
    assert jones["reach_cm"] == 215.9
    assert jones["stance"] == "Orthodox"
    assert pd.Timestamp(jones["dob"]) == pd.Timestamp("1987-07-19")


def test_missing_stance_and_dob_stay_missing():
    fighters = build_fighters(_raw_fighters())
    nunes = fighters[fighters["name"] == "Amanda Nunes"].iloc[0]
    assert pd.isna(nunes["stance"])
    assert pd.isna(nunes["dob"])


def test_duplicate_ids_rejected():
    raw = pd.concat([_raw_fighters(), _raw_fighters().iloc[[0]]])
    try:
        build_fighters(raw)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_missing_id_rejected():
    raw = _raw_fighters()
    raw.loc[0, "id"] = None
    with pytest.raises(ValueError):
        build_fighters(raw)


def test_missing_name_stays_missing():
    raw = _raw_fighters()
    raw.loc[0, "name"] = None
    fighters = build_fighters(raw)
    assert fighters["name"].isna().sum() == 1


def test_leaky_career_columns_dropped():
    fighters = build_fighters(_raw_fighters())
    for leaky in ("wins", "losses", "splm", "td_avg"):
        assert leaky not in fighters.columns
