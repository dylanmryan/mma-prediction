import pandas as pd
import pytest

from mma.dataset import build_bonuses, build_round_stats
from scripts.reconcile_sources import reconcile

_TARGETS = ("head", "body", "leg", "distance", "clinch", "ground")


def _raw_round():
    rows = {
        "fight_id": ["f1", "f1", "f2"],
        "round_no": [2, 1, 1],
        "r_id": ["cm", "cm", "jj"],
        "b_id": ["ed", "ed", "dc"],
    }
    for corner, base in (("r", 10), ("b", 20)):
        rows[f"{corner}_kd"] = [base, base + 1, base + 2]
        rows[f"{corner}_sig_landed"] = [base + 3, base + 4, base + 5]
        rows[f"{corner}_sig_atmp"] = [base + 6, base + 7, base + 8]
        rows[f"{corner}_total_str_landed"] = [1, 2, 3]
        rows[f"{corner}_total_str_atmp"] = [4, 5, 6]
        rows[f"{corner}_td_success"] = [0, 1, 2]
        rows[f"{corner}_td_atmp"] = [1, 2, 3]
        rows[f"{corner}_sub_att"] = [0, 0, 1]
        rows[f"{corner}_rev"] = [0, 1, 0]
        rows[f"{corner}_ctrl"] = ["1:05", "0:00", "--"]
        for j, target in enumerate(_TARGETS):
            rows[f"{corner}_sig_str_landed_{target}"] = [base * 10 + j] * 3
            rows[f"{corner}_sig_str_atmp_{target}"] = [base * 10 + j + 1] * 3
    return pd.DataFrame(rows)


def test_round_stats_schema_and_order():
    rounds = build_round_stats(_raw_round())
    assert list(rounds.columns[:5]) == ["fight_id", "round_no", "corner", "fighter_id", "kd"]
    assert list(rounds.columns[5:14]) == [
        "sig_landed", "sig_attempted", "total_landed", "total_attempted",
        "td_landed", "td_attempted", "sub_att", "ctrl_sec", "rev",
    ]
    assert list(rounds.columns[14:]) == [
        f"{target}_{kind}" for target in _TARGETS for kind in ("landed", "attempted")
    ]
    assert list(zip(rounds["fight_id"], rounds["round_no"], rounds["corner"])) == [
        ("f1", 1, "a"), ("f1", 1, "b"), ("f1", 2, "a"), ("f1", 2, "b"), ("f2", 1, "a"), ("f2", 1, "b"),
    ]


def test_round_values_and_control_parsing():
    rounds = build_round_stats(_raw_round()).set_index(["fight_id", "round_no", "corner"])
    a2 = rounds.loc[("f1", 2, "a")]
    assert a2["fighter_id"] == "cm" and a2["kd"] == 10 and a2["sig_landed"] == 13
    assert a2["ctrl_sec"] == 65
    assert rounds.loc[("f1", 1, "b"), "ctrl_sec"] == 0
    assert pd.isna(rounds.loc[("f2", 1, "a"), "ctrl_sec"])  # "--" stays missing
    assert rounds.loc[("f2", 1, "b"), "head_landed"] == 200
    assert rounds.loc[("f1", 1, "a"), "td_landed"] == 1


def test_round_dtypes():
    rounds = build_round_stats(_raw_round())
    assert rounds["fight_id"].dtype == "string" and rounds["fighter_id"].dtype == "string"
    assert rounds["corner"].dtype == "string"
    assert rounds["round_no"].dtype == "int64"


def test_duplicate_fight_round_rejected():
    raw = _raw_round()
    raw.loc[1, "round_no"] = 2
    with pytest.raises(ValueError, match="fight_id, round_no"):
        build_round_stats(raw)


def test_missing_round_no_rejected():
    raw = _raw_round()
    raw.loc[0, "round_no"] = None
    with pytest.raises(ValueError, match="fight_id, round_no"):
        build_round_stats(raw)


def test_missing_corner_id_rejected():
    raw = _raw_round()
    raw.loc[0, "b_id"] = None
    with pytest.raises(ValueError, match="missing corner"):
        build_round_stats(raw)


def test_bonuses():
    raw = pd.DataFrame({
        "fight_id": ["f2", "f1", "f1"],
        "bonus_type": ["Performance of the Night", "Fight of the Night", "Fight of the Night"],
    })
    bonuses = build_bonuses(raw)
    assert list(bonuses.columns) == ["fight_id", "bonus_type"]
    assert len(bonuses) == 2  # exact duplicate row dropped
    assert list(bonuses["fight_id"]) == ["f1", "f2"]
    assert bonuses["fight_id"].dtype == "string" and bonuses["bonus_type"].dtype == "string"


def test_reconcile_counts_and_agreement():
    old = pd.DataFrame({
        "fight_id": ["f1", "f2"], "winner": ["a", "b"], "method": ["ko_tko", "decision"],
        "finish_round": [1, None], "scheduled_rounds": [3, 3], "weight_class": ["Lightweight"] * 2,
        "fighter_a_id": ["x", "y"], "fighter_b_id": ["p", "q"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
    })
    new = pd.concat([old, pd.DataFrame({
        "fight_id": ["f3"], "winner": ["a"], "method": ["submission"], "finish_round": [2],
        "scheduled_rounds": [3], "weight_class": ["Lightweight"], "fighter_a_id": ["z"],
        "fighter_b_id": ["r"], "date": pd.to_datetime(["2021-03-01"]),
    })], ignore_index=True)
    new.loc[1, "winner"] = "a"  # one disagreement
    report = reconcile(old, new)
    assert report["n_overlap"] == 2 and report["n_added_by_new"] == 1 and report["n_dropped_by_new"] == 0
    assert report["agreement"]["winner"] == 0.5
    assert report["agreement"]["finish_round"] == 1.0  # NaN == NaN counts as agreement
    assert report["added_by_year"] == {2021: 1}


def test_reconcile_counts_one_sided_nulls_as_disagreement():
    old = pd.DataFrame({
        "fight_id": ["f1", "f2"], "winner": ["a", "b"],
        "method": pd.array(["ko_tko", "decision"], dtype="string"),
        "finish_round": [1, 3], "scheduled_rounds": [3, 3], "weight_class": ["Lightweight"] * 2,
        "fighter_a_id": ["x", "y"], "fighter_b_id": ["p", "q"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
    })
    new = old.copy()
    new["method"] = pd.array(["ko_tko", pd.NA], dtype="string")
    report = reconcile(old, new)
    assert report["agreement"]["method"] == 0.5
