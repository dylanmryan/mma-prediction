import pandas as pd
import pytest

from mma.dataset import build_fight_stats

_SUFFIXES = {
    "kd": "kd", "sig_landed": "sig_landed", "sig_atmp": "sig_atmp",
    "total_str_landed": "total_str_landed", "total_str_atmp": "total_str_atmp",
    "td_success": "td_success", "td_atmp": "td_atmp", "sub_att": "sub_att",
    "rev": "rev", "ctrl_seconds": "ctrl_seconds",
}
_TARGETS = ("head", "body", "leg", "distance", "clinch", "ground")


def _raw_master():
    rows = {
        "fight_id": ["f2", "f1"],
        "r_fighter_id": ["jj", "cm"],
        "b_fighter_id": ["dc", "ed"],
    }
    base = {"r": 10, "b": 20}
    for corner in ("r", "b"):
        for i, suffix in enumerate(_SUFFIXES.values()):
            rows[f"{corner}_total_{suffix}"] = [base[corner] + i, base[corner] + i + 100]
        for j, target in enumerate(_TARGETS):
            rows[f"{corner}_total_sig_str_landed_{target}"] = [base[corner] * 10 + j, None]
            rows[f"{corner}_total_sig_str_atmp_{target}"] = [base[corner] * 10 + j + 1, None]
    return pd.DataFrame(rows)


def test_two_rows_per_fight_schema_and_order():
    stats = build_fight_stats(_raw_master())
    assert list(stats.columns[:12]) == [
        "fight_id", "fighter_id", "corner", "kd", "sig_landed", "sig_attempted",
        "total_landed", "total_attempted", "td_landed", "td_attempted",
        "sub_att", "ctrl_sec",
    ]
    assert list(stats.columns[12:]) == ["rev"] + [
        f"{target}_{kind}" for target in _TARGETS for kind in ("landed", "attempted")
    ]
    assert list(zip(stats["fight_id"], stats["corner"])) == [
        ("f1", "a"), ("f1", "b"), ("f2", "a"), ("f2", "b"),
    ]


def test_values_unpivoted_to_correct_corner():
    stats = build_fight_stats(_raw_master()).set_index(["fight_id", "corner"])
    a = stats.loc[("f2", "a")]
    b = stats.loc[("f2", "b")]
    assert a["fighter_id"] == "jj" and b["fighter_id"] == "dc"
    assert a["kd"] == 10 and b["kd"] == 20
    assert a["td_landed"] == 15 and a["td_attempted"] == 16  # td_success, td_atmp
    assert a["ctrl_sec"] == 19 and b["ctrl_sec"] == 29
    assert a["rev"] == 18
    assert a["head_landed"] == 100 and a["head_attempted"] == 101
    assert b["ground_landed"] == 205


def test_missing_stat_stays_missing():
    raw = _raw_master()
    raw.loc[0, "r_total_kd"] = None
    stats = build_fight_stats(raw).set_index(["fight_id", "corner"])
    assert pd.isna(stats.loc[("f2", "a"), "kd"])
    assert pd.isna(stats.loc[("f1", "a"), "head_landed"])


def test_duplicate_fight_ids_rejected():
    raw = _raw_master()
    raw.loc[1, "fight_id"] = "f2"
    with pytest.raises(ValueError, match="fight_id"):
        build_fight_stats(raw)
