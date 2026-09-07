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
        "rounds_fought": [3, 2],
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


def test_no_round_record_nulls_stats():
    # f2 (row 0) has real target-column values in the fixture, unlike f1
    # (row 1), whose target columns are already None regardless of
    # rounds_fought -- so null it out here to make the target-column
    # assertions below non-vacuous.
    raw = _raw_master()
    raw.loc[0, "rounds_fought"] = 0
    stats = build_fight_stats(raw).set_index(["fight_id", "corner"])
    for corner in ("a", "b"):
        row = stats.loc[("f2", corner)]
        assert pd.isna(row["kd"])
        assert pd.isna(row["sig_landed"])
        assert pd.isna(row["ctrl_sec"])
        assert pd.isna(row["rev"])
        assert pd.isna(row["head_landed"])
        assert pd.isna(row["ground_attempted"])
        assert pd.notna(row["fighter_id"])
    assert stats.loc[("f2", "a"), "fighter_id"] == "jj"
    assert stats.loc[("f2", "b"), "fighter_id"] == "dc"
    f1 = stats.loc[("f1", "a")]
    assert pd.notna(f1["kd"]) and pd.notna(f1["sig_landed"])
    assert pd.notna(f1["ctrl_sec"])


def test_missing_corner_id_rejected():
    raw = _raw_master()
    raw.loc[0, "r_fighter_id"] = None
    with pytest.raises(ValueError, match="missing corner"):
        build_fight_stats(raw)
