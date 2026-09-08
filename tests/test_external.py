"""The external per-fighter table and the join that turns it into block state.

The join is by ufcstats `fighter_id` and nothing else -- the derived table
carries no name column at all, which is the structural version of the
"never match by name" rule. Everything a fighter is missing from comes back
NaN with `external_missing` set, so a half-matched corner is visible in the
`external_missing` walk-forward slice rather than silently imputed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma import external

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture
def table() -> pd.DataFrame:
    """Two clean fighters plus one whose recorded pro debut is a source error."""
    return pd.DataFrame({
        "fighter_id": ["aaa", "bbb", "bad"],
        "pre_ufc_wins": [10, 3, 5],
        "pre_ufc_losses": [2, 1, 0],
        "pre_ufc_finish_rate": [0.5, 1.0, 0.4],
        "pre_ufc_finish_loss_rate": [0.5, 0.0, np.nan],
        "pre_ufc_avg_opp_wins": [4.0, 2.5, 3.0],
        "pro_debut_date": pd.to_datetime(["2010-01-01", "2012-06-01", "2020-01-01"]),
        "first_ufc_date": pd.to_datetime(["2015-01-01", "2016-01-01", "2015-01-01"]),
        "nationality": pd.array(["Brazil", "Brazil", "Russia"], dtype="string"),
        "gym_id": pd.array(["1-x", "2-y", None], dtype="string"),
    })


def _side(fighter_ids, dates, **extra) -> pd.DataFrame:
    return pd.DataFrame({
        "fighter_id": fighter_ids,
        "date": pd.to_datetime(dates),
        **extra,
    })


def test_unmatched_fighter_is_all_nan_and_flagged(table):
    side = external.attach(_side(["aaa", "nope"], ["2018-01-01", "2018-01-01"]), table)
    assert side["external_missing"].tolist() == [False, True]
    for column in external.NUMERIC_STATE_KEYS + ("days_since_pro_debut",):
        assert pd.notna(side[column].iloc[0]), column
        assert pd.isna(side[column].iloc[1]), column
    assert pd.isna(side["nationality"].iloc[1])


def test_pro_debut_after_first_ufc_bout_is_flagged_not_negative(table):
    """A recorded debut later than the fighter's first UFC bout is a source
    data error: the pre-UFC window it implies is wrong, so the whole fighter
    is dropped rather than emitting a negative duration."""
    side = external.attach(_side(["bad"], ["2018-01-01"]), table)
    assert bool(side["external_missing"].iloc[0])
    assert pd.isna(side["days_since_pro_debut"].iloc[0])
    assert pd.isna(side["pre_ufc_wins"].iloc[0])


def test_days_since_pro_debut_is_relative_to_the_fight_date(table):
    """Two fights of the SAME fighter must get different values -- the column
    is a duration measured from a fixed date, not a stored constant."""
    side = external.attach(_side(["aaa", "aaa"], ["2016-01-01", "2020-01-01"]), table)
    assert side["days_since_pro_debut"].tolist() == [
        (pd.Timestamp("2016-01-01") - pd.Timestamp("2010-01-01")).days,
        (pd.Timestamp("2020-01-01") - pd.Timestamp("2010-01-01")).days,
    ]


def test_the_join_never_falls_back_to_names(table):
    """The derived table has no name column, so there is nothing to fall back
    to; a row whose id does not match stays unmatched however well the name
    would have."""
    assert "name" not in table.columns
    side = _side(["unmatched-id"], ["2018-01-01"], name=["Jon Jones"])
    assert bool(external.attach(side, table)["external_missing"].iloc[0])


def test_attach_preserves_row_order_and_length(table):
    side = _side(["bbb", "nope", "aaa"], ["2018-01-01", "2019-01-01", "2020-01-01"])
    attached = external.attach(side, table)
    assert attached["fighter_id"].tolist() == ["bbb", "nope", "aaa"]
    assert attached["pre_ufc_wins"].tolist()[0] == 3


def test_state_for_matches_the_attach_path(table):
    """Serving and training must agree value-for-value on the same fighter."""
    as_of = pd.Timestamp("2019-05-05")
    served = external.state_for("aaa", as_of, table)
    trained = external.attach(_side(["aaa"], [as_of]), table).iloc[0]
    for key in external.STATE_KEYS:
        assert served[key] == trained[key], key


def test_state_for_unknown_fighter_is_missing(table):
    state = external.state_for("nope", pd.Timestamp("2019-01-01"), table)
    assert state["external_missing"] is True
    assert all(pd.isna(state[key]) for key in external.NUMERIC_STATE_KEYS)
    assert pd.isna(state["days_since_pro_debut"])


def test_fight_context_scalars(table):
    as_of = pd.Timestamp("2019-01-01")
    a = external.state_for("aaa", as_of, table)
    b = external.state_for("bbb", as_of, table)
    missing = external.state_for("nope", as_of, table)
    assert external.fight_context(a, b) == {"external_missing": False, "same_country": True}
    assert external.fight_context(a, missing) == {
        "external_missing": True, "same_country": False
    }


def test_fight_context_is_vectorised_and_matches_the_scalar_path(table):
    dates = ["2019-01-01", "2019-01-01"]
    left = external.attach(_side(["aaa", "aaa"], dates), table)
    right = external.attach(_side(["bbb", "nope"], dates), table)
    context = external.fight_context(left, right)
    assert list(context["external_missing"]) == [False, True]
    assert list(context["same_country"]) == [True, False]


def test_every_state_key_the_block_declares_is_one_this_module_emits():
    """`feature_blocks` reads its state from this module; a key the loader does
    not emit would raise at build time, which is the point."""
    from mma.feature_blocks import BLOCKS, EXTERNAL_BLOCK

    block = BLOCKS[EXTERNAL_BLOCK]
    declared = [key for key, _ in block.differentials] + list(block.booleans)
    assert set(declared) <= set(external.STATE_KEYS)


def test_missingness_is_fight_level_only_never_per_corner():
    """Regression guard for a measured leak (SP2 Task 11).

    Membership of the snapshot's cross-source fighter mapping is a function of
    how long a fighter's UFC career turned out to be: among 2013-2022
    debutants 24% of the one-and-done fighters are in it against 100% of those
    with 11+ bouts. A per-corner `external_missing_a`/`_b` pair therefore
    encodes which corner went on to have a career -- the unmapped corner loses
    76% of the time overall and 90% of the time in the debut slice, and the
    walk-forward result flips from -0.014 on the historical folds to +0.049 on
    2025 once missingness starts meaning "debuted after the snapshot" instead.
    The row-level OR is symmetric in the corners, so it keeps the missingness
    visible to the slice without saying who wins.
    """
    from mma.feature_blocks import BLOCKS, EXTERNAL_BLOCK, columns_of

    block = BLOCKS[EXTERNAL_BLOCK]
    columns = columns_of(block)
    assert "external_missing" in block.fight_level
    assert "external_missing_a" not in columns and "external_missing_b" not in columns
    assert not any(column.endswith(("_a", "_b")) for column in columns)


@pytest.mark.skipif(
    not (external.DEFAULT_PATH).exists(), reason="data/external not built"
)
def test_committed_table_drops_source_errors_and_carries_no_names():
    raw = pd.read_parquet(external.DEFAULT_PATH)
    loaded = external.load_table()
    assert "name" not in raw.columns
    errors = (raw["pro_debut_date"] > raw["first_ufc_date"]).sum()
    assert errors > 0, "fixture assumption: this snapshot has source errors to drop"
    assert len(loaded) == len(raw) - errors
    assert (loaded["pro_debut_date"] <= loaded["first_ufc_date"]).all()


@pytest.mark.skipif(
    not (external.DEFAULT_PATH).exists() or not (PROCESSED / "fights.parquet").exists(),
    reason="data not built",
)
def test_no_negative_durations_over_the_real_fights_table():
    """The guard's purpose, checked end to end: every UFC fight of a matched
    fighter is on or after their recorded pro debut."""
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    side = external.attach(
        fights[["fighter_a_id", "date"]].rename(columns={"fighter_a_id": "fighter_id"})
    )
    days = side["days_since_pro_debut"].dropna()
    assert len(days) > 0
    assert (days >= 0).all()
