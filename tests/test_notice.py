"""The per-corner short-notice / missed-weight table and its feature state.

The source (Bet MMA, via the `ehan03/jds-mma-data` snapshot) records only the
fighters who WERE late replacements or who DID miss weight. On its own that
makes "no row" ambiguous between "full camp" and "nobody looked". What
disambiguates it is the bout universe: Bet MMA covers its own event list
(2013-04-20 .. 2024-12-14) bout by bout, so within a bout the source carries,
absence of a replacement row is an observation ("full camp"), and outside it
absence is ignorance.

`scripts/build_external.py` therefore emits one row per CORNER of every
OBSERVED fight -- both corners or neither -- so membership of
`fight_notice.parquet` is the three-state boundary: present means observed
(`notice_days` NaN there means "not a late replacement"), absent means
unknown. That symmetry is also what keeps the block clear of the per-corner
selection effect that sank the first `external` attempt: see
`test_an_unknown_fight_carries_exactly_one_value_tuple`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma import notice

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture
def table() -> pd.DataFrame:
    """Two observed fights: one clean, one with a replacement and a miss."""
    return pd.DataFrame({
        "fight_id": ["f1", "f1", "f2", "f2"],
        "fighter_id": ["aaa", "bbb", "aaa", "ccc"],
        # NaN is "observed, and it did not happen": not a late replacement /
        # made weight. Absence of the ROW is what means "unknown".
        "notice_days": [np.nan, np.nan, 5.0, np.nan],
        "missed_weight": [False, False, False, True],
        "missed_weight_over_lbs": [np.nan, np.nan, np.nan, 3.5],
    })


def _side(fight_ids, fighter_ids) -> pd.DataFrame:
    return pd.DataFrame({"fight_id": fight_ids, "fighter_id": fighter_ids})


def test_an_observed_full_camp_corner_is_zero_not_nan(table):
    side = notice.attach(_side(["f1"], ["aaa"]), table)
    row = side.iloc[0]
    assert row["notice_missing"] == False  # noqa: E712 -- numpy bool
    assert row["notice_shortfall_days"] == 0.0
    assert row["missed_weight_over_lbs"] == 0.0
    assert not row["short_notice_7"] and not row["short_notice_30"]
    assert not row["missed_weight"]


def test_a_short_notice_corner_is_binned_and_hinged(table):
    side = notice.attach(_side(["f2"], ["aaa"]), table)
    row = side.iloc[0]
    assert row["short_notice_7"] and row["short_notice_30"]
    assert row["notice_shortfall_days"] == notice.NOTICE_HINGE_DAYS - 5.0


def test_a_thirty_day_replacement_is_short_notice_30_only():
    table = pd.DataFrame({
        "fight_id": ["f", "f"], "fighter_id": ["x", "y"],
        "notice_days": [21.0, np.nan],
        "missed_weight": [False, False],
        "missed_weight_over_lbs": [np.nan, np.nan],
    })
    row = notice.attach(_side(["f"], ["x"]), table).iloc[0]
    assert row["short_notice_30"] and not row["short_notice_7"]
    assert row["notice_shortfall_days"] == 9.0


def test_a_missed_weight_corner_carries_the_overage(table):
    row = notice.attach(_side(["f2"], ["ccc"]), table).iloc[0]
    assert row["missed_weight"]
    assert row["missed_weight_over_lbs"] == 3.5


def test_an_unobserved_fight_is_all_unknown(table):
    side = notice.attach(_side(["nope", "f1"], ["aaa", "aaa"]), table)
    assert side["notice_missing"].tolist() == [True, False]
    unknown = side.iloc[0]
    for key in notice.NUMERIC_STATE_KEYS:
        assert pd.isna(unknown[key]), key
    for key in notice.BOOLEAN_STATE_KEYS:
        assert unknown[key] == False, key  # noqa: E712


def test_a_corner_absent_from_an_observed_fight_is_unknown(table):
    """Half a fight is not an observation. The derivation emits both corners
    or neither, so this cannot arise from the committed table -- but a caller
    passing a fighter who was not in the bout must not silently read as a
    full camp."""
    row = notice.attach(_side(["f1"], ["zzz"]), table).iloc[0]
    assert bool(row["notice_missing"])
    assert pd.isna(row["notice_shortfall_days"])


def test_attach_preserves_row_order_and_length(table):
    side = notice.attach(_side(["f2", "nope", "f1"], ["ccc", "aaa", "bbb"]), table)
    assert side["fighter_id"].tolist() == ["ccc", "aaa", "bbb"]
    assert side["notice_missing"].tolist() == [False, True, False]


def test_state_for_matches_the_attach_path(table):
    served = notice.state_for("f2", "aaa", table)
    trained = notice.attach(_side(["f2"], ["aaa"]), table).iloc[0]
    for key in notice.STATE_KEYS + ("notice_missing",):
        assert served[key] == trained[key], key


def test_unknown_state_is_what_an_unmatched_corner_gets(table):
    state = notice.state_for("nope", "aaa", table)
    assert state["notice_missing"] is True
    assert all(pd.isna(state[key]) for key in notice.NUMERIC_STATE_KEYS)
    assert all(state[key] is False for key in notice.BOOLEAN_STATE_KEYS)


def test_fight_context_scalars(table):
    a = notice.state_for("f1", "aaa", table)
    b = notice.state_for("f1", "bbb", table)
    missing = notice.state_for("nope", "aaa", table)
    assert notice.fight_context(a, b) == {"notice_unknown": False}
    assert notice.fight_context(a, missing) == {"notice_unknown": True}


def test_fight_context_is_vectorised_and_matches_the_scalar_path(table):
    left = notice.attach(_side(["f1", "f1"], ["aaa", "aaa"]), table)
    right = notice.attach(_side(["f1", "nope"], ["bbb", "bbb"]), table)
    assert list(notice.fight_context(left, right)["notice_unknown"]) == [False, True]


# --- properties of the real derived table ------------------------------------


@pytest.mark.skipif(not notice.DEFAULT_PATH.exists(), reason="data/external not built")
def test_the_committed_table_carries_both_corners_of_every_fight():
    """The three-state boundary. A fight with one corner in the table would
    mean "we know A had a full camp but nothing about B", which is exactly the
    per-corner asymmetry the `external` block was corrected to avoid."""
    table = notice.load_table()
    per_fight = table.groupby("fight_id").size()
    assert (per_fight == 2).all(), per_fight[per_fight != 2].head()
    assert not table.duplicated(["fight_id", "fighter_id"]).any()


@pytest.mark.skipif(
    not notice.DEFAULT_PATH.exists() or not (PROCESSED / "fights.parquet").exists(),
    reason="data not built",
)
def test_the_committed_table_names_the_real_corners_of_each_fight():
    fights = pd.read_parquet(PROCESSED / "fights.parquet").set_index("fight_id")
    table = notice.load_table()
    assert set(table["fight_id"]) <= set(fights.index)
    pairs = table.groupby("fight_id")["fighter_id"].apply(frozenset)
    expected = fights.loc[pairs.index].apply(
        lambda f: frozenset({f["fighter_a_id"], f["fighter_b_id"]}), axis=1
    )
    assert pairs.equals(expected)


@pytest.mark.skipif(
    not notice.DEFAULT_PATH.exists() or not (PROCESSED / "fights.parquet").exists(),
    reason="data not built",
)
def test_an_unknown_fight_carries_exactly_one_state_tuple():
    """The data-level leak guard, the same one `external` carries.

    Whether a fight is observed is a property of the SOURCE's bout universe,
    not of either fighter, so no per-corner channel may survive: over every
    fight the source has not seen, the state this module emits must take
    exactly ONE distinct value tuple. If it does, nothing computed from it can
    separate a fight whose unknown corner is A from one whose unknown corner
    is B -- whatever the columns downstream are called. A feature block built
    on this module inherits the property; the `notice` block did, and the
    check ran on its feature table before the block was reverted (5,641
    unknown rows, one tuple).
    """
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    corners = pd.concat([
        fights[["fight_id", "fighter_a_id"]].rename(
            columns={"fighter_a_id": "fighter_id"}),
        fights[["fight_id", "fighter_b_id"]].rename(
            columns={"fighter_b_id": "fighter_id"}),
    ])
    attached = notice.attach(corners)
    unknown = attached[attached["notice_missing"].astype(bool)]
    assert len(unknown) > 1000, f"only {len(unknown)} unknown corners"
    distinct = {
        tuple(
            "NaN" if pd.isna(row[key]) else row[key]
            for key in notice.STATE_KEYS
        )
        for _, row in unknown.iterrows()
    }
    assert len(distinct) == 1, (
        f"{len(distinct)} distinct state tuples over {len(unknown)} unknown corners; "
        "the loader can tell which corner the source is missing"
    )


def test_the_flag_stays_excluded_even_though_the_block_is_unregistered():
    """The block's shipping form (`notice_noflag`), asserted structurally.

    `notice_unknown` would have to reach the table -- it is the only column
    that says the source has not seen a fight, and every 2025-and-later row is
    one -- but modelling it is modelling the calendar: the Bet MMA bout list
    ends 2024-12-14, so the flag separates the recent folds from the old ones
    rather than one fighter from the other. Measured in SP2 Task 12: torch
    0.6482 with the flag in the model, 0.6472 with it held out.

    The block itself is not registered: it was rejected in SP2 and again in
    SP2.1, where it was one of the four blocks combined into the treatment arm
    (neither that arm nor its control cleared the bar). What this test pins is
    that the two exclusion lists still name the flag, so re-registering the
    block is uncommenting `feature_blocks.py` and nothing else -- an exclusion
    silently dropped in the meantime is how the rejected variant would ship by
    accident.
    """
    from mma.feature_blocks import BLOCKS, NOTICE_BLOCK
    from mma.models.xgb import NON_FEATURES
    from mma.tensors import DROPPED

    assert NOTICE_BLOCK not in BLOCKS, "rejected in SP2 and again in SP2.1"
    assert "notice_unknown" in NON_FEATURES
    assert "notice_unknown" in DROPPED
