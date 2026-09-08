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


@pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists()
    or not (PROCESSED / "fights.parquet").exists()
    or not external.DEFAULT_PATH.exists(),
    reason="data not built",
)
def test_a_half_matched_fight_carries_exactly_one_value_tuple():
    """The data-level leak guard for this block (SP2 Task 11).

    Membership of the snapshot's cross-source fighter mapping is a function of
    how long a fighter's UFC career turned out to be -- among 2013-2022
    debutants, 24% of the one-and-done fighters are in it against 100% of
    those with 11+ bouts -- and the unmapped corner loses 76% of the time
    overall, 90% of the time in the debut slice. A per-corner
    `external_missing_a`/`_b` pair therefore told the model which corner went
    on to have a career, and the walk-forward signature confirmed it: -0.014
    on the historical folds, +0.049 on 2025, where missingness starts meaning
    "debuted after the snapshot" instead.

    The property that makes that unreachable is not a naming rule, so this
    does not check names: over the rows where exactly ONE corner is unmapped
    -- the rows the selection effect could be read off -- the block's eight
    columns must take exactly ONE distinct value tuple. If they do, no
    function of them can separate a fight whose unmapped corner is A from one
    whose unmapped corner is B, whatever the columns happen to be called.
    Adding any per-corner channel breaks this immediately.
    """
    from mma.feature_blocks import BLOCKS, EXTERNAL_BLOCK, columns_of

    features = pd.read_parquet(PROCESSED / "features.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    mapped = set(external.load_table()["fighter_id"])
    unmapped_a = ~fights["fighter_a_id"].isin(mapped)
    unmapped_b = ~fights["fighter_b_id"].isin(mapped)
    half = set(fights.loc[unmapped_a ^ unmapped_b, "fight_id"])
    assert len(half) > 100, f"fixture assumption: too few half-matched fights ({len(half)})"

    columns = list(columns_of(BLOCKS[EXTERNAL_BLOCK]))
    # Not every fight reaches the feature table (undecided outcomes are
    # filtered out), so this is a subset of `half`, not all of it.
    rows = features.loc[features["fight_id"].isin(half), columns]
    assert len(rows) > 1000, f"only {len(rows)} half-matched feature rows"
    distinct = {
        tuple("NaN" if pd.isna(value) else value for value in row)
        for row in rows.itertuples(index=False)
    }
    assert len(distinct) == 1, (
        f"{len(distinct)} distinct value tuples over {len(rows)} half-matched rows; "
        "the block can tell which corner is unmapped, which is the measured leak"
    )
    # Cheap extra: the naming rule that used to stand in for the property.
    assert "external_missing" in BLOCKS[EXTERNAL_BLOCK].fight_level
    assert not any(column.endswith(("_a", "_b")) for column in columns)


def test_a_missing_pro_debut_date_is_dropped_rather_than_flagged_present(table):
    """A NaT debut used to slip through: `NaT > date` is False, so the negated
    sanity condition kept the fighter, whose `days_since_pro_debut` then came
    out NaN while `external_missing` said False -- a present flag over absent
    values, which is the one invariant this module has."""
    with_nat = pd.concat([table, pd.DataFrame([{
        "fighter_id": "nat", "pre_ufc_wins": 7, "pre_ufc_losses": 1,
        "pre_ufc_finish_rate": 0.5, "pre_ufc_finish_loss_rate": 0.0,
        "pre_ufc_avg_opp_wins": 3.0, "pro_debut_date": pd.NaT,
        "first_ufc_date": pd.Timestamp("2015-01-01"),
        "nationality": "Brazil", "gym_id": None,
    }])], ignore_index=True)

    assert "nat" not in set(external.drop_source_errors(with_nat)["fighter_id"])
    side = external.attach(_side(["nat"], ["2018-01-01"]), with_nat).iloc[0]
    assert bool(side["external_missing"])
    assert pd.isna(side["days_since_pro_debut"])
    assert pd.isna(side["pre_ufc_wins"])
    assert external.state_for("nat", pd.Timestamp("2018-01-01"), with_nat)[
        "external_missing"] is True


@pytest.mark.skipif(
    not (PROCESSED / "fights.parquet").exists() or not external.DEFAULT_PATH.exists(),
    reason="data not built",
)
def test_pre_ufc_values_are_constant_across_a_fighters_ufc_career():
    """The real point-in-time guarantee for this block, asserted directly.

    `test_no_leakage_truncation_invariance` cannot provide it: the external
    table is a static committed artifact, so truncating the fights table
    cannot change anything derived from it and the test passes regardless of
    what is in here. What actually makes the block safe is the cut in
    `scripts/build_external.derive` -- `history["date"] < first_ufc_date` --
    which means every `pre_ufc_*` value summarises a window that closed
    before the fighter's UFC career began, and is therefore the SAME at every
    one of their UFC fights. A column that leaked a later result would vary
    across a fighter's own fights; these do not.
    """
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    corners = pd.concat([
        fights[["fighter_a_id", "date"]].rename(columns={"fighter_a_id": "fighter_id"}),
        fights[["fighter_b_id", "date"]].rename(columns={"fighter_b_id": "fighter_id"}),
    ])
    attached = external.attach(corners)
    repeat = attached[attached["external_missing"].eq(False)]
    counts = repeat["fighter_id"].value_counts()
    veterans = set(counts[counts >= 2].index)
    repeat = repeat[repeat["fighter_id"].isin(veterans)]
    assert len(veterans) > 500, f"fixture assumption: {len(veterans)} multi-fight fighters"

    varying = repeat.groupby("fighter_id")[list(external.NUMERIC_STATE_KEYS)].nunique(
        dropna=False
    ).max()
    assert (varying == 1).all(), f"varies within a fighter's UFC career:\n{varying}"

    # `days_since_pro_debut` is the deliberate exception: it is a duration from
    # a fixed date, so it MUST advance across a fighter's career.
    spans = repeat.groupby("fighter_id")["days_since_pro_debut"].nunique(dropna=False)
    assert (spans > 1).any(), "days_since_pro_debut should vary within a career"


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


def test_the_fight_level_flags_are_produced_but_never_modelled():
    """The shipped shape of this block: flags in the TABLE, out of the MODEL.

    `external_missing` has to reach the table for
    `mma.walkforward.slice_masks` to report its slice, and both flags have to
    reach the served row for `tests/test_serving_parity.py` to stay meaningful
    -- but neither is a feature. Modelling them makes the block decay as the
    snapshot ages (SP2 Task 11 shipping note: +0.0006 row-weighted on the
    2024-2025 folds with the flags, -0.0006 without), because the flags track
    the snapshot's coverage rather than the fight.
    """
    from mma.feature_blocks import BLOCKS, EXTERNAL_BLOCK
    from mma.models.xgb import NON_FEATURES
    from mma.tensors import DROPPED

    flags = {"external_missing", "same_country"}
    assert flags <= set(BLOCKS[EXTERNAL_BLOCK].fight_level), "must still be produced"
    assert flags <= NON_FEATURES, "xgb must not model them"
    assert flags <= set(DROPPED), "torch must not model them"


@pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists(), reason="processed data not built"
)
def test_the_flags_survive_into_the_table_and_out_of_both_model_matrices():
    """The same property end to end, on the committed feature table."""
    from mma.models.xgb import feature_frame
    from mma.tensors import Preprocessor
    from mma.walkforward import slice_masks

    features = pd.read_parquet(PROCESSED / "features.parquet")
    flags = {"external_missing", "same_country"}
    assert flags <= set(features.columns)
    assert "external_missing" in slice_masks(features)
    assert not flags & set(feature_frame(features).columns)
    prep = Preprocessor.fit(features, train_mask=np.ones(len(features), dtype=bool))
    assert not flags & set(prep.numeric_columns)


# --- the id-only join, guarded at the serving boundary -----------------------


@pytest.mark.parametrize("label", [
    "Jon Jones",                 # a name-indexed bio row: the failure that motivated this
    "07f72a2a7591b30b ",         # trailing whitespace
    "07F72A2A7591B30B",          # uppercase hex
    "07f72a2a7591b30",           # 15 characters
    "07f72a2a7591b30bb",         # 17 characters
    "07f72a2a7591b30g",          # not hex
    None,                        # unlabelled
    12345,                       # not a string at all
])
def test_only_a_ufcstats_shaped_id_is_accepted_as_a_bio_label(label):
    """A name-indexed bio row must raise, not serve an all-NaN external corner.

    `fighters.set_index("name").loc["Jon Jones"]` is the obvious wrong call and
    it produces a bio row whose label is a string, so the old
    `isinstance(str)` check passed it. The lookup then matched nothing in the
    external table and the fighter -- who may well be in it -- was served NaN
    everywhere with `external_missing` set. Silent, and exactly the
    name-matching failure the id-only join exists to prevent.
    """
    from mma.inference import _fighter_id

    bio = pd.Series({"height_cm": 180.0}, name=label)
    with pytest.raises(KeyError, match="ufcstats fighter id"):
        _fighter_id(bio, "a")


def test_a_real_ufcstats_id_is_accepted():
    from mma.inference import _fighter_id

    bio = pd.Series({"height_cm": 180.0}, name="07f72a2a7591b30b")
    assert _fighter_id(bio, "a") == "07f72a2a7591b30b"


@pytest.mark.skipif(
    not (PROCESSED / "fighters.parquet").exists(), reason="processed data not built"
)
def test_every_real_fighter_id_passes_the_shape_check():
    """The validation rule is only safe because the whole id space fits it."""
    from mma.inference import _fighter_id

    ids = pd.read_parquet(PROCESSED / "fighters.parquet")["fighter_id"]
    for fighter_id in ids:
        assert _fighter_id(pd.Series(dtype=float, name=fighter_id), "a") == fighter_id
