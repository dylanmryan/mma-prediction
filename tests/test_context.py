"""The `context` block: referee tendency, home advantage, bonus record.

Every number below is hand-checkable off a three- or four-fight chronological
fixture, because the whole point of these columns is that they are computed
from PRIOR fights only -- a rate that quietly included the fight it describes
would look like signal and be a leak.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma import context

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _fights() -> pd.DataFrame:
    """Three fights for one referee, then a fourth with none recorded."""
    return pd.DataFrame({
        "fight_id": ["f1", "f2", "f3", "f4"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "fighter_a_id": ["x", "x", "x", "x"],
        "fighter_b_id": ["o1", "o2", "o3", "o4"],
        "winner": ["a", "b", "a", "a"],
        "method": ["ko_tko", "decision", "submission", "decision"],
        "referee": ["Herb Dean", "Herb Dean", "Herb Dean", None],
        "location": [
            "Las Vegas, Nevada, USA", "Sao Paulo, Sao Paulo, Brazil",
            "London, England, United Kingdom", None,
        ],
        "duration_sec": [300.0, 900.0, 200.0, 900.0],
    })


# --- referee -----------------------------------------------------------------


def test_referee_rates_use_prior_fights_only():
    history = context.referee_history(_fights()).set_index("fight_id")
    # f1 is Herb Dean's first: nothing to average yet, but he IS recorded.
    assert pd.isna(history.loc["f1", "referee_finish_rate"])
    assert pd.isna(history.loc["f1", "referee_decision_rate"])
    assert history.loc["f1", "referee_missing"] is np.False_ or not history.loc["f1", "referee_missing"]
    # f2 sees exactly f1: one finish, no decision.
    assert history.loc["f2", "referee_finish_rate"] == pytest.approx(1.0)
    assert history.loc["f2", "referee_decision_rate"] == pytest.approx(0.0)
    # f3 sees f1 and f2: one finish, one decision.
    assert history.loc["f3", "referee_finish_rate"] == pytest.approx(0.5)
    assert history.loc["f3", "referee_decision_rate"] == pytest.approx(0.5)


def test_a_fight_with_no_recorded_referee_is_flagged_and_rated_nan():
    history = context.referee_history(_fights()).set_index("fight_id")
    assert bool(history.loc["f4", "referee_missing"])
    assert pd.isna(history.loc["f4", "referee_finish_rate"])
    assert pd.isna(history.loc["f4", "referee_decision_rate"])
    assert not history.loc[["f1", "f2", "f3"], "referee_missing"].any()


def test_referee_history_is_point_in_time_under_truncation():
    fights = _fights()
    full = context.referee_history(fights).set_index("fight_id")
    early = context.referee_history(fights[fights["date"] < "2020-03-01"]).set_index("fight_id")
    pd.testing.assert_frame_equal(full.loc[early.index], early)


def test_referee_history_covers_every_input_fight_once():
    fights = _fights()
    history = context.referee_history(fights)
    assert list(history["fight_id"]) == ["f1", "f2", "f3", "f4"]
    assert len(history) == len(fights)


def test_serving_rates_are_the_whole_table_and_the_missing_path_is_the_default():
    rates = context.referee_rates(_fights())
    # Three recorded fights: two finishes, one decision.
    assert rates["Herb Dean"] == (pytest.approx(2 / 3), pytest.approx(1 / 3))
    served = context.referee_context("Herb Dean", rates)
    assert served["referee_finish_rate"] == pytest.approx(2 / 3)
    assert served["referee_missing"] is False
    # No referee named -- the state every Wikipedia-sourced future card is in.
    for absent in (None, pd.NA, np.nan):
        blank = context.referee_context(absent, rates)
        assert blank["referee_missing"] is True
        assert pd.isna(blank["referee_finish_rate"])
        assert pd.isna(blank["referee_decision_rate"])
    # A referee we have never seen is missing rates but is not "missing".
    unseen = context.referee_context("Nobody At All", rates)
    assert unseen["referee_missing"] is False
    assert pd.isna(unseen["referee_finish_rate"])


# --- event country and home advantage ----------------------------------------


@pytest.mark.parametrize("location,expected", [
    ("Las Vegas, Nevada, USA", "united states"),
    ("Sao Paulo, Sao Paulo, Brazil", "brazil"),
    ("London, England, United Kingdom", "united kingdom"),
    ("Abu Dhabi, United Arab Emirates", "united arab emirates"),
    ("Las Vegas", None),          # no comma: a city, not a country
    (None, None),
    (pd.NA, None),
    ("", None),
])
def test_event_country_is_the_last_component(location, expected):
    assert context.event_country(location) == expected


@pytest.mark.parametrize("nationality,country,expected", [
    ("United States", "united states", True),
    ("USA", "united states", True),          # the snapshot writes both spellings
    ("England", "united kingdom", True),     # home nations vs the event's country
    ("Scotland", "united kingdom", True),
    ("Brazil", "united states", False),
    (None, "united states", False),
    ("Brazil", None, False),
    (None, None, False),
])
def test_home_country_compares_normalised_names(nationality, country, expected):
    assert context.home_country(nationality, country) is expected


def test_home_country_unknown_is_symmetric_in_the_corners():
    assert context.home_context("Brazil", "United States", "brazil")["home_country_unknown"] is False
    assert context.home_context(None, "United States", "brazil")["home_country_unknown"] is True
    assert context.home_context("Brazil", None, "brazil")["home_country_unknown"] is True
    assert context.home_context("Brazil", "United States", None)["home_country_unknown"] is True


def test_home_country_unknown_vectorises():
    flags = context.home_context(
        pd.Series(["Brazil", None, "Brazil"]),
        pd.Series(["United States", "United States", "United States"]),
        pd.Series(["brazil", "brazil", None]),
    )["home_country_unknown"]
    assert list(flags) == [False, True, True]


# --- bonuses -----------------------------------------------------------------


def test_bonus_fights_reads_a_passed_table_and_defaults_to_the_committed_one():
    passed = context.bonus_fights(pd.DataFrame({"fight_id": ["f1", "f3"]}))
    assert passed == frozenset({"f1", "f3"})
    if (PROCESSED / "bonuses.parquet").exists():
        assert len(context.bonus_fights()) > 1000


# --- why `home_country` is not a per-corner model input ----------------------


@pytest.mark.skipif(
    not (PROCESSED / "fights.parquet").exists()
    or not (Path(__file__).resolve().parents[1] / "data" / "external"
            / "fighter_external.parquet").exists(),
    reason="processed/external data not built",
)
def test_a_per_corner_home_country_pair_is_a_coverage_channel():
    """The measurement that kept `home_country_a`/`_b` out of the model.

    `home_country` is False when a corner's nationality is unknown, and
    nationality only exists for fighters the `external` snapshot mapped --
    membership of which tracks how long a fighter's UFC career turned out to
    be. So over the fights where exactly ONE corner has a nationality, the
    pair is not constant (three distinct tuples), and those are exactly the
    rows where the mapped corner wins about three quarters of the time. A
    model given the pair can read the coverage artifact off it, which is the
    `external` block's measured leak in a second channel.

    This asserts the FACT rather than a fix, because the fix is a choice made
    elsewhere: SP2 Task 8 measured this, held the two columns out of both
    model matrices, and then rejected the block on its own numbers. If this
    test ever fails because the pair became constant on half-known rows, the
    reasoning behind that exclusion has changed and should be re-read.
    """
    from mma import external

    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    table = external.load_table()
    nationality = {
        fighter: value
        for fighter, value in zip(table["fighter_id"], table["nationality"])
        if pd.notna(value)
    }
    known_a = fights["fighter_a_id"].isin(nationality)
    known_b = fights["fighter_b_id"].isin(nationality)
    half = fights[(known_a ^ known_b) & fights["winner"].isin(["a", "b"])]
    assert len(half) > 500, f"fixture assumption: too few half-known fights ({len(half)})"

    countries = [context.event_country(value) for value in half["location"]]
    pairs = {
        (context.home_country(nationality.get(a), country),
         context.home_country(nationality.get(b), country))
        for a, b, country in zip(half["fighter_a_id"], half["fighter_b_id"], countries)
    }
    assert len(pairs) > 1, "the pair is constant here; re-read the exclusion argument"

    known_corner_wins = (
        (half["fighter_a_id"].isin(nationality) & (half["winner"] == "a"))
        | (half["fighter_b_id"].isin(nationality) & (half["winner"] == "b"))
    ).mean()
    assert known_corner_wins > 0.65, known_corner_wins


def test_the_home_country_pair_stays_excluded_even_though_the_block_is_not():
    """The block's shipping form (`context_nohome`), asserted structurally.

    Unlike `notice_unknown` this is not a tuning preference: the pair FAILS
    the constant-vector leak check (see
    `test_a_per_corner_home_country_pair_is_a_coverage_channel` above, and the
    whole-table version in `tests/test_external.py`), because a per-corner
    boolean is False when the corner's nationality is unknown and nationality
    exists only for fighters the `external` snapshot mapped.

    The block is not registered -- rejected in SP2 and again inside SP2.1's
    combined treatment arm -- so the pair is not in the table today. The
    exclusions outlive the registration on purpose: this is a leak guard, not
    a tuning choice, and whoever uncomments the block next must not have to
    rediscover it. The property itself is asserted one level lower, on
    `mma.context` over the real tables, in the test named above.
    """
    from mma.feature_blocks import BLOCKS, CONTEXT_BLOCK
    from mma.models.xgb import NON_FEATURES
    from mma.tensors import DROPPED

    pair = {"home_country_a", "home_country_b"}
    assert CONTEXT_BLOCK not in BLOCKS, "rejected in SP2 and again in SP2.1"
    assert pair <= NON_FEATURES, "xgb must not model them"
    assert pair <= set(DROPPED), "torch must not model them"
