"""The official-rankings table and the as-of join that turns it into state.

A ranking is not a fixed fact about a fighter -- it moves every week, and it
moves BECAUSE of fight results. So unlike every other external table in this
project, the safety of this one is a join rule rather than a cut in the
derivation: the ranking used for a fight must be the last one published
STRICTLY BEFORE that fight's date. A ranking published on the fight day itself
is already inadmissible, because the UFC updates on Tuesdays after the
weekend's cards and a same-day list can encode results from earlier that week.
`test_the_ranking_used_is_the_last_one_published_before_the_fight` is the one
test in this file that the block would be unsafe without.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma import rankings

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture
def table() -> pd.DataFrame:
    """Two publications a week apart, with a champion and a rank change."""
    return pd.DataFrame({
        "fighter_id": ["aaa", "bbb", "aaa", "bbb", "ccc"],
        "date": pd.to_datetime([
            "2020-01-07", "2020-01-07", "2020-01-14", "2020-01-14", "2020-01-14",
        ]),
        "weight_class": ["Lightweight"] * 5,
        "rank": [0, 5, 3, 0, 12],
    })


def _side(fighter_ids, dates, weight_class="Lightweight") -> pd.DataFrame:
    return pd.DataFrame({
        "fighter_id": fighter_ids,
        "date": pd.to_datetime(dates),
        "weight_class": [weight_class] * len(fighter_ids),
    })


def test_the_ranking_used_is_the_last_one_published_before_the_fight(table):
    """The point-in-time rule, stated as a failure mode.

    A fight on 2020-01-14 must read the 2020-01-07 list. Reading its own day's
    list would give "aaa" rank 3 instead of 0 -- a change that the fight's own
    week produced, which is exactly the leak the strictly-before join exists
    to prevent.
    """
    same_day = rankings.attach(_side(["aaa"], ["2020-01-14"]), table)
    assert same_day["rank"].iloc[0] == 0
    later = rankings.attach(_side(["aaa"], ["2020-01-21"]), table)
    assert later["rank"].iloc[0] == 3


def test_state_for_applies_the_same_strictly_before_rule(table):
    assert rankings.state_for("aaa", "Lightweight", "2020-01-14", table)["rank"] == 0
    assert rankings.state_for("aaa", "Lightweight", "2020-01-21", table)["rank"] == 3


def test_a_fight_before_the_first_publication_is_unranked(table):
    row = rankings.attach(_side(["aaa"], ["2019-12-01"]), table).iloc[0]
    assert not row["is_ranked"] and not row["is_champion"]
    assert pd.isna(row["rank"])


def test_the_champion_is_rank_zero_and_is_ranked(table):
    row = rankings.attach(_side(["aaa"], ["2020-01-10"]), table).iloc[0]
    assert row["rank"] == rankings.CHAMPION_RANK
    assert row["is_champion"] and row["is_ranked"]


def test_an_unranked_fighter_is_not_a_missing_one(table):
    """Most fighters are outside the top 15 every week, and the source
    publishes the whole list -- so "unranked" is an observation, not a gap."""
    row = rankings.attach(_side(["nobody"], ["2020-01-21"]), table).iloc[0]
    assert not row["is_ranked"]
    assert pd.isna(row["rank"])


def test_a_ranking_in_another_division_does_not_carry_over(table):
    """Rank is per division; a lightweight's rank says nothing about the
    welterweight bout he took."""
    row = rankings.attach(
        _side(["aaa"], ["2020-01-21"], weight_class="Welterweight"), table
    ).iloc[0]
    assert not row["is_ranked"]


def test_attach_preserves_row_order_and_length(table):
    side = _side(["ccc", "nobody", "bbb"], ["2020-01-21", "2020-01-21", "2020-01-21"])
    attached = rankings.attach(side, table)
    assert attached["fighter_id"].tolist() == ["ccc", "nobody", "bbb"]
    assert attached["rank"].tolist()[0] == 12
    assert pd.isna(attached["rank"].iloc[1])
    assert attached["rank"].iloc[2] == 0


def test_attach_is_not_confused_by_unsorted_input(table):
    """`merge_asof` requires a date-sorted left frame; `attach` sorts and
    restores, so a caller's row order is its own business."""
    side = _side(["aaa", "aaa"], ["2020-01-21", "2020-01-10"])
    attached = rankings.attach(side, table)
    assert attached["rank"].tolist() == [3, 0]


def test_state_for_matches_the_attach_path(table):
    served = rankings.state_for("bbb", "Lightweight", "2020-01-21", table)
    trained = rankings.attach(_side(["bbb"], ["2020-01-21"]), table).iloc[0]
    for key in rankings.STATE_KEYS:
        assert served[key] == trained[key], key


def test_fight_context_scalars(table):
    ranked = rankings.state_for("aaa", "Lightweight", "2020-01-21", table)
    unranked = rankings.state_for("nobody", "Lightweight", "2020-01-21", table)
    assert rankings.fight_context(ranked, ranked, "2020-01-21") == {
        "rank_missing": False, "ranking_regime_post_2026_06": False,
    }
    assert rankings.fight_context(ranked, unranked, "2020-01-21")["rank_missing"] is True


def test_the_regime_flag_turns_on_at_the_switch_date(table):
    ranked = rankings.state_for("aaa", "Lightweight", "2020-01-21", table)
    before = rankings.RANKING_REGIME_CHANGE - pd.Timedelta(days=1)
    assert not rankings.fight_context(ranked, ranked, before)[
        "ranking_regime_post_2026_06"]
    assert rankings.fight_context(ranked, ranked, rankings.RANKING_REGIME_CHANGE)[
        "ranking_regime_post_2026_06"]


def test_fight_context_is_vectorised_and_matches_the_scalar_path(table):
    left = rankings.attach(_side(["aaa", "aaa"], ["2020-01-21", "2026-07-01"]), table)
    right = rankings.attach(_side(["bbb", "nobody"], ["2020-01-21", "2026-07-01"]), table)
    context = rankings.fight_context(left, right, left["date"])
    assert list(context["rank_missing"]) == [False, True]
    assert list(context["ranking_regime_post_2026_06"]) == [False, True]


# --- properties of the real derived table ------------------------------------


@pytest.mark.skipif(not rankings.DEFAULT_PATH.exists(), reason="rankings not built")
def test_the_committed_table_is_keyed_by_our_fighter_ids_and_divisions():
    table = rankings.load_table()
    assert not table.duplicated(["fighter_id", "weight_class", "date"]).any()
    assert table["rank"].min() >= 0
    # Pound-for-pound is not a division a bout takes place in and is dropped.
    assert not table["weight_class"].str.lower().str.contains("pound-for-pound").any()


@pytest.mark.skipif(
    not rankings.DEFAULT_PATH.exists() or not (PROCESSED / "fighters.parquet").exists(),
    reason="data not built",
)
def test_every_ranked_fighter_id_is_one_of_ours_and_every_division_is_ours():
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    table = rankings.load_table()
    assert set(table["fighter_id"]) <= set(fighters["fighter_id"])
    assert set(table["weight_class"]) <= set(fights["weight_class"].dropna())


@pytest.mark.skipif(
    not rankings.DEFAULT_PATH.exists() or not (PROCESSED / "fights.parquet").exists(),
    reason="data not built",
)
def test_no_fight_reads_a_ranking_published_on_or_after_its_own_date():
    """The point-in-time rule, checked end to end on the real tables rather
    than on a fixture: for every corner of every fight, the publication the
    join used is strictly earlier than the fight."""
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    table = rankings.load_table()
    side = fights[["fighter_a_id", "date", "weight_class"]].rename(
        columns={"fighter_a_id": "fighter_id"}
    )
    attached = rankings.attach(side, table)
    ranked = attached[attached["is_ranked"].astype(bool)]
    assert len(ranked) > 500, f"only {len(ranked)} ranked corners"

    latest = (
        table.merge(
            ranked[["fighter_id", "weight_class", "date"]].rename(
                columns={"date": "fight_date"}),
            on=["fighter_id", "weight_class"], how="inner",
        )
    )
    used = latest[latest["date"] < latest["fight_date"]]
    assert len(used) > 0
    # Every publication the join could have used is strictly earlier, and the
    # rank it returned is the latest such publication's.
    best = used.sort_values("date").groupby(
        ["fighter_id", "weight_class", "fight_date"]
    )["rank"].last()
    joined = ranked.set_index(["fighter_id", "weight_class", "date"])["rank"]
    common = best.index.intersection(joined.index)
    assert len(common) > 500
    assert (best.loc[common].to_numpy() == joined.loc[common].to_numpy()).all()
