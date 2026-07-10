from pathlib import Path

import pandas as pd
import pytest

from mma.elo import INITIAL_RATING

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "ratings.parquet").exists(),
    reason="ratings not built (run scripts/build_ratings.py)",
)


def _load():
    return (
        pd.read_parquet(PROCESSED / "ratings.parquet"),
        pd.read_parquet(PROCESSED / "fights.parquet"),
    )


def test_two_rows_per_rated_fight_and_nc_excluded():
    ratings, fights = _load()
    rated = fights[fights["winner"].isin(["a", "b", "draw"])]
    assert set(ratings["fight_id"]) == set(rated["fight_id"])
    assert (ratings.groupby("fight_id").size() == 2).all()


def test_debut_pre_rating_is_initial():
    ratings, _ = _load()
    debuts = ratings[ratings["pre_fights"] == 0]
    assert (debuts["pre_overall"] == INITIAL_RATING).all()
    assert len(debuts) > 2000  # most fighters debut at some point


def test_point_in_time_pre_equals_previous_post():
    ratings, _ = _load()
    ordered = ratings.sort_values(["fighter_id", "date"], kind="stable")
    for _, group in list(ordered.groupby("fighter_id"))[:200]:
        posts = group["post_overall"].to_list()
        pres = group["pre_overall"].to_list()
        assert pres[1:] == posts[:-1]


def test_ratings_in_sane_range():
    ratings, _ = _load()
    assert ratings["post_overall"].between(1000, 2200).all()
