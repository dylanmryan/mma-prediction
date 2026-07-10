from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "fights.parquet").exists(),
    reason="processed data not built (run scripts/make_dataset.py)",
)


def test_fights_volume_and_range():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert len(fights) > 6000
    assert fights["date"].min().year <= 1995
    assert fights["date"].max().year >= 2024


def test_method_distribution_plausible():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    shares = fights["method"].value_counts(normalize=True)
    assert shares["decision"] > 0.30
    assert shares["ko_tko"] > 0.25
    assert shares["submission"] > 0.10


def test_finishes_have_round_decisions_do_not():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    finishes = fights[fights["method"].isin(["ko_tko", "submission"])]
    decisions = fights[fights["method"] == "decision"]
    assert finishes["finish_round"].notna().all()
    assert decisions["finish_round"].isna().all()


def test_winners_mostly_decisive():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert fights["winner"].isin(["a", "b", "draw", "nc"]).all()
    assert fights["winner"].isin(["a", "b"]).mean() > 0.97


def test_stats_join_to_fights():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    assert set(stats["fight_id"]) == set(fights["fight_id"])
    assert len(stats) == 2 * len(fights)


def test_fight_participants_exist_in_fighters():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    participants = set(fights["fighter_a_id"]) | set(fights["fighter_b_id"])
    orphans = participants - set(fighters["fighter_id"])
    assert len(orphans) < 0.02 * len(participants)
