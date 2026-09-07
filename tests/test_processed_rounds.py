from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "round_stats.parquet").exists(),
    reason="processed data not built (run scripts/make_dataset.py)",
)


def test_round_stats_volume_and_shape():
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    assert len(rounds) > 44000  # 2 corners x ~22k+ rounds
    assert set(rounds["corner"]) == {"a", "b"}
    assert rounds["round_no"].between(1, 6).all()
    assert not rounds.duplicated(["fight_id", "round_no", "corner"]).any()


def test_round_stats_join_to_fights():
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert set(rounds["fight_id"]) <= set(fights["fight_id"])
    joined = rounds.merge(fights[["fight_id", "fighter_a_id", "fighter_b_id"]], on="fight_id")
    a = joined[joined["corner"] == "a"]
    assert (a["fighter_id"] == a["fighter_a_id"]).all()


def test_round_totals_match_fight_totals_for_recent_fights():
    """Fight-total sig strikes must equal the sum over rounds where both exist."""
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    recent = fights[fights["date"] >= "2015-01-01"]["fight_id"]
    summed = (
        rounds[rounds["fight_id"].isin(recent)]
        .groupby(["fight_id", "corner"])["sig_landed"].sum()
    )
    totals = stats.set_index(["fight_id", "corner"]).loc[summed.index, "sig_landed"]
    agree = (summed.values == totals.values).mean()
    assert agree > 0.99


def test_control_seconds_plausible():
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    assert rounds["ctrl_sec"].dropna().between(0, 720).all()  # 10-min rounds in early eras


def test_bonuses_table():
    bonuses = pd.read_parquet(PROCESSED / "bonuses.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert len(bonuses) > 2000
    assert set(bonuses["fight_id"]) <= set(fights["fight_id"])
    assert set(bonuses["bonus_type"]) == {
        "Performance of the Night", "Fight of the Night",
        "Knockout of the Night", "Submission of the Night",
    }
