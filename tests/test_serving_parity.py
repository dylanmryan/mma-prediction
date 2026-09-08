"""The row served for a future fight must equal the row trained on.

`build_features` (training) and `build_matchup` (serving) must produce
identical feature values for the same matchup as of the same date. This is
value-level, not just column-level: it replays history to just before a
real historical fight, builds the served row from the resulting snapshots,
and compares it against that fight's row in the training table.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma.features import build_features
from mma.history import build_history
from mma.inference import build_matchup
from mma.snapshots import build_snapshots

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists(),
    reason="processed data not built (run scripts/make_dataset.py)",
)


@pytest.fixture(scope="module")
def tables():
    return {
        name: pd.read_parquet(PROCESSED / f"{name}.parquet")
        for name in ("fights", "fight_stats", "fighters", "ratings")
    }


def _target_fight(fights: pd.DataFrame) -> pd.Series:
    """A 2024 fight where both fighters already have several UFC bouts."""
    counts = pd.concat([fights["fighter_a_id"], fights["fighter_b_id"]]).value_counts()
    veterans = set(counts[counts >= 5].index)
    candidates = fights[
        (fights["date"] >= "2024-01-01")
        & (fights["winner"].isin(["a", "b"]))
        & fights["fighter_a_id"].isin(veterans)
        & fights["fighter_b_id"].isin(veterans)
    ]
    assert len(candidates), "no suitable veteran-vs-veteran fight found"
    return candidates.sort_values("date").iloc[0]


def test_served_row_equals_training_row(tables):
    fights, stats, fighters, ratings = (
        tables["fights"], tables["fight_stats"], tables["fighters"], tables["ratings"]
    )
    target = _target_fight(fights)

    past = fights[fights["date"] < target["date"]]
    past_ids = set(past["fight_id"])
    snapshots = build_snapshots(
        past, stats[stats["fight_id"].isin(past_ids)], ratings[ratings["fight_id"].isin(past_ids)]
    )
    indexed = fighters.set_index("fighter_id")
    served = build_matchup(
        snapshots.loc[target["fighter_a_id"]], snapshots.loc[target["fighter_b_id"]],
        indexed.loc[target["fighter_a_id"]], indexed.loc[target["fighter_b_id"]],
        target["weight_class"], bool(target["title_fight"]),
        int(target["scheduled_rounds"]), target["date"],
    )

    history = build_history(fights, stats, ratings)
    trained = build_features(fights, fighters, ratings, history)
    row = trained[trained["fight_id"] == target["fight_id"]]
    assert len(row) == 1
    row = row.iloc[0]

    sign = -1.0 if bool(row["swapped"]) else 1.0
    mismatches = []
    compared = 0
    for column in served.columns:
        if column in ("weight_class", "title_fight", "scheduled_rounds"):
            continue
        assert column in row.index, f"served column {column} missing from the training table"
        if not column.endswith("_diff"):
            continue
        served_value = served.iloc[0][column]
        trained_value = row[column]
        if pd.isna(served_value) and pd.isna(trained_value):
            continue
        compared += 1
        if not (float(served_value) == pytest.approx(sign * float(trained_value), abs=1e-6)):
            mismatches.append((column, float(served_value), sign * float(trained_value)))
    assert compared >= 15, f"only {compared} differential columns compared"
    assert not mismatches, f"served != trained for: {mismatches}"


def test_training_table_and_served_row_have_the_same_columns(tables):
    fights, stats, fighters, ratings = (
        tables["fights"], tables["fight_stats"], tables["fighters"], tables["ratings"]
    )
    target = _target_fight(fights)
    past = fights[fights["date"] < target["date"]]
    past_ids = set(past["fight_id"])
    snapshots = build_snapshots(
        past, stats[stats["fight_id"].isin(past_ids)], ratings[ratings["fight_id"].isin(past_ids)]
    )
    indexed = fighters.set_index("fighter_id")
    served = build_matchup(
        snapshots.loc[target["fighter_a_id"]], snapshots.loc[target["fighter_b_id"]],
        indexed.loc[target["fighter_a_id"]], indexed.loc[target["fighter_b_id"]],
        target["weight_class"], bool(target["title_fight"]),
        int(target["scheduled_rounds"]), target["date"],
    )
    trained = pd.read_parquet(PROCESSED / "features.parquet")
    identifiers = {"fight_id", "date", "swapped", "y_winner", "y_method", "y_finish_round"}
    assert set(served.columns) == set(trained.columns) - identifiers
