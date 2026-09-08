"""The row served for a future fight must equal the row trained on.

`build_features` (training) and `build_matchup` (serving) must produce
identical feature values for the same matchup as of the same date. This is
value-level, not just column-level: it replays history to just before a
real historical fight, builds the served row from the resulting snapshots,
and compares it against that fight's row in the training table.
"""
import bisect
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from mma.feature_blocks import BASE_BLOCK, spec_for
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


def _prior_bouts(fights: pd.DataFrame) -> "callable":
    """(fighter_id, date) -> number of that fighter's bouts strictly before it."""
    ordered = fights.sort_values("date", kind="stable")
    dates: dict[str, list] = defaultdict(list)
    for fight in ordered.itertuples(index=False):
        dates[fight.fighter_a_id].append(fight.date)
        dates[fight.fighter_b_id].append(fight.date)
    return lambda fighter_id, date: bisect.bisect_left(dates[fighter_id], date)


def _target_fight(fights: pd.DataFrame) -> pd.Series:
    """The 2024+ fight whose two fighters have the deepest records so far.

    "Veteran" has to mean bouts BEFORE the target date: counting career
    totals (which this did originally) let a fighter qualify on bouts that
    had not happened yet, and picked a matchup with 2 and 1 prior bouts --
    most per-corner columns were then NaN on both sides and silently skipped
    by the value comparison. Maximising min(prior_a, prior_b) picks a pair
    with real history on both sides.
    """
    prior = _prior_bouts(fights)
    candidates = fights[
        (fights["date"] >= "2024-01-01") & fights["winner"].isin(["a", "b"])
    ].sort_values(["date", "fight_id"], kind="stable")
    assert len(candidates), "no 2024+ decisive fight found"
    depth = candidates.apply(
        lambda f: min(prior(f["fighter_a_id"], f["date"]),
                      prior(f["fighter_b_id"], f["date"])),
        axis=1,
    )
    return candidates.loc[depth.idxmax()]


def test_the_target_matchup_has_real_history_on_both_sides(tables):
    """A shallow matchup leaves most columns NaN on both sides, which the
    value comparison skips -- the test would then pass while comparing
    almost nothing."""
    fights = tables["fights"]
    target = _target_fight(fights)
    prior = _prior_bouts(fights)
    for corner in ("a", "b"):
        bouts = prior(target[f"fighter_{corner}_id"], target["date"])
        assert bouts >= 5, f"corner {corner} has only {bouts} prior bouts"


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

    # The training table may have swapped corners for this fight; the served
    # row is always A-vs-B. Differentials flip sign, per-corner columns swap
    # which corner they came from, and the derived flags are symmetric.
    sign = -1.0 if bool(row["swapped"]) else 1.0
    sa, sb = ("b", "a") if bool(row["swapped"]) else ("a", "b")

    spec = spec_for([BASE_BLOCK])
    diff_columns = [f"{stem}_diff" for _, stem in spec.differentials]
    corner_pairs = [
        (f"{stem}_a", f"{stem}_{sa}") for stem in spec.absolutes + spec.booleans
    ] + [
        (f"{stem}_b", f"{stem}_{sb}") for stem in spec.absolutes + spec.booleans
    ]
    flag_columns = [name for name, _ in spec.derived_booleans]
    boolean_stems = set(spec.booleans)

    for column in served.columns:
        if column in ("weight_class", "title_fight", "scheduled_rounds"):
            continue
        assert column in row.index, f"served column {column} missing from the training table"

    mismatches = []
    compared = 0

    def compare(served_column, trained_column, expected):
        """Count a column as compared unless it is NaN on BOTH sides."""
        nonlocal compared
        got, want = served.iloc[0][served_column], row[trained_column]
        if pd.isna(got) and pd.isna(want):
            return
        compared += 1
        if got != expected(want):
            mismatches.append((served_column, got, expected(want)))

    for column in diff_columns:
        compare(column, column,
                lambda v: pytest.approx(sign * float(v), abs=1e-6))
    for served_column, trained_column in corner_pairs:
        stem = served_column.rsplit("_", 1)[0]
        if stem in boolean_stems:
            compare(served_column, trained_column, lambda v: bool(v))
        else:
            compare(served_column, trained_column,
                    lambda v: pytest.approx(float(v), abs=1e-6))
    for column in flag_columns:
        compare(column, column, lambda v: bool(v))

    # A column that collapses to NaN on both sides is skipped above, which is
    # exactly how a silently-unpopulated feature would hide here: require all
    # but a couple of the spec's columns to have actually been compared.
    expected_columns = len(diff_columns) + len(corner_pairs) + len(flag_columns)
    assert compared >= expected_columns - 2, (
        f"only {compared} of {expected_columns} spec columns compared"
    )
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
