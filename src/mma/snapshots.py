"""Each fighter's CURRENT career state (after their last rated fight).

Reuses the history accumulator: replay all rated fights chronologically,
then snapshot every fighter's final state. `days_since_last` is left to
the caller (needs an as-of date); `last_date` is provided instead.
"""
from __future__ import annotations

import pandas as pd

from mma.history import _SCORES, _FighterState


def build_snapshots(
    fights: pd.DataFrame, stats: pd.DataFrame, ratings: pd.DataFrame
) -> pd.DataFrame:
    stat_lookup = stats.set_index(["fight_id", "corner"]).to_dict("index")
    elo_lookup = ratings.set_index(["fight_id", "corner"])["pre_overall"].to_dict()

    states: dict[str, _FighterState] = {}
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for fight in ordered.itertuples(index=False):
        if fight.winner not in _SCORES:
            continue
        score_a, score_b = _SCORES[fight.winner]
        stats_a = stat_lookup.get((fight.fight_id, "a"), {})
        stats_b = stat_lookup.get((fight.fight_id, "b"), {})
        method = fight.method if pd.notna(fight.method) else None
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            states.setdefault(fighter_id, _FighterState()).update(
                score, own, opp, method, fight.duration_sec, fight.date,
                elo_lookup.get((fight.fight_id, opp_corner)),
            )

    last_elo = (
        ratings.sort_values("date", kind="stable")
        .groupby("fighter_id")[["post_overall", "post_striking", "post_grappling"]]
        .last()
        .rename(columns={
            "post_overall": "elo_overall",
            "post_striking": "elo_striking",
            "post_grappling": "elo_grappling",
        })
        if "date" in ratings.columns
        else ratings.groupby("fighter_id")[["post_overall", "post_striking", "post_grappling"]]
        .last()
        .rename(columns={
            "post_overall": "elo_overall",
            "post_striking": "elo_striking",
            "post_grappling": "elo_grappling",
        })
    )

    rows = {}
    for fighter_id, state in states.items():
        snapshot = state.snapshot(state.last_date)  # days_since_last -> 0, ignored
        snapshot["last_date"] = state.last_date
        rows[fighter_id] = snapshot
    snapshots = pd.DataFrame.from_dict(rows, orient="index")
    snapshots.index.name = "fighter_id"
    snapshots = snapshots.drop(columns=["days_since_last"])
    return snapshots.join(last_elo, how="left")
