"""Each fighter's CURRENT career state (after their last rated fight).

Reuses the history accumulator: replay all rated fights chronologically,
then snapshot every fighter's final state. `days_since_last` is left to
the caller (needs an as-of date); `last_date` is provided instead, and
`years_since_ufc_debut` is dropped for the same reason with `first_date`
provided in its place.

The last Elo and Glicko-2 triples ride along from the ratings table. Elo's
`post_overall` is what `mma.inference.build_matchup` serves as the fighter's
pre-fight rating; the Glicko triple is the same thing for the `trajectory`
block, except that its DEVIATION has to be grown over the days since the last
fight (`mma.glicko.decay_days`) before it is the value that fight would have
trained on.
"""
from __future__ import annotations

import pandas as pd

from mma.history import _SCORES, _FighterState, elo_lookups
from mma import context

# The ratings table's post-fight columns, and the snapshot names they take.
_POST_COLUMNS = {
    "post_overall": "elo_overall",
    "post_striking": "elo_striking",
    "post_grappling": "elo_grappling",
    "post_glicko_mu": "glicko_mu",
    "post_glicko_phi": "glicko_phi",
    "post_glicko_sigma": "glicko_sigma",
}


def build_snapshots(
    fights: pd.DataFrame, stats: pd.DataFrame, ratings: pd.DataFrame,
    bonuses: pd.DataFrame | None = None,
) -> pd.DataFrame:
    stat_lookup = stats.set_index(["fight_id", "corner"]).to_dict("index")
    elo_lookup, post_lookup = elo_lookups(ratings)
    bonus_ids = context.bonus_fights(bonuses)

    states: dict[str, _FighterState] = {}
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for fight in ordered.itertuples(index=False):
        if fight.winner not in _SCORES:
            continue
        score_a, score_b = _SCORES[fight.winner]
        stats_a = stat_lookup.get((fight.fight_id, "a"), {})
        stats_b = stat_lookup.get((fight.fight_id, "b"), {})
        method = fight.method if pd.notna(fight.method) else None
        bonus = str(fight.fight_id) in bonus_ids
        for fighter_id in (fight.fighter_a_id, fight.fighter_b_id):
            states.setdefault(fighter_id, _FighterState())
        allowed = {
            corner: states[fighter_id].allowed()
            for corner, fighter_id in (("a", fight.fighter_a_id),
                                       ("b", fight.fighter_b_id))
        }
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            states[fighter_id].update(
                score, own, opp, method, fight.duration_sec, fight.date,
                elo_lookup.get((fight.fight_id, opp_corner)),
                opp_allowed=allowed[opp_corner],
                pre_elo=elo_lookup.get((fight.fight_id, corner)),
                post_elo=post_lookup.get((fight.fight_id, corner)),
                bonus=bonus,
            )

    present = [c for c in _POST_COLUMNS if c in ratings.columns]
    renames = {c: _POST_COLUMNS[c] for c in present}
    last_elo = (
        ratings.sort_values("date", kind="stable")
        .groupby("fighter_id")[present]
        .last()
        .rename(columns=renames)
        if "date" in ratings.columns
        else ratings.groupby("fighter_id")[present].last().rename(columns=renames)
    )

    rows = {}
    for fighter_id, state in states.items():
        snapshot = state.snapshot(state.last_date)  # days_since_last -> 0, ignored
        snapshot["last_date"] = state.last_date
        snapshot["first_date"] = state.first_date
        rows[fighter_id] = snapshot
    snapshots = pd.DataFrame.from_dict(rows, orient="index")
    snapshots.index.name = "fighter_id"
    snapshots = snapshots.drop(columns=["days_since_last", "years_since_ufc_debut"])
    return snapshots.join(last_elo, how="left")
