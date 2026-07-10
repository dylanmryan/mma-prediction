"""MMA Elo rating engine.

Three parallel ratings per fighter (overall, striking, grappling). The
ratings table stores each fighter's rating *before* every fight, so
downstream features are point-in-time correct by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

INITIAL_RATING = 1500.0
# Grappling-to-striking exchange rate for the striking-share heuristic:
# one takedown or submission attempt ~ 5 significant strikes; one minute
# of control time ~ 1 significant strike-equivalent per the design plan.
GRAPPLING_EVENT_WEIGHT = 5.0


@dataclass(frozen=True)
class EloParams:
    k_early: float = 40.0
    k_late: float = 24.0
    early_fights: int = 5
    finish_bonus: float = 1.2


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the logistic Elo model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def striking_share(sig_landed, td_landed, sub_att, ctrl_sec) -> float:
    """Fraction of the fight that was striking-dominated, in [0, 1].

    Computed from both fighters' combined totals. 0.5 when stats are
    missing or empty (neutral: both style Elos update equally).
    """
    def _num(value) -> float:
        return 0.0 if value is None or pd.isna(value) else float(value)

    sig = _num(sig_landed)
    grappling = GRAPPLING_EVENT_WEIGHT * (_num(td_landed) + _num(sub_att)) + _num(
        ctrl_sec
    ) / 60.0
    total = sig + grappling
    if total == 0:
        return 0.5
    return sig / total


def _k_factor(fight_count: int, params: EloParams) -> float:
    return params.k_early if fight_count < params.early_fights else params.k_late


_SCORES = {"a": (1.0, 0.0), "b": (0.0, 1.0), "draw": (0.5, 0.5)}


def run_elo(
    fights: pd.DataFrame, stats: pd.DataFrame, params: EloParams
) -> pd.DataFrame:
    """Chronological Elo pass. Returns one row per fighter per rated fight.

    No-contests are skipped: no rating change, no fight-count increment.
    """
    combined = (
        stats.groupby("fight_id")[["sig_landed", "td_landed", "sub_att", "ctrl_sec"]]
        .sum(min_count=1)
        .to_dict("index")
    )
    overall: dict[str, float] = {}
    striking: dict[str, float] = {}
    grappling: dict[str, float] = {}
    counts: dict[str, int] = {}

    rows = []
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for fight in ordered.itertuples(index=False):
        if fight.winner not in _SCORES:
            continue
        score_a, score_b = _SCORES[fight.winner]
        id_a, id_b = fight.fighter_a_id, fight.fighter_b_id
        fight_stats = combined.get(fight.fight_id, {})
        share = striking_share(
            fight_stats.get("sig_landed"),
            fight_stats.get("td_landed"),
            fight_stats.get("sub_att"),
            fight_stats.get("ctrl_sec"),
        )
        bonus = (
            params.finish_bonus
            if pd.notna(fight.method) and fight.method in ("ko_tko", "submission")
            else 1.0
        )

        pre = {}
        for fighter in (id_a, id_b):
            pre[fighter] = (
                overall.get(fighter, INITIAL_RATING),
                striking.get(fighter, INITIAL_RATING),
                grappling.get(fighter, INITIAL_RATING),
                counts.get(fighter, 0),
            )

        expected_a = expected_score(pre[id_a][0], pre[id_b][0])
        expected_striking_a = expected_score(pre[id_a][1], pre[id_b][1])
        expected_grappling_a = expected_score(pre[id_a][2], pre[id_b][2])

        for fighter, score, exp_o, exp_s, exp_g in (
            (id_a, score_a, expected_a, expected_striking_a, expected_grappling_a),
            (id_b, score_b, 1 - expected_a, 1 - expected_striking_a, 1 - expected_grappling_a),
        ):
            k = _k_factor(pre[fighter][3], params) * bonus
            overall[fighter] = pre[fighter][0] + k * (score - exp_o)
            striking[fighter] = pre[fighter][1] + k * share * (score - exp_s)
            grappling[fighter] = pre[fighter][2] + k * (1 - share) * (score - exp_g)
            counts[fighter] = pre[fighter][3] + 1

        for fighter, corner in ((id_a, "a"), (id_b, "b")):
            rows.append(
                {
                    "fight_id": fight.fight_id,
                    "date": fight.date,
                    "fighter_id": fighter,
                    "corner": corner,
                    "pre_overall": pre[fighter][0],
                    "pre_striking": pre[fighter][1],
                    "pre_grappling": pre[fighter][2],
                    "pre_fights": pre[fighter][3],
                    "post_overall": overall[fighter],
                    "post_striking": striking[fighter],
                    "post_grappling": grappling[fighter],
                }
            )

    ratings = pd.DataFrame(rows)
    for column in ("fight_id", "fighter_id", "corner"):
        ratings[column] = ratings[column].astype("string")
    return ratings
