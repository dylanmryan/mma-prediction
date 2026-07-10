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
