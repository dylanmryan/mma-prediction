"""Assemble the model-ready feature table.

One row per decisive fight. Corners are deterministically swapped by
md5(fight_id) parity so column order cannot encode the winner (the red
corner wins ~65% of raw fights). Numeric features enter as A-minus-B
differentials plus a few absolutes; missing values stay NaN.

The feature columns themselves are not defined here: `mma.serving` owns
the contract and both this module and `mma.inference.build_matchup` build
their rows with `serving.feature_row`, so a feature cannot exist in the
training table but be missing (or different) at prediction time.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from mma import serving

# The state keys `serving.feature_row` reads that come from the Elo ratings
# table and the career-history accumulator respectively.
_ELO_FEATURES = ["pre_overall", "pre_striking", "pre_grappling", "pre_fights"]
_HISTORY_FEATURES = [
    "career_fights", "career_wins", "career_win_rate", "career_finish_rate",
    "kd_pf", "sub_att_pf", "td_landed_pf", "td_acc", "td_def",
    "sig_pm", "sig_absorbed_pm", "ctrl_share", "streak", "days_since_last",
    "last5_win_rate", "last5_avg_opp_elo",
]


def swap_corner(fight_id: str) -> bool:
    """Deterministic, platform-stable coin flip per fight."""
    return int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 1


def _side_frame(fights, fighters, ratings, history, corner: str) -> pd.DataFrame:
    """One corner's per-fight state: every key `serving.feature_row` reads."""
    fighter_col = f"fighter_{corner}_id"
    side = fights[["fight_id", "date", fighter_col]].rename(
        columns={fighter_col: "fighter_id"}
    )
    side = side.merge(fighters, on="fighter_id", how="left")
    side = side.merge(
        ratings[ratings["corner"] == corner][["fight_id"] + _ELO_FEATURES],
        on="fight_id", how="left",
    )
    side = side.merge(
        history[history["corner"] == corner][["fight_id"] + _HISTORY_FEATURES],
        on="fight_id", how="left",
    )
    side["age"] = (side["date"] - side["dob"]).dt.days / 365.25
    side["reach_missing"] = side["reach_cm"].isna()
    side["dob_missing"] = side["dob"].isna()
    side["southpaw"] = (side["stance"] == "Southpaw").fillna(False)
    side["debut"] = serving.debut_flag(side["career_fights"])
    return side


def build_features(fights, fighters, ratings, history) -> pd.DataFrame:
    """Targets, identifiers and one `serving.feature_row` per decisive fight.

    The whole table is built in a single vectorised pass: `serving.feature_row`
    is fed the two corners' side *frames* rather than one row's dict at a
    time, so each of its expressions evaluates column-wise. That keeps the
    build at well under a second over ~11k fights and, more importantly,
    preserves the exact dtypes pandas subtraction produces (integer counters
    such as `career_fights_diff` stay int64), which a per-row Python loop
    would silently widen to float. It is still literally the same builder
    serving uses -- `tests/test_serving_parity.py` compares the values.
    """
    decisive = fights[fights["winner"].isin(["a", "b"])].reset_index(drop=True)
    side_a = _side_frame(decisive, fighters, ratings, history, "a")
    side_b = _side_frame(decisive, fighters, ratings, history, "b")

    swapped = decisive["fight_id"].map(swap_corner).to_numpy(dtype=bool)
    # positional row-swap: both frames share identical columns and index
    first = side_a.copy()
    second = side_b.copy()
    first.loc[swapped] = side_b.loc[swapped].values
    second.loc[swapped] = side_a.loc[swapped].values

    identifiers = {
        "fight_id": decisive["fight_id"],
        "date": decisive["date"],
        "swapped": swapped,
        "y_winner": np.where(
            swapped,
            (decisive["winner"] == "b").astype(int),
            (decisive["winner"] == "a").astype(int),
        ),
        "y_method": decisive["method"],
        "y_finish_round": decisive["finish_round"]
        .map(lambda r: "45" if pd.notna(r) and r >= 4 else (str(int(r)) if pd.notna(r) else None))
        .astype("string"),
    }
    context = {
        "weight_class": decisive["weight_class"].astype("string"),
        "title_fight": decisive["title_fight"],
        "scheduled_rounds": decisive["scheduled_rounds"],
    }
    return pd.DataFrame({**identifiers, **serving.feature_row(first, second, context)})
