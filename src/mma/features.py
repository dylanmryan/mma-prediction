"""Assemble the model-ready feature table.

One row per decisive fight. Corners are deterministically swapped by
md5(fight_id) parity so column order cannot encode the winner (the red
corner wins ~65% of raw fights). Numeric features enter as A-minus-B
differentials plus a few absolutes; missing values stay NaN.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_HISTORY_FEATURES = [
    "career_fights", "career_wins", "career_win_rate", "career_finish_rate",
    "kd_pf", "sub_att_pf", "td_landed_pf", "td_acc", "td_def",
    "sig_pm", "sig_absorbed_pm", "ctrl_share", "streak", "days_since_last",
    "last5_win_rate", "last5_avg_opp_elo",
]
_ELO_FEATURES = ["pre_overall", "pre_striking", "pre_grappling", "pre_fights"]
_DIFF_RENAMES = {"pre_overall": "elo", "pre_striking": "striking_elo",
                 "pre_grappling": "grappling_elo", "pre_fights": "elo_fights"}


def swap_corner(fight_id: str) -> bool:
    """Deterministic, platform-stable coin flip per fight."""
    return int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 1


def _side_frame(fights, fighters, ratings, history, corner: str) -> pd.DataFrame:
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
    side["debut"] = side["career_fights"].fillna(0) == 0
    return side


def build_features(fights, fighters, ratings, history) -> pd.DataFrame:
    decisive = fights[fights["winner"].isin(["a", "b"])].reset_index(drop=True)
    side_a = _side_frame(decisive, fighters, ratings, history, "a")
    side_b = _side_frame(decisive, fighters, ratings, history, "b")

    swapped = decisive["fight_id"].map(swap_corner).to_numpy(dtype=bool)
    # positional row-swap: both frames share identical columns and index
    first = side_a.copy()
    second = side_b.copy()
    first.loc[swapped] = side_b.loc[swapped].values
    second.loc[swapped] = side_a.loc[swapped].values

    features = pd.DataFrame(
        {
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
            "weight_class": decisive["weight_class"].astype("string"),
            "title_fight": decisive["title_fight"],
            "scheduled_rounds": decisive["scheduled_rounds"],
        }
    )

    numeric = (
        {name: name for name in _HISTORY_FEATURES}
        | {name: _DIFF_RENAMES[name] for name in _ELO_FEATURES}
        | {"height_cm": "height", "reach_cm": "reach", "age": "age"}
    )
    for source, out in numeric.items():
        features[f"{out}_diff"] = pd.to_numeric(
            first[source], errors="coerce"
        ) - pd.to_numeric(second[source], errors="coerce")

    for side, frame in (("a", first), ("b", second)):
        features[f"age_{side}"] = frame["age"]
        features[f"career_fights_{side}"] = frame["career_fights"]
        features[f"reach_missing_{side}"] = frame["reach_missing"].astype(bool)
        features[f"dob_missing_{side}"] = frame["dob_missing"].astype(bool)
        features[f"southpaw_{side}"] = frame["southpaw"].astype(bool)
        features[f"debut_{side}"] = frame["debut"].astype(bool)
    features["debut_matchup"] = features["debut_a"] ^ features["debut_b"]
    features["stance_mismatch"] = features["southpaw_a"] ^ features["southpaw_b"]
    return features
