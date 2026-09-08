"""Assemble the model-ready feature table.

One row per decisive fight. Corners are deterministically swapped by
md5(fight_id) parity so column order cannot encode the winner (the red
corner wins ~65% of raw fights). Numeric features enter as A-minus-B
differentials plus a few absolutes; missing values stay NaN.

The feature columns themselves are not defined here: `mma.feature_blocks`
owns the registry of named blocks and `mma.serving` owns the builder that
walks it. Both this module and `mma.inference.build_matchup` build their
rows with `serving.feature_row`, so a feature cannot exist in the training
table but be missing (or different) at prediction time.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from mma import serving
from mma.feature_blocks import BASE_BLOCK, state_key_blocks, state_keys

# State keys `_side_frame` computes itself rather than merging in from a
# source table (they need the fight date, the bio row, or another key).
_DERIVED_STATE_KEYS = ("age", "reach_missing", "dob_missing", "southpaw", "debut")


def swap_corner(fight_id: str) -> bool:
    """Deterministic, platform-stable coin flip per fight."""
    return int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 1


def _source_columns(blocks, fighters, ratings, history) -> tuple[list[str], list[str]]:
    """Split the spec's state keys across the tables that provide them.

    Returns the ratings and history merge lists. These used to be two
    hand-maintained module constants, which meant a block whose state key was
    added to the registry but not to both lists produced a silent all-NaN
    column -- and then measured as "no improvement" for the wrong reason. They
    are now derived from `feature_blocks.state_keys`, and a key no table
    provides is a loud `ValueError` naming the key and its block.
    """
    keys = state_keys(blocks)
    from_fighters = [k for k in keys if k in fighters.columns]
    from_ratings = [k for k in keys
                    if k in ratings.columns and k not in from_fighters]
    from_history = [k for k in keys
                    if k in history.columns and k not in from_fighters
                    and k not in from_ratings]
    accounted = set(from_fighters) | set(from_ratings) | set(from_history)
    accounted |= set(_DERIVED_STATE_KEYS)
    missing = [k for k in keys if k not in accounted]
    if missing:
        owners = state_key_blocks(blocks)
        named = ", ".join(f"{k!r} (block {owners[k]!r})" for k in missing)
        raise ValueError(
            f"no source table provides feature state key(s): {named}; "
            "add the column to history/ratings/fighters, or derive it in "
            "mma.features._side_frame"
        )
    return from_ratings, from_history


def _side_frame(fights, fighters, ratings, history, corner: str,
                blocks=(BASE_BLOCK,)) -> pd.DataFrame:
    """One corner's per-fight state: every key `serving.feature_row` reads."""
    elo_features, history_features = _source_columns(
        blocks, fighters, ratings, history
    )
    fighter_col = f"fighter_{corner}_id"
    side = fights[["fight_id", "date", fighter_col]].rename(
        columns={fighter_col: "fighter_id"}
    )
    side = side.merge(fighters, on="fighter_id", how="left")
    side = side.merge(
        ratings[ratings["corner"] == corner][["fight_id"] + elo_features],
        on="fight_id", how="left",
    )
    side = side.merge(
        history[history["corner"] == corner][["fight_id"] + history_features],
        on="fight_id", how="left",
    )
    side["age"] = (side["date"] - side["dob"]).dt.days / 365.25
    side["reach_missing"] = side["reach_cm"].isna()
    side["dob_missing"] = side["dob"].isna()
    side["southpaw"] = (side["stance"] == "Southpaw").fillna(False)
    side["debut"] = serving.debut_flag(side["career_fights"])
    return side


def build_features(fights, fighters, ratings, history,
                   blocks=(BASE_BLOCK,)) -> pd.DataFrame:
    """Targets, identifiers and one `serving.feature_row` per decisive fight.

    The whole table is built in a single vectorised pass: `serving.feature_row`
    is fed the two corners' side *frames* rather than one row's dict at a
    time, so each of its expressions evaluates column-wise. That keeps the
    build at well under a second over ~11k fights and, more importantly,
    preserves the exact dtypes pandas subtraction produces (integer counters
    such as `career_fights_diff` stay int64), which a per-row Python loop
    would silently widen to float. It is still literally the same builder
    serving uses -- `tests/test_serving_parity.py` compares the values.

    `blocks` is the requested feature-block list (see `mma.feature_blocks`);
    it defaults to the `base` v1 contract, so the committed feature table is
    what an unadorned call produces.
    """
    decisive = fights[fights["winner"].isin(["a", "b"])].reset_index(drop=True)
    side_a = _side_frame(decisive, fighters, ratings, history, "a", blocks)
    side_b = _side_frame(decisive, fighters, ratings, history, "b", blocks)

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
    row = serving.feature_row(first, second, context, blocks=blocks)
    return pd.DataFrame({**identifiers, **row})
