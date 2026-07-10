"""Builders that turn the raw Kaggle UFC CSVs into clean tables.

Source schema: neelagiriaditya/ufc-datasets-1994-2025 (pre-parsed numeric
values, stable hex fighter ids). See the Phase 1 plan addendum for details.
"""
from __future__ import annotations

import pandas as pd

from mma.labels import decision_subtype, map_method, parse_weight_class


def build_fighters(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter: stable id + biographical fields only.

    Career-aggregate columns (wins, splm, td_avg, ...) are dropped on
    purpose: they are as-of-scrape values and would leak the future if
    joined to historical fights.
    """
    ids = raw["id"].astype("string").str.strip()
    if ids.isna().any():
        raise ValueError(f"{int(ids.isna().sum())} fighter rows have missing ids")
    fighters = pd.DataFrame(
        {
            "fighter_id": ids,
            "name": raw["name"].astype("string").str.strip(),
            "height_cm": pd.to_numeric(raw["height"], errors="coerce"),
            "reach_cm": pd.to_numeric(raw["reach"], errors="coerce"),
            "stance": raw["stance"],
            "dob": pd.to_datetime(raw["dob"], format="mixed", errors="coerce"),
        }
    )
    if not fighters["fighter_id"].is_unique:
        duplicated = fighters.loc[fighters["fighter_id"].duplicated(), "fighter_id"]
        raise ValueError(f"duplicate fighter ids: {sorted(set(duplicated))[:5]}")
    return fighters.sort_values("fighter_id").reset_index(drop=True)


def _winner_code(winner_id, id_a: str, id_b: str, method: str | None) -> str:
    """'a'/'b' from the winning corner; 'draw'/'nc' when there is no winner."""
    if pd.isna(winner_id):
        return "draw" if method == "decision" else "nc"
    winner = str(winner_id).strip()
    if winner == id_a:
        return "a"
    if winner == id_b:
        return "b"
    return "nc"


def build_fights(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fight: ids, date, winner code, targets, context."""
    ids_a = raw["r_id"].astype("string").str.strip()
    ids_b = raw["b_id"].astype("string").str.strip()
    method = raw["method"].map(map_method)
    fights = pd.DataFrame(
        {
            "fight_id": raw["fight_id"].astype("string").str.strip(),
            "date": pd.to_datetime(raw["date"], format="mixed", errors="coerce"),
            "fighter_a_id": ids_a,
            "fighter_b_id": ids_b,
            "winner": [
                _winner_code(winner_id, id_a, id_b, m)
                for winner_id, id_a, id_b, m in zip(
                    raw["winner_id"], ids_a, ids_b, method
                )
            ],
            "method": method,
            "method_raw": raw["method"],
            "decision_subtype": raw["method"].map(decision_subtype),
            "scheduled_rounds": pd.to_numeric(
                raw["total_rounds"], errors="coerce"
            ).astype("Int64"),
            "weight_class": raw["division"].map(parse_weight_class),
            "title_fight": raw["title_fight"].fillna(0).astype(bool),
        }
    )
    # finish_round only for finishes: decisions go the distance by definition,
    # and the raw column stores the last round fought for every fight.
    last_round = pd.to_numeric(raw["finish_round"], errors="coerce").astype("Int64")
    is_finish = fights["method"].isin(["ko_tko", "submission"])
    fights["finish_round"] = last_round.where(is_finish)

    columns = [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight",
    ]
    return (
        fights[columns]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
