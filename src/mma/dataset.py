"""Builders that turn the raw Kaggle UFC CSVs into clean tables.

Source schema: neelagiriaditya/ufc-datasets-1994-2025 (pre-parsed numeric
values, stable hex fighter ids). See the Phase 1 plan addendum for details.
"""
from __future__ import annotations

import pandas as pd


def build_fighters(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter: stable id + biographical fields only.

    Career-aggregate columns (wins, splm, td_avg, ...) are dropped on
    purpose: they are as-of-scrape values and would leak the future if
    joined to historical fights.
    """
    fighters = pd.DataFrame(
        {
            "fighter_id": raw["id"].astype(str).str.strip(),
            "name": raw["name"].astype(str).str.strip(),
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
