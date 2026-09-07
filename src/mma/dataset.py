"""Builders that turn the raw Kaggle UFC CSVs into clean tables.

Source schema (rebuilt 2026-08-11): neelagiriaditya/ufc-datasets-1994-2025 —
master.csv (one row per fight incl. fight-total stats), fighter.csv,
round.csv (per-round stats), fighter_bonus.csv. Stable ufcstats hex ids
throughout. See docs/superpowers/plans/2026-09-06-sp0-repair-ingestion.md.
"""
from __future__ import annotations

import pandas as pd

from mma.labels import decision_subtype, map_method, parse_scheduled_rounds, parse_weight_class
from mma.parsing import parse_height_inches, parse_mmss_seconds


_INCH_CM = 2.54


def build_fighters(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter: stable id + biographical fields only.

    Career-aggregate columns (slpm, td_avg, ...) and weight_lbs are dropped
    on purpose: they are as-of-scrape values and would leak the future if
    joined to historical fights. Height arrives as `5' 10"` text and reach
    as inches; both are converted to centimetres to keep the processed
    schema identical to the pre-2026 one.
    """
    ids = raw["fighter_id"].astype("string").str.strip()
    if ids.isna().any():
        raise ValueError(f"{int(ids.isna().sum())} fighter rows have missing ids")
    height_in = raw["height"].map(parse_height_inches)
    fighters = pd.DataFrame(
        {
            "fighter_id": ids,
            "name": raw["fighter_name"].astype("string").str.strip(),
            "height_cm": pd.to_numeric(height_in, errors="coerce") * _INCH_CM,
            "reach_cm": pd.to_numeric(raw["reach_inches"], errors="coerce") * _INCH_CM,
            "stance": raw["stance"].astype("string").str.strip(),
            "dob": pd.to_datetime(raw["dob"], format="mixed", errors="coerce"),
        }
    )
    if not fighters["fighter_id"].is_unique:
        duplicated = fighters.loc[fighters["fighter_id"].duplicated(), "fighter_id"]
        raise ValueError(f"duplicate fighter ids: {sorted(set(duplicated))[:5]}")
    return fighters.sort_values("fighter_id").reset_index(drop=True)


def _winner_code(status, winner_id, id_a: str, id_b: str) -> str:
    """'a'/'b' from the winning corner; 'draw'/'nc' from result_status.

    result_status is authoritative for no-winner fights ('draw',
    'no_contest'); a 'win' whose winner_id matches neither corner is
    treated as a no-contest rather than guessed.
    """
    text = "" if pd.isna(status) else str(status).strip().lower()
    if text == "draw":
        return "draw"
    if text == "no_contest":
        return "nc"
    winner = None if pd.isna(winner_id) else str(winner_id).strip()
    if winner == id_a:
        return "a"
    if winner == id_b:
        return "b"
    return "nc"


def _require_unique_fight_ids(raw: pd.DataFrame) -> pd.Series:
    fight_ids = raw["fight_id"].astype("string").str.strip()
    if fight_ids.isna().any() or not fight_ids.is_unique:
        raise ValueError("fight_id must be present and unique")
    return fight_ids


def build_fights(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fight from master.csv: ids, date, winner code, targets, context."""
    ids_a = raw["r_fighter_id"].astype("string").str.strip()
    ids_b = raw["b_fighter_id"].astype("string").str.strip()
    fight_ids = _require_unique_fight_ids(raw)
    if ids_a.isna().any() or ids_b.isna().any():
        raise ValueError("fights with missing corner fighter ids")
    method = raw["method"].map(map_method)
    fights = pd.DataFrame(
        {
            "fight_id": fight_ids,
            "date": pd.to_datetime(raw["event_date"], format="mixed", errors="coerce"),
            "fighter_a_id": ids_a,
            "fighter_b_id": ids_b,
            "winner": [
                _winner_code(status, winner_id, id_a, id_b)
                for status, winner_id, id_a, id_b in zip(
                    raw["result_status"], raw["winner_id"], ids_a, ids_b
                )
            ],
            "method": method,
            "method_raw": raw["method"],
            "decision_subtype": raw["method"].map(decision_subtype),
            "scheduled_rounds": pd.array(
                raw["time_format"].map(parse_scheduled_rounds).tolist(), dtype="Int64"
            ),
            "weight_class": raw["weight_class"].map(parse_weight_class),
            "title_fight": pd.to_numeric(raw["title_fight"], errors="coerce")
            .fillna(0)
            .astype(bool),
            "referee": raw["referee"],
            "event_id": raw["event_id"],
            "location": raw["event_location"],
        }
    )
    # finish_round only for finishes: decisions go the distance by definition,
    # and the raw column stores the last round fought for every fight.
    last_round = pd.to_numeric(raw["finish_round"], errors="coerce").astype("Int64")
    is_finish = fights["method"].isin(["ko_tko", "submission"])
    fights["finish_round"] = last_round.where(is_finish)

    # finish_time is the clock WITHIN the final round fought ("m:ss"), not
    # total duration. Derive elapsed duration_sec with 5-minute rounds:
    #   - finish: (finish_round - 1) * 300 + final-round clock
    #   - true decision: scheduled_rounds * 300 (went the distance)
    #   - everything else (DQ, Overturned, Could Not Continue, "Other",
    #     no-time-limit era): most end early, so fall back to the raw last
    #     round fought + clock.
    last_round_sec = pd.to_numeric(
        raw["finish_time"].map(parse_mmss_seconds), errors="coerce"
    )
    raw_last_round = last_round.astype("Float64")
    duration_sec = pd.Series(pd.NA, index=fights.index, dtype="Float64")
    duration_sec = duration_sec.where(
        fights["finish_round"].isna(),
        (fights["finish_round"].astype("Float64") - 1) * 300 + last_round_sec,
    )
    distance_mask = duration_sec.isna() & (fights["method"] == "decision")
    duration_sec = duration_sec.where(
        ~distance_mask, fights["scheduled_rounds"].astype("Float64") * 300
    )
    fallback_mask = duration_sec.isna()
    fallback = (raw_last_round - 1) * 300 + last_round_sec
    duration_sec = duration_sec.where(~fallback_mask, fallback)
    fights["duration_sec"] = duration_sec

    for column in (
        "winner", "method", "method_raw", "decision_subtype", "weight_class",
        "referee", "event_id", "location",
    ):
        fights[column] = fights[column].astype("string")

    columns = [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight", "duration_sec",
        "referee", "event_id", "location",
    ]
    return (
        fights[columns]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


_STAT_COLUMNS = {
    # output name -> raw column suffix (dataset spells "attempted" as "atmpted")
    "kd": "kd",
    "sig_landed": "sig_str_landed",
    "sig_attempted": "sig_str_atmpted",
    "total_landed": "total_str_landed",
    "total_attempted": "total_str_atmpted",
    "td_landed": "td_landed",
    "td_attempted": "td_atmpted",
    "sub_att": "sub_att",
    "ctrl_sec": "ctrl",
}


def build_fight_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Two rows per fight (one per fighter) with in-fight performance stats."""
    fight_ids = raw["fight_id"].astype("string").str.strip()
    if fight_ids.isna().any() or not fight_ids.is_unique:
        raise ValueError("fight_id must be present and unique")
    frames = []
    for corner, prefix, id_column in (("a", "r_", "r_id"), ("b", "b_", "b_id")):
        frame = pd.DataFrame(
            {
                "fight_id": fight_ids,
                "fighter_id": raw[id_column].astype("string").str.strip(),
                "corner": pd.Series(corner, index=raw.index, dtype="string"),
            }
        )
        for out_name, suffix in _STAT_COLUMNS.items():
            frame[out_name] = pd.to_numeric(raw[prefix + suffix], errors="coerce")
        frames.append(frame)
    stats = pd.concat(frames, ignore_index=True)
    return stats.sort_values(["fight_id", "corner"]).reset_index(drop=True)
