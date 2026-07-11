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
            "stance": raw["stance"].astype("string").str.strip(),
            "dob": pd.to_datetime(raw["dob"], format="mixed", errors="coerce"),
        }
    )
    if not fighters["fighter_id"].is_unique:
        duplicated = fighters.loc[fighters["fighter_id"].duplicated(), "fighter_id"]
        raise ValueError(f"duplicate fighter ids: {sorted(set(duplicated))[:5]}")
    return fighters.sort_values("fighter_id").reset_index(drop=True)


_NO_CONTEST_METHODS = {"overturned", "could not continue", "dq"}


def _winner_code(winner_id, id_a: str, id_b: str, method_raw) -> str:
    """'a'/'b' from the winning corner; 'draw'/'nc' when there is no winner.

    No-winner fights are no-contests only for the explicit NC method values;
    anything else with no winner (decisions, early-era "Other") is a draw.
    """
    if pd.isna(winner_id):
        text = "" if pd.isna(method_raw) else str(method_raw).strip().lower()
        return "nc" if text in _NO_CONTEST_METHODS else "draw"
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
    fight_ids = raw["fight_id"].astype("string").str.strip()
    if fight_ids.isna().any() or not fight_ids.is_unique:
        raise ValueError("fight_id must be present and unique")
    if ids_a.isna().any() or ids_b.isna().any():
        raise ValueError("fights with missing corner fighter ids")
    method = raw["method"].map(map_method)
    fights = pd.DataFrame(
        {
            "fight_id": fight_ids,
            "date": pd.to_datetime(raw["date"], format="mixed", errors="coerce"),
            "fighter_a_id": ids_a,
            "fighter_b_id": ids_b,
            "winner": [
                _winner_code(winner_id, id_a, id_b, m)
                for winner_id, id_a, id_b, m in zip(
                    raw["winner_id"], ids_a, ids_b, raw["method"]
                )
            ],
            "method": method,
            "method_raw": raw["method"],
            "decision_subtype": raw["method"].map(decision_subtype),
            "scheduled_rounds": pd.to_numeric(
                raw["total_rounds"], errors="coerce"
            ).astype("Int64"),
            "weight_class": raw["division"].map(parse_weight_class),
            "title_fight": pd.to_numeric(raw["title_fight"], errors="coerce")
            .fillna(0)
            .astype(bool),
        }
    )
    # finish_round only for finishes: decisions go the distance by definition,
    # and the raw column stores the last round fought for every fight.
    last_round = pd.to_numeric(raw["finish_round"], errors="coerce").astype("Int64")
    is_finish = fights["method"].isin(["ko_tko", "submission"])
    fights["finish_round"] = last_round.where(is_finish)

    # match_time_sec in the raw data is the clock time WITHIN the final round
    # fought, not total fight duration (verified: 5-round decisions always
    # show exactly 300s there). Derive true elapsed duration_sec instead,
    # approximating every round as a fixed 5 minutes (300s) -- the UFC's
    # standard round length, though not strictly true for old non-title
    # 3-round-cap or historic no-time-limit bouts:
    #   - finish (output finish_round not NA): (finish_round - 1) * 300 + last_round_sec
    #   - true decision (method == "decision"): went the distance by
    #     definition -> scheduled_rounds * 300
    #   - everything else (DQ, Overturned, Could Not Continue, "Other", and
    #     early no-time-limit-era fights): these are NOT finishes by our
    #     method mapping, but scheduled_rounds being present does NOT mean
    #     they went the distance -- most end early (79 of 113 such rows in
    #     the raw data have finish_round < total_rounds). Fall back to the
    #     RAW finish_round column, which is populated for every fight in the
    #     source csv (our own `finish_round` output above is nulled for
    #     non-finishes, so we can't reuse it here).
    last_round_sec = pd.to_numeric(raw["match_time_sec"], errors="coerce")
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

    for column in ("winner", "method", "method_raw", "decision_subtype", "weight_class"):
        fights[column] = fights[column].astype("string")

    columns = [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight", "duration_sec",
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
