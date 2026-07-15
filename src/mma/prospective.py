"""Assemble and persist prospective prediction records for upcoming UFC events.

Ties together `mma.wiki_cards` (event/card parsing), `mma.snapshots`
(current fighter state), and `mma.inference` (the committed ensemble)
into one JSON record per event under `predictions/`, written idempotently
so a re-run never rewrites an already-timestamped prediction.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from mma.inference import build_matchup, predict_symmetrized


def normalize_name(name: str) -> str:
    """Unicode-normalize (NFC) + casefold for exact, non-fuzzy name matching.

    Deliberately does NOT strip accents: "Jose Aldo" and "Jose Aldo"
    (composed vs. decomposed forms of the same string) are equal, but
    "Jose Aldo" and "Jose Aldo" (accent dropped) are not -- per the design,
    matching must be exact-after-normalization, never a fuzzy guess.
    """
    return unicodedata.normalize("NFC", name).strip().casefold()


def build_name_index(fighters: pd.DataFrame) -> dict[str, list[str]]:
    """normalized name -> list of fighter_ids sharing that normalized name."""
    index: dict[str, list[str]] = {}
    for fighter_id, name in zip(fighters["fighter_id"], fighters["name"]):
        if pd.isna(name):
            continue
        index.setdefault(normalize_name(name), []).append(fighter_id)
    return index


def match_fighter_id(
    name: str, index: dict[str, list[str]]
) -> tuple[str | None, str | None]:
    """Exact match only. 0 or >=2 matches are both failures -- never guess."""
    matches = index.get(normalize_name(name), [])
    if len(matches) == 1:
        return matches[0], None
    if len(matches) == 0:
        return None, f"no fighter in fighters.parquet matches name {name!r}"
    return None, f"name {name!r} is ambiguous: matches {len(matches)} fighter ids"


def event_filename(event_name: str, event_date: str) -> str:
    """UFC_<slug>_<date>.json, e.g. 'UFC Fight Night: du Plessis vs. Usman' ->
    UFC_Fight_Night_du_Plessis_vs_Usman_2026-07-18.json
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", event_name).strip("_")
    return f"UFC_{slug}_{event_date}.json"


def predict_fight(
    wiki_fight: dict,
    name_index: dict[str, list[str]],
    snapshots: pd.DataFrame,
    fighters: pd.DataFrame,
    ensemble,
    as_of: pd.Timestamp,
) -> dict:
    """One fight-card entry (from `mma.wiki_cards.parse_fight_card`) -> one
    prediction-record fight dict. Never raises on a bad/unmatched name --
    returns a `skipped: True` record with a reason instead.

    `ensemble` only needs to work with `mma.inference.build_matchup` /
    `predict_symmetrized` -- pass a fake in unit tests to avoid loading the
    real torch checkpoints.
    """
    name_a = wiki_fight["fighter_a_name"]
    name_b = wiki_fight["fighter_b_name"]
    base = {
        "fighter_a_name": name_a,
        "fighter_b_name": name_b,
        "weight_class": wiki_fight.get("weight_class"),
    }

    id_a, reason_a = match_fighter_id(name_a, name_index)
    id_b, reason_b = match_fighter_id(name_b, name_index)
    if id_a is None or id_b is None:
        reasons = [r for r in (reason_a, reason_b) if r]
        return {**base, "skipped": True, "reason": "; ".join(reasons)}

    missing_snapshot = [
        n for n, fid in ((name_a, id_a), (name_b, id_b)) if fid not in snapshots.index
    ]
    if missing_snapshot:
        return {
            **base,
            "skipped": True,
            "reason": f"matched fighter id has no fight history: {missing_snapshot}",
        }

    snap_a, snap_b = snapshots.loc[id_a], snapshots.loc[id_b]
    bio_a, bio_b = fighters.loc[id_a], fighters.loc[id_b]
    title_fight = bool(wiki_fight.get("title_fight", False))
    # Heuristic: UFC main events and title fights are scheduled for 5 rounds,
    # everything else for 3. Wikipedia fight-card tables don't reliably state
    # round count directly, so this is inferred rather than parsed.
    scheduled_rounds = 5 if title_fight or wiki_fight.get("main_event") else 3
    weight_class = wiki_fight.get("weight_class") or "Lightweight"

    matchup_ab = build_matchup(
        snap_a, snap_b, bio_a, bio_b, weight_class, title_fight, scheduled_rounds, as_of
    )
    matchup_ba = build_matchup(
        snap_b, snap_a, bio_b, bio_a, weight_class, title_fight, scheduled_rounds, as_of
    )
    result = predict_symmetrized(ensemble, matchup_ab, matchup_ba)

    return {
        **base,
        "fighter_a_id": id_a,
        "fighter_b_id": id_b,
        "title_fight": title_fight,
        "scheduled_rounds": scheduled_rounds,
        "p_a_wins": float(result["winner_prob"]),
        "method_probs": {
            cls: float(p)
            for cls, p in zip(result["method_classes"], result["method_probs"])
        },
        "round_probs": {
            cls: float(p)
            for cls, p in zip(result["round_classes"], result["round_probs"])
        },
        # Pre-fight Elo at prediction time -- makes the higher-Elo-wins dummy
        # baseline gradeable later without recomputing ratings retroactively.
        "elo_a": float(snap_a["elo_overall"]),
        "elo_b": float(snap_b["elo_overall"]),
        "skipped": False,
    }


def predict_event(
    event: dict,
    wiki_fights: list[dict],
    name_index: dict[str, list[str]],
    snapshots: pd.DataFrame,
    fighters: pd.DataFrame,
    ensemble,
) -> list[dict]:
    """Predict every fight on a card. `event['date']` is an ISO date string
    used as the as-of date for age / days-since-last-fight features."""
    as_of = pd.Timestamp(event["date"])
    return [
        predict_fight(fight, name_index, snapshots, fighters, ensemble, as_of)
        for fight in wiki_fights
    ]


def load_event_record(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_event_prediction(
    predictions_dir: Path,
    event_name: str,
    event_date: str,
    source_page: str,
    model_version: str,
    predicted_at_utc: str,
    fight_predictions: list[dict],
) -> tuple[Path, dict, int]:
    """Idempotently create-or-append an event's prediction record.

    A fight already present in the file (matched by fighter-name pair) is
    NEVER modified -- its prediction and timestamp are the point of this
    whole system. Only fights not already recorded are appended, each
    stamped with its own predicted_at_utc/model_version. Returns
    (path, record, n_new_fights_written).
    """
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    path = predictions_dir / event_filename(event_name, event_date)
    existing = load_event_record(path)

    stamped = []
    for fight in fight_predictions:
        fight = dict(fight)
        fight["predicted_at_utc"] = predicted_at_utc
        fight["model_version"] = model_version
        stamped.append(fight)

    if existing is None:
        record = {
            "event_name": event_name,
            "event_date": event_date,
            "source_page": source_page,
            "predicted_at_utc": predicted_at_utc,
            "model_version": model_version,
            "fights": stamped,
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
        return path, record, len(stamped)

    existing_pairs = {
        (f.get("fighter_a_name"), f.get("fighter_b_name")) for f in existing["fights"]
    }
    new_fights = [
        f for f in stamped
        if (f["fighter_a_name"], f["fighter_b_name"]) not in existing_pairs
    ]
    if new_fights:
        existing["fights"].extend(new_fights)
        path.write_text(json.dumps(existing, indent=2) + "\n")
    return path, existing, len(new_fights)
