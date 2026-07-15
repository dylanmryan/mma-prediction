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
    """Unicode-normalize (NFC) + casefold for exact (tier-1) name matching.

    Deliberately does NOT strip accents: "Jose Aldo" and "Jose Aldo"
    (composed vs. decomposed forms of the same string) are equal, but
    "Jose Aldo" and "Jose Aldo" (accent dropped) are not. Accent bridging
    is tier 2's job -- see `fold_accents` -- and still never guesses.
    """
    return unicodedata.normalize("NFC", name).strip().casefold()


def fold_accents(name: str) -> str:
    """Tier-2 normalization: NFKD-decompose, strip combining marks, casefold.

    Wikipedia renders fighters' names with full diacritics ("Rakić",
    "Uroš") while fighters.parquet stores ufcstats' ASCII transliterations
    ("Rakic", "Uros") -- the single biggest source of skips in the first
    live run. Folding BOTH sides bridges that gap. A folded match must
    still be UNIQUE (two distinct fighters that collide after folding are
    ambiguous and skipped), so this widens the net without ever guessing.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.strip().casefold()


def build_name_index(fighters: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Two-tier lookup: {"exact": normalized name -> fighter_ids,
    "folded": accent-folded name -> fighter_ids}."""
    exact: dict[str, list[str]] = {}
    folded: dict[str, list[str]] = {}
    for fighter_id, name in zip(fighters["fighter_id"], fighters["name"]):
        if pd.isna(name):
            continue
        exact.setdefault(normalize_name(name), []).append(fighter_id)
        folded.setdefault(fold_accents(name), []).append(fighter_id)
    return {"exact": exact, "folded": folded}


def match_fighter_id(
    name: str, index: dict[str, dict[str, list[str]]]
) -> tuple[str | None, str | None, str | None]:
    """Two-tier match; returns (fighter_id, match_tier, skip_reason).

    Tier 1: exact unicode-normalized match. Tier 2 (only when tier 1 finds
    ZERO candidates): accent-folded match on both sides. Either tier must
    yield exactly ONE fighter id -- 0 or >=2 candidates skip the fight with
    a reason, never guess. An exact-tier ambiguity does NOT fall through to
    folding (folding can only make an ambiguous name more ambiguous).
    """
    exact_matches = index["exact"].get(normalize_name(name), [])
    if len(exact_matches) == 1:
        return exact_matches[0], "exact", None
    if len(exact_matches) >= 2:
        return None, None, (
            f"name {name!r} is ambiguous: matches {len(exact_matches)} fighter ids"
        )
    folded_matches = index["folded"].get(fold_accents(name), [])
    if len(folded_matches) == 1:
        return folded_matches[0], "accent_folded", None
    if len(folded_matches) >= 2:
        return None, None, (
            f"name {name!r} is ambiguous after accent folding: "
            f"matches {len(folded_matches)} fighter ids"
        )
    return None, None, (
        f"no fighter in fighters.parquet matches name {name!r} "
        "(exact or accent-folded)"
    )


def event_filename(event_name: str, event_date: str) -> str:
    """UFC_<slug>_<date>.json, e.g. 'UFC Fight Night: du Plessis vs. Usman' ->
    UFC_Fight_Night_du_Plessis_vs_Usman_2026-07-18.json
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", event_name).strip("_")
    return f"UFC_{slug}_{event_date}.json"


def predict_fight(
    wiki_fight: dict,
    name_index: dict[str, dict[str, list[str]]],
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

    id_a, tier_a, reason_a = match_fighter_id(name_a, name_index)
    id_b, tier_b, reason_b = match_fighter_id(name_b, name_index)
    if id_a is None or id_b is None:
        reasons = [r for r in (reason_a, reason_b) if r]
        return {**base, "skipped": True, "reason": "; ".join(reasons)}
    # Weakest tier wins the label: if either side needed accent folding,
    # the whole match is recorded as "accent_folded" for auditability.
    match_tier = "accent_folded" if "accent_folded" in (tier_a, tier_b) else "exact"

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
        "match_tier": match_tier,
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
    name_index: dict[str, dict[str, list[str]]],
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
    """Idempotently create-or-merge an event's prediction record.

    Immutability rule: a fight already present WITH A REAL PREDICTION
    (skipped: false) is NEVER modified -- its probability and timestamp are
    the point of this whole system. Two kinds of writes are allowed on an
    existing file:

    1. Fights not previously recorded are appended, each stamped with its
       own predicted_at_utc/model_version.
    2. Re-attempt policy for skips: an existing `skipped: true` stub
       contains no prediction, only a failure reason, so there is nothing
       whose timestamp integrity could be violated by replacing it. If a
       re-run now produces a real prediction for that same fighter pair
       (e.g. the fighter has since entered fighters.parquet, or the
       matcher improved), the stub is REPLACED by the prediction with a
       fresh predicted_at_utc -- still strictly pre-event, since this only
       ever runs for upcoming cards. A skip re-attempted and still skipped
       leaves the original stub untouched (no churn).

    Returns (path, record, n_fights_written) where n_fights_written counts
    appended + stub-replaced fights.
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

    index_by_pair = {
        (f.get("fighter_a_name"), f.get("fighter_b_name")): i
        for i, f in enumerate(existing["fights"])
    }
    n_written = 0
    for fight in stamped:
        pair = (fight["fighter_a_name"], fight["fighter_b_name"])
        if pair not in index_by_pair:
            existing["fights"].append(fight)
            n_written += 1
            continue
        current = existing["fights"][index_by_pair[pair]]
        if current.get("skipped") and not fight.get("skipped"):
            existing["fights"][index_by_pair[pair]] = fight
            n_written += 1
    if n_written:
        path.write_text(json.dumps(existing, indent=2) + "\n")
    return path, existing, n_written
