"""Grade past prospective predictions against actual results.

For each prediction file in predictions/ whose event_date has passed, looks
up each fight's actual result in data/processed/fights.parquet (matched on
the unordered pair {fighter_a_id, fighter_b_id} + a fight date within
GRADE_WINDOW_DAYS of the event's scheduled date). Graded fights get grading
fields appended IN PLACE -- the original prediction fields (p_a_wins,
method_probs, ...) are never touched, only new keys are added. Fights with
no matching result yet stay pending and are retried on the next run (the
Kaggle mirror lags real events by days to weeks -- this is expected, not an
error).

Also (re)writes predictions/track_record.json: per-model_version and
overall accuracy/log-loss/Brier over all graded fights, plus the coin-flip
and higher-Elo-wins-dummy comparison baselines graded alongside the model.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "predictions"
FIGHTS_PARQUET = ROOT / "data" / "processed" / "fights.parquet"
GRADE_WINDOW_DAYS = 3
# Coin flip's log-loss/Brier are the same constant for every fight (p=0.5);
# only its "accuracy" (tie-break-to-corner-A) varies per fight.
COIN_FLIP_LOG_LOSS = math.log(2.0)
COIN_FLIP_BRIER = 0.25


def find_result(fight: dict, fights: pd.DataFrame, event_date: str) -> pd.Series | None:
    """Match a predicted fight to its actual result row in fights.parquet.

    Matches on the unordered pair {fighter_a_id, fighter_b_id} plus a fight
    date within GRADE_WINDOW_DAYS of the event's scheduled date (handles
    fights bumped a day or two, or dark/off-card additions). Returns None
    (still pending) if nothing matches -- never raises. If multiple rows
    match (shouldn't normally happen), the closest by date wins.
    """
    id_a, id_b = fight.get("fighter_a_id"), fight.get("fighter_b_id")
    if id_a is None or id_b is None or fights.empty:
        return None
    event_ts = pd.Timestamp(event_date)
    same_pair = (
        ((fights["fighter_a_id"] == id_a) & (fights["fighter_b_id"] == id_b))
        | ((fights["fighter_a_id"] == id_b) & (fights["fighter_b_id"] == id_a))
    )
    within_window = (fights["date"] - event_ts).abs().dt.days <= GRADE_WINDOW_DAYS
    candidates = fights[same_pair & within_window]
    if candidates.empty:
        return None
    candidates = candidates.assign(_delta=(candidates["date"] - event_ts).abs())
    return candidates.sort_values("_delta").iloc[0]


def grade_fight(fight: dict, result: pd.Series) -> dict:
    """Pure grading math for one fight: model + coin-flip + higher-Elo-dummy.

    `result` is the matched fights.parquet row. Its `winner` ('a'/'b'/'draw')
    is relative to the RESULT ROW's own corner order, which may differ from
    the prediction's fighter_a/b order (Wikipedia and the dataset don't
    guarantee the same fighter is listed first) -- so this resolves
    everything to fighter ids and never assumes corners line up.
    """
    predicted_a, predicted_b = fight["fighter_a_id"], fight["fighter_b_id"]
    if result["winner"] == "a":
        actual_winner_id = result["fighter_a_id"]
    elif result["winner"] == "b":
        actual_winner_id = result["fighter_b_id"]
    else:
        actual_winner_id = None  # draw / no contest

    p_a = float(fight["p_a_wins"])
    if actual_winner_id is None:
        # Draws/no-contests: log-loss/Brier scored against y=0.5 (standard
        # treatment); accuracy is undefined (excluded from the aggregate).
        y = 0.5
        correct = None
        coin_flip_correct = None
        elo_dummy_correct = None
    else:
        y = 1.0 if actual_winner_id == predicted_a else 0.0
        predicted_winner_id = predicted_a if p_a >= 0.5 else predicted_b
        correct = predicted_winner_id == actual_winner_id
        coin_flip_correct = actual_winner_id == predicted_a  # always "picks" A
        elo_a, elo_b = fight.get("elo_a"), fight.get("elo_b")
        if elo_a is None or elo_b is None or elo_a == elo_b:
            elo_dummy_correct = None
        else:
            elo_favorite_id = predicted_a if elo_a > elo_b else predicted_b
            elo_dummy_correct = elo_favorite_id == actual_winner_id

    p_clipped = min(max(p_a, 1e-12), 1 - 1e-12)
    log_loss_contribution = -(
        y * math.log(p_clipped) + (1 - y) * math.log(1 - p_clipped)
    )
    brier_contribution = (p_a - y) ** 2

    return {
        "actual_winner": actual_winner_id if actual_winner_id is not None else "draw",
        "correct": correct,
        "log_loss_contribution": log_loss_contribution,
        "brier_contribution": brier_contribution,
        # Comparison baselines, graded alongside the model on the same fight
        # (higher-Elo dummy uses elo_a/elo_b as recorded at prediction time).
        "coin_flip_correct": coin_flip_correct,
        "coin_flip_log_loss_contribution": COIN_FLIP_LOG_LOSS,
        "coin_flip_brier_contribution": COIN_FLIP_BRIER,
        "elo_dummy_correct": elo_dummy_correct,
    }


def _summarize(predicted: list[dict], graded: list[dict], first_prediction_at=None) -> dict:
    scored = [f for f in graded if f.get("correct") is not None]
    summary = {
        "n_predicted": len(predicted),
        "n_graded": len(graded),
        "accuracy": (
            sum(1 for f in scored if f["correct"]) / len(scored) if scored else None
        ),
        "log_loss": (
            sum(f["log_loss_contribution"] for f in graded) / len(graded)
            if graded else None
        ),
        "brier": (
            sum(f["brier_contribution"] for f in graded) / len(graded)
            if graded else None
        ),
    }
    if first_prediction_at is not None:
        summary["first_prediction_at"] = first_prediction_at
    return summary


def aggregate_track_record(event_records: list[dict], generated_at_utc: str) -> dict:
    """Pure aggregation over event records whose fights already carry
    grading fields (see `grade_fight`). No file/network I/O."""
    by_version: dict[str, dict] = {}
    overall_predicted: list[dict] = []
    overall_graded: list[dict] = []

    for record in event_records:
        for fight in record.get("fights", []):
            if fight.get("skipped"):
                continue
            version = fight.get("model_version")
            bucket = by_version.setdefault(
                version, {"predicted": [], "graded": [], "first_prediction_at": None}
            )
            bucket["predicted"].append(fight)
            overall_predicted.append(fight)
            predicted_at = fight.get("predicted_at_utc")
            if predicted_at and (
                bucket["first_prediction_at"] is None
                or predicted_at < bucket["first_prediction_at"]
            ):
                bucket["first_prediction_at"] = predicted_at
            if "actual_winner" in fight:
                bucket["graded"].append(fight)
                overall_graded.append(fight)

    model_versions = {
        version: _summarize(b["predicted"], b["graded"], b["first_prediction_at"])
        for version, b in by_version.items()
    }
    overall = _summarize(overall_predicted, overall_graded)

    coin_flip_scored = [f for f in overall_graded if f.get("coin_flip_correct") is not None]
    elo_scored = [f for f in overall_graded if f.get("elo_dummy_correct") is not None]
    baselines = {
        "coin_flip": {
            "n_graded": len(overall_graded),
            "accuracy": (
                sum(1 for f in coin_flip_scored if f["coin_flip_correct"])
                / len(coin_flip_scored)
                if coin_flip_scored else None
            ),
            "log_loss": COIN_FLIP_LOG_LOSS if overall_graded else None,
            "brier": COIN_FLIP_BRIER if overall_graded else None,
        },
        "higher_elo_dummy": {
            "n_graded": len(elo_scored),
            "accuracy": (
                sum(1 for f in elo_scored if f["elo_dummy_correct"]) / len(elo_scored)
                if elo_scored else None
            ),
        },
    }

    return {
        "generated_at_utc": generated_at_utc,
        "overall": overall,
        "model_versions": model_versions,
        "baselines": baselines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--fights-parquet", type=Path, default=FIGHTS_PARQUET)
    args = parser.parse_args()

    predictions_dir: Path = args.predictions_dir
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if args.fights_parquet.exists():
        fights = pd.read_parquet(args.fights_parquet)
        fights["date"] = pd.to_datetime(fights["date"])
    else:
        fights = pd.DataFrame()

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    files = sorted(predictions_dir.glob("UFC_*.json"))
    event_records = []
    n_graded_now = 0
    n_pending = 0

    for path in files:
        record = json.loads(path.read_text())
        event_records.append(record)
        event_date = pd.Timestamp(record["event_date"])
        if event_date > today:
            continue  # hasn't happened yet
        changed = False
        for fight in record["fights"]:
            if fight.get("skipped") or "actual_winner" in fight:
                continue
            result = find_result(fight, fights, record["event_date"])
            if result is None:
                n_pending += 1
                continue
            fight.update(grade_fight(fight, result))
            changed = True
            n_graded_now += 1
        if changed:
            path.write_text(json.dumps(record, indent=2) + "\n")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    track_record = aggregate_track_record(event_records, generated_at)
    (predictions_dir / "track_record.json").write_text(
        json.dumps(track_record, indent=2) + "\n"
    )

    print(f"graded {n_graded_now} fight(s) this run; {n_pending} still pending (data lag)")
    print(json.dumps(track_record["overall"], indent=2))


if __name__ == "__main__":
    main()
