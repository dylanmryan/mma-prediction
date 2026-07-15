"""Predict every fight on UFC events scheduled in the next N days.

Fetches "List of UFC events" from Wikipedia, keeps events within the
horizon, fetches each event's own page, parses its fight card, matches
fighter names against fighters.parquet, and predicts each matched fight
with the committed ensemble. Writes one JSON record per event under
predictions/ (idempotent -- see mma.prospective.write_event_prediction).

Only `select_upcoming_events` is a pure function (unit-tested); the rest
of this module is network + artifact I/O, exercised by the live run
rather than pytest.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PREDICTIONS_DIR = ROOT / "predictions"
PROCESSED = ROOT / "data" / "processed"
DEFAULT_HORIZON_DAYS = 30
SCHEDULED_EVENTS_PAGE = "List_of_UFC_events"


def select_upcoming_events(events: list[dict], today: date, horizon_days: int) -> list[dict]:
    """Pure filter: events with today <= date <= today + horizon_days,
    soonest first."""
    cutoff = today + timedelta(days=horizon_days)
    selected = [
        event for event in events
        if today <= date.fromisoformat(event["date"]) <= cutoff
    ]
    return sorted(selected, key=lambda e: e["date"])


def _git_short_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> None:
    from mma.inference import Ensemble
    from mma.prospective import (
        build_name_index, predict_event, write_event_prediction,
    )
    from mma.snapshots import build_snapshots
    from mma.wiki_cards import fetch_page_html, parse_fight_card, parse_scheduled_events

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()

    fights_df = pd.read_parquet(PROCESSED / "fights.parquet")
    stats_df = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters_df = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings_df = pd.read_parquet(PROCESSED / "ratings.parquet")
    snapshots = build_snapshots(fights_df, stats_df, ratings_df)
    fighters_indexed = fighters_df.set_index("fighter_id")
    name_index = build_name_index(fighters_df)
    ensemble = Ensemble.load()
    model_version = _git_short_sha()

    print(f"Fetching scheduled events list (model_version={model_version})...")
    events_html = fetch_page_html(SCHEDULED_EVENTS_PAGE)
    all_events = parse_scheduled_events(events_html)
    today = datetime.now(timezone.utc).date()
    upcoming = select_upcoming_events(all_events, today, args.horizon_days)
    print(f"{len(all_events)} scheduled events found; {len(upcoming)} within "
          f"the next {args.horizon_days} days (as of {today.isoformat()}).")

    total_matched, total_skipped = 0, 0
    for event in upcoming:
        print(f"\n== {event['event_name']} ({event['date']}) ==")
        card_html = fetch_page_html(event["wiki_title"])
        wiki_fights = parse_fight_card(card_html)
        print(f"  {len(wiki_fights)} fight(s) on the card")

        predictions = predict_event(
            event, wiki_fights, name_index, snapshots, fighters_indexed, ensemble
        )
        predicted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _, record, n_new = write_event_prediction(
            args.predictions_dir, event["event_name"], event["date"],
            event["page_url"], model_version, predicted_at, predictions,
        )

        matched = sum(1 for f in predictions if not f.get("skipped"))
        skipped = [f for f in predictions if f.get("skipped")]
        total_matched += matched
        total_skipped += len(skipped)
        print(f"  matched {matched}/{len(predictions)}; {n_new} new fight(s) written")
        for f in skipped:
            print(f"  SKIPPED {f['fighter_a_name']} vs {f['fighter_b_name']}: {f['reason']}")
        for f in predictions:
            if not f.get("skipped"):
                print(
                    f"  {f['fighter_a_name']} {f['p_a_wins']:.1%} vs "
                    f"{f['fighter_b_name']} {1 - f['p_a_wins']:.1%} "
                    f"({f['weight_class']})"
                )

    total = total_matched + total_skipped
    rate = (total_matched / total) if total else 0.0
    print(f"\n=== SUMMARY: {len(upcoming)} event(s), {total} fight(s), "
          f"{total_matched} matched ({rate:.0%}), {total_skipped} skipped ===")


if __name__ == "__main__":
    main()
