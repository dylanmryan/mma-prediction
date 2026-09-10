"""Refresh the raw Kaggle dataset and report whether a pipeline rebuild is needed.

TWO SOURCES ASK FOR A REBUILD, not one. Since the daily ufcstats scrape
became a standing stage of `make_dataset.py`, the processed tables run AHEAD
of the Kaggle mirror by design -- so comparing the freshly downloaded
snapshot against the whole processed table would report "nothing new" for
every Kaggle refresh from now on, and the tables would freeze at whatever the
scrape last supplied. Two changes follow. The Kaggle comparison is made
against the PRIMARY-sourced rows only, which is what `provenance.parquet` is
for; and the scrape's own event list is probed for events beyond the
processed tables, so a week where only the scrape has moved still rebuilds.
The probe is fail-soft: if it cannot be reached the Kaggle decision stands
alone.

Downloads the latest snapshot of the Kaggle UFC dataset (the same maintained
mirror `download_data.py` bootstraps from), then compares it against the
already-processed fights table to decide whether the fuller rebuild chain
(make_dataset -> build_ratings -> build_features -> train_xgb -> train_torch
-> train_hazard -> check_display_calibration) is warranted. Exits 0 whenever the snapshot is
readable, printing a machine-readable `REFRESH_NEEDED=true|false` line for
CI to parse; raises (non-zero exit) if the downloaded snapshot is missing a
file the pipeline requires, so the weekly Action goes red instead of
silently skipping the refresh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import RAW_DIR, download  # noqa: E402
from refresh_secondary import FILES, SOURCE_URL, fetch_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FIGHTS = ROOT / "data" / "processed" / "fights.parquet"
PROCESSED_PROVENANCE = ROOT / "data" / "processed" / "provenance.parquet"


def primary_fights(processed_fights: pd.DataFrame, provenance: pd.DataFrame | None):
    """The processed rows the Kaggle mirror itself put there.

    With no provenance sidecar -- a checkout that has never built one -- every
    row counts as primary, which is the comparison that predates it.
    """
    if provenance is None or not len(provenance) or "fight_id" not in processed_fights:
        return processed_fights
    fights_rows = provenance[provenance["table"] == "fights"]
    from_primary = set(fights_rows.loc[fights_rows["source"] == "primary", "row_id"])
    return processed_fights[processed_fights["fight_id"].isin(from_primary)]


def secondary_max_event_date(base: str | None = None) -> pd.Timestamp | None:
    """Newest event the daily scrape has published, or None if unreachable.

    One small CSV (787 rows today), and it lists completed events only, so
    its newest date is a cheap and honest probe for "has anything happened
    that our tables do not have". Every failure returns None: this decides
    whether to rebuild, and being unable to ask must never be an error.
    """
    try:
        events = fetch_csv(FILES["events"], base=base)
        dates = pd.to_datetime(events["DATE"], format="mixed", errors="coerce")
        newest = dates.max()
    except Exception as error:  # noqa: BLE001 -- a probe that cannot fail
        print(f"note: could not probe {SOURCE_URL} for freshness ({error})")
        return None
    return None if pd.isna(newest) else newest


def refresh_needed(
    raw_fights: pd.DataFrame,
    processed_fights: pd.DataFrame,
    provenance: pd.DataFrame | None = None,
    secondary_max_date: pd.Timestamp | None = None,
) -> tuple[bool, str]:
    """Pure comparison of both sources against the already-processed table.

    Returns (True, reason) if the Kaggle snapshot has a strictly newer max
    date, or more rows, than the PRIMARY-sourced processed rows, or if the
    daily scrape has an event later than anything in the processed table.
    Otherwise (False, reason). Never raises: unparseable raw dates and an
    absent `secondary_max_date` are both treated as "no evidence of newer
    data" rather than as a reason to rebuild.
    """
    primary = primary_fights(processed_fights, provenance)
    raw_dates = pd.to_datetime(raw_fights.get("date"), format="mixed", errors="coerce")
    primary_dates = pd.to_datetime(primary.get("date"), format="mixed", errors="coerce")
    all_dates = pd.to_datetime(
        processed_fights.get("date"), format="mixed", errors="coerce"
    )

    raw_rows = len(raw_fights)
    primary_rows = len(primary)
    raw_max_date = raw_dates.max() if len(raw_dates) else pd.NaT
    primary_max_date = primary_dates.max() if len(primary_dates) else pd.NaT
    processed_max_date = all_dates.max() if len(all_dates) else pd.NaT

    scrape_ahead = (
        secondary_max_date is not None
        and pd.notna(secondary_max_date)
        and (pd.isna(processed_max_date) or secondary_max_date > processed_max_date)
    )

    if pd.isna(raw_max_date):
        if scrape_ahead:
            return True, (
                f"raw dates unparseable ({raw_rows} rows), but the daily scrape has "
                f"an event dated {pd.Timestamp(secondary_max_date).date()}"
            )
        return False, (
            f"raw dates unparseable ({raw_rows} rows); skipping refresh for safety"
        )

    if pd.notna(primary_max_date) and raw_max_date > primary_max_date:
        return True, (
            f"raw max date {raw_max_date.date()} is newer than the primary-sourced "
            f"processed max date {primary_max_date.date()}"
        )

    # Secondary tie-break only: raw and primary-sourced processed rows are 1:1
    # today, but if make_dataset.py ever filters rows this comparison would stay
    # permanently True — the date check above is the robust primary signal.
    if raw_rows > primary_rows:
        return True, (
            f"raw has more rows ({raw_rows}) than the primary-sourced processed "
            f"rows ({primary_rows})"
        )

    if scrape_ahead:
        return True, (
            f"the daily scrape has an event dated "
            f"{pd.Timestamp(secondary_max_date).date()}, later than the processed "
            f"max date {processed_max_date.date() if pd.notna(processed_max_date) else 'unknown'}"
        )

    processed_max_str = (
        processed_max_date.date() if pd.notna(processed_max_date) else "unknown"
    )
    return False, (
        f"neither source has anything new: raw max date {raw_max_date.date()} / "
        f"{raw_rows} rows against {primary_rows} primary-sourced processed rows, "
        f"and the processed tables already reach {processed_max_str}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="skip the comparison and report REFRESH_NEEDED=true unconditionally",
    )
    args = parser.parse_args()

    download(RAW_DIR)

    raw_fights = pd.read_csv(
        RAW_DIR / "master.csv", usecols=["fight_id", "event_date"]
    ).rename(columns={"event_date": "date"})
    if PROCESSED_FIGHTS.exists():
        processed_fights = pd.read_parquet(PROCESSED_FIGHTS)
    else:
        processed_fights = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})
    provenance = (
        pd.read_parquet(PROCESSED_PROVENANCE) if PROCESSED_PROVENANCE.exists() else None
    )

    if args.force:
        needed, reason = True, "--force flag set"
        scrape_date = None
    else:
        scrape_date = secondary_max_event_date()
        needed, reason = refresh_needed(
            raw_fights, processed_fights, provenance, scrape_date
        )

    print("\n=== REFRESH CHECK ===")
    print(f"raw fights:       {len(raw_fights)} rows")
    print(f"processed fights: {len(processed_fights)} rows "
          f"({len(primary_fights(processed_fights, provenance))} primary-sourced)")
    print(f"daily scrape:     newest event "
          f"{scrape_date.date() if scrape_date is not None else 'unknown'}")
    print(reason)
    print(f"REFRESH_NEEDED={'true' if needed else 'false'}")


if __name__ == "__main__":
    main()
