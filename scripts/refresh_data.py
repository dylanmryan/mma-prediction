"""Refresh the raw Kaggle dataset and report whether a pipeline rebuild is needed.

Downloads the latest snapshot of the Kaggle UFC dataset (the same maintained
mirror `download_data.py` bootstraps from), then compares it against the
already-processed fights table to decide whether the fuller rebuild chain
(make_dataset -> build_ratings -> build_features -> train_xgb -> train_torch
-> build_display_priors) is warranted. Always exits 0; prints a
machine-readable `REFRESH_NEEDED=true|false` line for CI to parse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import RAW_DIR, download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FIGHTS = ROOT / "data" / "processed" / "fights.parquet"


def refresh_needed(
    raw_fights: pd.DataFrame, processed_fights: pd.DataFrame
) -> tuple[bool, str]:
    """Pure comparison of the freshly downloaded raw fights against the
    already-processed fights table.

    Returns (True, reason) if the raw data has a strictly newer max date, or
    more rows, than the processed data; otherwise (False, reason). Never
    raises: unparseable raw dates are treated as "no evidence of newer data"
    rather than triggering a refresh.
    """
    raw_dates = pd.to_datetime(raw_fights.get("date"), format="mixed", errors="coerce")
    processed_dates = pd.to_datetime(
        processed_fights.get("date"), format="mixed", errors="coerce"
    )

    raw_rows = len(raw_fights)
    processed_rows = len(processed_fights)
    raw_max_date = raw_dates.max() if len(raw_dates) else pd.NaT
    processed_max_date = processed_dates.max() if len(processed_dates) else pd.NaT

    if pd.isna(raw_max_date):
        return False, (
            f"raw dates unparseable ({raw_rows} rows); skipping refresh for safety"
        )

    if pd.notna(processed_max_date) and raw_max_date > processed_max_date:
        return True, (
            f"raw max date {raw_max_date.date()} is newer than processed "
            f"max date {processed_max_date.date()}"
        )

    if raw_rows > processed_rows:
        return True, f"raw has more rows ({raw_rows}) than processed ({processed_rows})"

    processed_max_str = (
        processed_max_date.date() if pd.notna(processed_max_date) else "unknown"
    )
    return False, (
        f"raw data unchanged: max date {raw_max_date.date()} / {raw_rows} rows "
        f"vs processed max date {processed_max_str} / {processed_rows} rows"
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

    raw_fights = pd.read_csv(RAW_DIR / "UFC.csv")
    if PROCESSED_FIGHTS.exists():
        processed_fights = pd.read_parquet(PROCESSED_FIGHTS)
    else:
        processed_fights = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})

    if args.force:
        needed, reason = True, "--force flag set"
    else:
        needed, reason = refresh_needed(raw_fights, processed_fights)

    print("\n=== REFRESH CHECK ===")
    print(f"raw fights:       {len(raw_fights)} rows")
    print(f"processed fights: {len(processed_fights)} rows")
    print(reason)
    print(f"REFRESH_NEEDED={'true' if needed else 'false'}")


if __name__ == "__main__":
    main()
