"""Download the Kaggle UFC dataset into data/raw/ and report its schema.

Dataset: https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025
(a ufcstats.com scrape; rebuilt 2026-08-11 with a new layout). Files used
downstream: master.csv (one row per fight with fight-total stats),
fighter.csv, round.csv (per-round stats), fighter_bonus.csv. The other
files in the snapshot (fight.csv, event.csv, scrape_error.csv) are copied
for reference but not read.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "neelagiriaditya/ufc-datasets-1994-2025"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REQUIRED_FILES = ("master.csv", "fighter.csv", "round.csv", "fighter_bonus.csv")


def verify_raw_files(raw_dir: Path) -> None:
    """Fail loudly if the snapshot lacks any file the pipeline reads."""
    missing = [name for name in REQUIRED_FILES if not (Path(raw_dir) / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Kaggle snapshot layout changed: missing {missing} under {raw_dir}"
        )


def download(raw_dir: Path) -> None:
    """Download the Kaggle dataset snapshot and copy its CSVs into raw_dir."""
    cache_path = Path(kagglehub.dataset_download(DATASET))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for src in cache_path.rglob("*.csv"):
        dest = raw_dir / src.name
        shutil.copy2(src, dest)
        print(f"copied {src.name}")
    verify_raw_files(raw_dir)


def main() -> None:
    download(RAW_DIR)

    print("\n=== SCHEMA REPORT ===")
    for csv in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(csv, nrows=5, sep=None, engine="python")
        print(f"\n{csv.name}  ({len(df.columns)} cols)")
        print("  columns:", list(df.columns))
        print(df.head(2).to_string(max_colwidth=25))


if __name__ == "__main__":
    main()
