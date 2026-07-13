"""Download the Kaggle UFC dataset into data/raw/ and report its schema.

Dataset: https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025
(a ufcstats.com scrape, 1994 - mid-2025).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "neelagiriaditya/ufc-datasets-1994-2025"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(raw_dir: Path) -> None:
    """Download the Kaggle dataset snapshot and copy its CSVs into raw_dir."""
    cache_path = Path(kagglehub.dataset_download(DATASET))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for src in cache_path.rglob("*.csv"):
        dest = raw_dir / src.name
        shutil.copy2(src, dest)
        print(f"copied {src.name}")


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
