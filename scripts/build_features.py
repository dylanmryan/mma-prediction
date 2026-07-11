"""Build the model-ready feature table from processed parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mma.features import build_features
from mma.history import build_history

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")

    history = build_history(fights, stats, ratings)
    features = build_features(fights, fighters, ratings, history)
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    print(f"{len(features)} rows, {features.shape[1]} columns")
    print("y_winner balance:", features["y_winner"].mean().round(4))
    print("swapped share:", features["swapped"].mean().round(4))
    per_year = features.groupby(features["date"].dt.year).size()
    print("rows/year (last 6):")
    print(per_year.tail(6).to_string())


if __name__ == "__main__":
    main()
