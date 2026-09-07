"""Compare a previous processed fights table against the current one.

Used once when switching Kaggle layouts (2026-09), kept so any future
source change can be audited the same way. Reports overlap, label
agreement on overlapping fights, and what the new source adds or drops.

Usage:
    python scripts/reconcile_sources.py --old /path/old_fights.parquet [--new data/processed/fights.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW_DEFAULT = ROOT / "data" / "processed" / "fights.parquet"
_COMPARE = ("winner", "method", "finish_round", "scheduled_rounds", "weight_class",
            "fighter_a_id", "fighter_b_id", "date")


def reconcile(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Pure comparison; returns a dict of counts and agreement rates."""
    old_ids, new_ids = set(old["fight_id"]), set(new["fight_id"])
    overlap = old.merge(new, on="fight_id", suffixes=("_old", "_new"))
    agreement = {}
    for column in _COMPARE:
        a, b = overlap[f"{column}_old"], overlap[f"{column}_new"]
        # With nullable dtypes (string, Int64) a one-sided null makes (a == b)
        # itself <NA>, which .mean() silently skips -- that would let a source
        # that nulls out a column on many overlapping fights still read 1.0.
        same = (a.isna() & b.isna()) | (a == b).fillna(False)
        agreement[column] = float(same.mean()) if len(overlap) else None
    new_only = new[~new["fight_id"].isin(old_ids)]
    return {
        "n_old": len(old),
        "n_new": len(new),
        "n_overlap": len(old_ids & new_ids),
        "n_dropped_by_new": len(old_ids - new_ids),
        "n_added_by_new": len(new_only),
        "added_by_year": new_only.groupby(new_only["date"].dt.year).size().to_dict(),
        "agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, default=NEW_DEFAULT)
    args = parser.parse_args()
    report = reconcile(pd.read_parquet(args.old), pd.read_parquet(args.new))
    print(f"old fights: {report['n_old']}   new fights: {report['n_new']}")
    print(f"overlap: {report['n_overlap']}   dropped by new: {report['n_dropped_by_new']}   added by new: {report['n_added_by_new']}")
    print("agreement on overlap:")
    for column, rate in report["agreement"].items():
        print(f"  {column:<18} {rate:.4f}" if rate is not None else f"  {column:<18} n/a")
    print("added by year:", {int(k): int(v) for k, v in report["added_by_year"].items()})


if __name__ == "__main__":
    main()
