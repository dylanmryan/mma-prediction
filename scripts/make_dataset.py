"""Build processed parquet tables from raw CSVs. Reproducible end to end."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mma.dataset import build_fight_stats, build_fighters, build_fights

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"integrity check failed: {message}")


def main() -> None:
    raw_fights = pd.read_csv(RAW / "UFC.csv")
    raw_fighters = pd.read_csv(RAW / "fighter_details.csv")

    fighters = build_fighters(raw_fighters)
    fights = build_fights(raw_fights)
    stats = build_fight_stats(raw_fights)

    check(fights["date"].notna().all(), "unparseable fight dates")
    check(len(stats) == 2 * len(fights), "stats rows != 2x fights")
    known = set(fighters["fighter_id"])
    in_fights = set(fights["fighter_a_id"]) | set(fights["fighter_b_id"])
    orphans = in_fights - known
    check(
        len(orphans) < 0.02 * len(in_fights),
        f"{len(orphans)} fight participants missing from fighters table",
    )
    method_rate = fights["method"].notna().mean()
    check(method_rate > 0.95, f"method mapped for only {method_rate:.1%} of fights")
    weight_rate = fights["weight_class"].notna().mean()
    check(weight_rate > 0.95, f"weight class for only {weight_rate:.1%} of fights")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    fighters.to_parquet(PROCESSED / "fighters.parquet", index=False)
    fights.to_parquet(PROCESSED / "fights.parquet", index=False)
    stats.to_parquet(PROCESSED / "fight_stats.parquet", index=False)

    print(f"fighters: {len(fighters)} rows")
    print(
        f"fights:   {len(fights)} rows, "
        f"{fights['date'].min():%Y-%m-%d} .. {fights['date'].max():%Y-%m-%d}"
    )
    print(f"stats:    {len(stats)} rows")
    print("\nwinner distribution:")
    print(fights["winner"].value_counts().to_string())
    print("\nmethod distribution:")
    print(fights["method"].value_counts(dropna=False).to_string())
    if orphans:
        print(f"\nwarning: {len(orphans)} orphan fighter ids (sample): {sorted(orphans)[:5]}")


if __name__ == "__main__":
    main()
