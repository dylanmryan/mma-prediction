"""Build processed parquet tables from raw CSVs. Reproducible end to end."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mma.dataset import (
    build_bonuses, build_fight_stats, build_fighters, build_fights, build_round_stats,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"integrity check failed: {message}")


def main() -> None:
    raw_master = pd.read_csv(RAW / "master.csv")
    raw_fighters = pd.read_csv(RAW / "fighter.csv")
    raw_rounds = pd.read_csv(RAW / "round.csv")
    raw_bonuses = pd.read_csv(RAW / "fighter_bonus.csv")

    fighters = build_fighters(raw_fighters)
    fights = build_fights(raw_master)
    stats = build_fight_stats(raw_master)
    rounds = build_round_stats(raw_rounds)
    bonuses = build_bonuses(raw_bonuses)

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
    round_fights = set(rounds["fight_id"])
    check(round_fights <= set(fights["fight_id"]), "round_stats has unknown fight ids")
    per_fight = rounds.groupby("fight_id").size()
    check((per_fight % 2 == 0).all(), "round_stats must have both corners per round")
    round_cover = len(round_fights) / len(fights)
    check(round_cover > 0.90, f"round stats cover only {round_cover:.1%} of fights")
    check(set(bonuses["fight_id"]) <= set(fights["fight_id"]), "bonuses has unknown fight ids")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    fighters.to_parquet(PROCESSED / "fighters.parquet", index=False)
    fights.to_parquet(PROCESSED / "fights.parquet", index=False)
    stats.to_parquet(PROCESSED / "fight_stats.parquet", index=False)
    rounds.to_parquet(PROCESSED / "round_stats.parquet", index=False)
    bonuses.to_parquet(PROCESSED / "bonuses.parquet", index=False)

    print(f"fighters:    {len(fighters)} rows")
    print(
        f"fights:      {len(fights)} rows, "
        f"{fights['date'].min():%Y-%m-%d} .. {fights['date'].max():%Y-%m-%d}"
    )
    print(f"stats:       {len(stats)} rows")
    print(f"round_stats: {len(rounds)} rows covering {len(round_fights)} fights")
    print(f"bonuses:     {len(bonuses)} rows")
    print("\nwinner distribution:")
    print(fights["winner"].value_counts().to_string())
    print("\nmethod distribution:")
    print(fights["method"].value_counts(dropna=False).to_string())
    if orphans:
        print(f"\nwarning: {len(orphans)} orphan fighter ids (sample): {sorted(orphans)[:5]}")


if __name__ == "__main__":
    main()
