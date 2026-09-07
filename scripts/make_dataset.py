"""Build processed parquet tables from raw CSVs. Reproducible end to end.

Inputs (data/raw/): master.csv (one row per fight incl. fight-total stats),
fighter.csv (one row per fighter), round.csv (per-round stats),
fighter_bonus.csv (post-fight bonus awards).
Outputs (data/processed/): fighters.parquet, fights.parquet,
fight_stats.parquet, round_stats.parquet, bonuses.parquet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mma.dataset import (
    build_bonuses, build_fight_stats, build_fighters, build_fights, build_round_stats,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_sources import reconcile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"integrity check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-regression",
        action="store_true",
        help="print the reconciliation report but do not fail on dropped fights "
        "or winner relabels (use after a human review of a legitimate upstream change)",
    )
    args = parser.parse_args()

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
    round_cover = len(round_fights) / len(fights)
    check(round_cover > 0.90, f"round stats cover only {round_cover:.1%} of fights")
    check(set(bonuses["fight_id"]) <= set(fights["fight_id"]), "bonuses has unknown fight ids")

    corner_check = rounds[["fight_id", "corner", "fighter_id"]].merge(
        fights[["fight_id", "fighter_a_id", "fighter_b_id"]], on="fight_id"
    )
    corner_a = corner_check[corner_check["corner"] == "a"]
    corner_b = corner_check[corner_check["corner"] == "b"]
    check(
        (corner_a["fighter_id"] == corner_a["fighter_a_id"]).all(),
        "round_stats corner 'a' fighter id disagrees with fights.fighter_a_id",
    )
    check(
        (corner_b["fighter_id"] == corner_b["fighter_b_id"]).all(),
        "round_stats corner 'b' fighter id disagrees with fights.fighter_b_id",
    )

    round_span = rounds.groupby("fight_id")["round_no"].agg(["min", "max", "nunique"])
    check((round_span["min"] == 1).all(), "some fight's rounds do not start at round 1")
    check(
        (round_span["nunique"] == round_span["max"]).all(),
        "some fight's rounds are not contiguous from 1",
    )

    modern_cutoff = pd.Timestamp("2014-01-01")
    modern_fights = set(fights.loc[fights["date"] >= modern_cutoff, "fight_id"])
    modern_cover = len(modern_fights & round_fights) / max(len(modern_fights), 1)
    # Rate, not zero-tolerance: upstream's own scrape_error.csv shows single
    # fights can fail to parse; one such fight must not stall the weekly refresh.
    check(
        modern_cover >= 0.995,
        f"round stats cover only {modern_cover:.2%} of fights dated on/after 2014-01-01",
    )

    today = pd.Timestamp.today().normalize()
    check((fights["date"] <= today).all(), "fight dated in the future")
    check(
        (fights["fighter_a_id"] != fights["fighter_b_id"]).all(),
        "a fight has the same fighter in both corners",
    )
    check(
        set(raw_master["result_status"].dropna()) <= {"win", "draw", "no_contest"},
        "raw master.csv result_status has an unrecognized value",
    )

    # Regression guard: a partial/corrupted upstream re-scrape must not get
    # auto-committed by the weekly refresh Action. Compare against whatever
    # fights.parquet is already committed (if any) before overwriting it. A
    # legitimate relabelling that pushes agreement below the threshold needs
    # a human review, then an explicit --allow-regression run, rather than
    # loosening this bar.
    old_fights_path = PROCESSED / "fights.parquet"
    if old_fights_path.exists():
        previous = pd.read_parquet(old_fights_path)
        report = reconcile(previous, fights)
        print("\nreconcile vs committed fights.parquet:")
        print(f"  n_dropped_by_new: {report['n_dropped_by_new']}")
        print(f"  winner agreement: {report['agreement']['winner']:.4f}")
        if args.allow_regression:
            print("  --allow-regression set: not enforcing the regression guard")
        else:
            check(
                report["n_dropped_by_new"] == 0,
                f"new fights table drops {report['n_dropped_by_new']} fights present in "
                "the committed table (re-run with --allow-regression after review)",
            )
            check(
                report["agreement"]["winner"] >= 0.999,
                f"winner agreement with committed table only "
                f"{report['agreement']['winner']:.4f} (re-run with --allow-regression after review)",
            )

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
