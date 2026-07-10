"""Tune Elo params on pre-2020 fights, build the full ratings table.

Validation metrics are reported for 2021-2023. 2024+ is never evaluated
here (held-out test years for the final model comparison).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from mma.elo import EloParams, expected_score, run_elo
from mma.evaluate import accuracy, brier_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TUNE_CUTOFF = "2020-01-01"
VAL_START, VAL_END = "2021-01-01", "2023-12-31"

GRID = {
    "k_early": [40.0, 48.0, 56.0, 64.0],
    "k_late": [24.0, 32.0, 40.0, 48.0],
    "finish_bonus": [1.2, 1.4, 1.6, 1.8],
}


def _predictions(ratings: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """One row per rated fight with decisive winner: p(A wins) from pre-fight Elo."""
    a = ratings[ratings["corner"] == "a"][["fight_id", "pre_overall"]]
    b = ratings[ratings["corner"] == "b"][["fight_id", "pre_overall"]]
    merged = (
        fights[fights["winner"].isin(["a", "b"])][["fight_id", "date", "winner"]]
        .merge(a, on="fight_id")
        .merge(b, on="fight_id", suffixes=("_a", "_b"))
    )
    merged["p_a"] = [
        expected_score(ra, rb)
        for ra, rb in zip(merged["pre_overall_a"], merged["pre_overall_b"])
    ]
    merged["y"] = (merged["winner"] == "a").astype(float)
    return merged


def main() -> None:
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")

    tune_fights = fights[fights["date"] < TUNE_CUTOFF]
    best_params, best_loss = None, float("inf")
    for k_early, k_late, bonus in itertools.product(
        GRID["k_early"], GRID["k_late"], GRID["finish_bonus"]
    ):
        params = EloParams(k_early=k_early, k_late=k_late, finish_bonus=bonus)
        preds = _predictions(run_elo(tune_fights, stats, params), tune_fights)
        loss = log_loss(preds["y"], preds["p_a"])
        print(f"k_early={k_early:>4} k_late={k_late:>4} bonus={bonus} "
              f"-> pre-2020 log-loss {loss:.4f}")
        if loss < best_loss:
            best_params, best_loss = params, loss

    print(f"\nbest: {best_params} (log-loss {best_loss:.4f})")

    ratings = run_elo(fights, stats, best_params)
    ratings.to_parquet(PROCESSED / "ratings.parquet", index=False)
    (PROCESSED / "elo_params.json").write_text(
        json.dumps(
            {
                "k_early": best_params.k_early,
                "k_late": best_params.k_late,
                "early_fights": best_params.early_fights,
                "finish_bonus": best_params.finish_bonus,
                "tuned_on": f"fights before {TUNE_CUTOFF}",
                "pre2020_log_loss": round(best_loss, 4),
            },
            indent=2,
        )
    )

    preds = _predictions(ratings, fights)
    val = preds[(preds["date"] >= VAL_START) & (preds["date"] <= VAL_END)]
    higher_elo_acc = accuracy(val["y"], (val["p_a"] > 0.5).astype(float))
    print(f"\n=== Elo baseline, validation 2021-2023 ({len(val)} fights) ===")
    print(f"accuracy:  {accuracy(val['y'], val['p_a']):.4f}")
    print(f"log-loss:  {log_loss(val['y'], val['p_a']):.4f}")
    print(f"brier:     {brier_score(val['y'], val['p_a']):.4f}")
    print(f"(higher-Elo-wins dummy accuracy: {higher_elo_acc:.4f})")

    peaks = (
        ratings.groupby("fighter_id")["post_overall"].max().nlargest(10).round(1)
    )
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    names = fighters.set_index("fighter_id")["name"]
    print("\nall-time peak Elo top 10:")
    for fighter_id, rating in peaks.items():
        print(f"  {names.get(fighter_id, fighter_id):<28} {rating}")


if __name__ == "__main__":
    main()
