"""Build the "does the model beat the market?" evaluation benchmark.

EVALUATION-ONLY: betting odds are a COMPARATOR here, never a model feature.
This script downloads a Kaggle odds-history dataset via kagglehub, aligns it
to our fighter_a/fighter_b feature convention using the shared 16-hex
ufcstats fight id, and compares the committed neural ensemble's pre-fight
win probabilities against the devigged market-implied probabilities on the
same historical fights -- reusing exactly the loading/prediction machinery
`scripts/final_test_eval.py` uses (features.parquet + the committed torch
checkpoints), never re-predicting from live snapshots.

Run once: .venv/bin/python scripts/build_odds_benchmark.py
Writes models/market_benchmark.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mma.evaluate import accuracy, brier_score, log_loss
from mma.inference import Ensemble
from mma.odds import consensus_odds, decimal_to_implied, devig_pair, extract_fight_id
from mma.prospective import build_name_index, match_fighter_id

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

ODDS_DATASET = "jerzyszocik/ufc-betting-odds-daily-dataset"
HEADLINE_START = "2021-01-01"  # model's validation+test era: NOT seen in training
THRESHOLDS = (0.00, 0.05, 0.10)

# Famous fights used to sanity-check corner alignment: (name_1, name_2,
# expected favorite's name, approximate event date). Chosen so the favorite
# sometimes lost (Rousey, Silva) -- proving this isn't just "favorite ==
# winner" -- and so the favorite appears in both odds-column positions
# across the two Silva/Weidman fights (the rematch's odds file row has
# fighter_1=Weidman, fighter_2=Silva -- the reverse order from the first
# fight), proving alignment isn't a naive "fighter_1 is always the
# favorite" bug. The date disambiguates the two Silva/Weidman meetings.
FAMOUS_FIGHTS = [
    ("Ronda Rousey", "Holly Holm", "Ronda Rousey", "2015-11-14"),
    ("Anderson Silva", "Chris Weidman", "Anderson Silva", "2013-07-06"),
    ("Chris Weidman", "Anderson Silva", "Anderson Silva", "2013-12-28"),  # rematch, reordered
    ("Khabib Nurmagomedov", "Conor McGregor", "Khabib Nurmagomedov", "2018-10-06"),
    ("Jon Jones", "Daniel Cormier", "Jon Jones", "2015-01-03"),
]


def load_odds_raw() -> pd.DataFrame:
    """kagglehub-download the odds CSV; one row per (fight, bookmaker)."""
    import kagglehub

    cache_dir = Path(kagglehub.dataset_download(ODDS_DATASET))
    csvs = list(cache_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSV found in kagglehub cache {cache_dir}")
    df = pd.read_csv(csvs[0], low_memory=False)
    df["fight_id"] = df["fight_url"].map(extract_fight_id)
    df["f1_id"] = df["fighter_1_url"].map(extract_fight_id)
    df["f2_id"] = df["fighter_2_url"].map(extract_fight_id)
    return df


def _orient_by_id(group: pd.DataFrame, fighter_a_id: str, fighter_b_id: str):
    """Return 'same' / 'swapped' / None using fighter_1_url/fighter_2_url ids."""
    id_rows = group.dropna(subset=["f1_id", "f2_id"])
    if id_rows.empty:
        return None
    f1_id, f2_id = id_rows.iloc[0][["f1_id", "f2_id"]]
    if f1_id == fighter_a_id and f2_id == fighter_b_id:
        return "same"
    if f1_id == fighter_b_id and f2_id == fighter_a_id:
        return "swapped"
    return None


def _orient_by_name(group: pd.DataFrame, fighter_a_id: str, fighter_b_id: str, name_index):
    """Fallback: resolve fighter_1/fighter_2 names via the accent-folding matcher."""
    row = group.iloc[0]
    name_1, name_2 = row.get("fighter_1"), row.get("fighter_2")
    if pd.isna(name_1) or pd.isna(name_2):
        return None, "missing fighter names"
    id_1, _, reason_1 = match_fighter_id(str(name_1), name_index)
    id_2, _, reason_2 = match_fighter_id(str(name_2), name_index)
    if id_1 is None or id_2 is None:
        reasons = [r for r in (reason_1, reason_2) if r]
        return None, "; ".join(reasons) or "name match failed"
    if id_1 == fighter_a_id and id_2 == fighter_b_id:
        return "same", None
    if id_1 == fighter_b_id and id_2 == fighter_a_id:
        return "swapped", None
    return None, f"resolved names {name_1!r}/{name_2!r} match neither corner of this fight"


def align_odds_to_fights(odds_raw: pd.DataFrame, fights: pd.DataFrame, fighters: pd.DataFrame):
    """Per-fight devigged market implied probabilities in fights.parquet's a/b convention.

    Returns (aligned_df, stats) where aligned_df has columns:
    fight_id, decimal_fights_a, decimal_fights_b, market_implied_fights_a,
    market_implied_fights_b, align_method ('id' | 'name'), n_books.
    `stats` is a dict of counters (n_id, n_name, n_skipped, skip_reasons).
    """
    fights_idx = fights.set_index("fight_id")
    fight_ids_known = set(fights_idx.index)
    name_index = build_name_index(fighters)

    records = []
    stats = {"n_id": 0, "n_name": 0, "n_skipped": 0, "skip_reasons": {}}

    for fight_id, group in odds_raw.groupby("fight_id"):
        if fight_id is None or fight_id not in fight_ids_known:
            continue
        fights_row = fights_idx.loc[fight_id]
        fighter_a_id, fighter_b_id = fights_row["fighter_a_id"], fights_row["fighter_b_id"]

        orientation = _orient_by_id(group, fighter_a_id, fighter_b_id)
        method = "id" if orientation is not None else None
        reason = None
        if orientation is None:
            orientation, reason = _orient_by_name(group, fighter_a_id, fighter_b_id, name_index)
            method = "name" if orientation is not None else None

        if orientation is None:
            stats["n_skipped"] += 1
            stats["skip_reasons"][reason or "unresolved orientation"] = (
                stats["skip_reasons"].get(reason or "unresolved orientation", 0) + 1
            )
            continue

        rows = group.to_dict("records")
        try:
            if orientation == "same":
                implied_1, implied_2 = consensus_odds(rows)
            else:
                implied_2, implied_1 = consensus_odds(
                    [{"odds_1": r["odds_2"], "odds_2": r["odds_1"]} for r in rows]
                )
        except ValueError as exc:
            stats["n_skipped"] += 1
            stats["skip_reasons"][str(exc)] = stats["skip_reasons"].get(str(exc), 0) + 1
            continue

        valid = group.dropna(subset=["odds_1", "odds_2"])
        n_books = int(len(valid))
        # median decimal odds, oriented, for ROI settlement (not just implied probs)
        if orientation == "same":
            decimal_a = float(valid["odds_1"].median())
            decimal_b = float(valid["odds_2"].median())
        else:
            decimal_a = float(valid["odds_2"].median())
            decimal_b = float(valid["odds_1"].median())

        stats["n_id" if method == "id" else "n_name"] += 1
        records.append(
            {
                "fight_id": fight_id,
                "decimal_fights_a": decimal_a,
                "decimal_fights_b": decimal_b,
                "market_implied_fights_a": implied_1,
                "market_implied_fights_b": implied_2,
                "align_method": method,
                "n_books": n_books,
            }
        )

    aligned = pd.DataFrame.from_records(records)
    return aligned, stats


def validate_famous_fights(fights: pd.DataFrame, fighters: pd.DataFrame, aligned: pd.DataFrame) -> None:
    """Assert the devigged favorite matches the historically known favorite.

    Uses fights.parquet's original a/b convention (before the deterministic
    features.py corner swap), matching by resolved fighter ids + fight date
    proximity so this is independent of the id/name alignment path being
    tested. Raises AssertionError -- loudly, on purpose -- if alignment is
    broken.
    """
    name_index = build_name_index(fighters)
    aligned_idx = aligned.set_index("fight_id")
    fights_idx = fights.set_index("fight_id")
    checked = 0

    for name_x, name_y, expected_favorite, approx_date in FAMOUS_FIGHTS:
        id_x, _, _ = match_fighter_id(name_x, name_index)
        id_y, _, _ = match_fighter_id(name_y, name_index)
        id_fav, _, _ = match_fighter_id(expected_favorite, name_index)
        if id_x is None or id_y is None:
            print(f"  [famous-fight check] SKIP {name_x} vs {name_y}: name not resolved")
            continue

        candidates = fights[
            (
                (fights["fighter_a_id"] == id_x) & (fights["fighter_b_id"] == id_y)
            )
            | (
                (fights["fighter_a_id"] == id_y) & (fights["fighter_b_id"] == id_x)
            )
        ]
        candidates = candidates[candidates["fight_id"].isin(aligned_idx.index)]
        if candidates.empty:
            print(
                f"  [famous-fight check] SKIP {name_x} vs {name_y}: "
                "no matched-odds fight found"
            )
            continue
        # Disambiguate rematches: pick the candidate closest to approx_date.
        target = pd.Timestamp(approx_date)
        candidates = candidates.assign(
            _date_gap=(candidates["date"] - target).abs()
        ).sort_values("_date_gap")

        fight_id = candidates.iloc[0]["fight_id"]
        fights_row = fights_idx.loc[fight_id]
        market_row = aligned_idx.loc[fight_id]

        fav_is_a = fights_row["fighter_a_id"] == id_fav
        fav_implied = (
            market_row["market_implied_fights_a"]
            if fav_is_a
            else market_row["market_implied_fights_b"]
        )
        checked += 1
        assert fav_implied > 0.5, (
            f"ALIGNMENT VALIDATION FAILED for {name_x} vs {name_y}: expected "
            f"{expected_favorite} to be the market favorite (implied > 0.5) "
            f"but got {fav_implied:.3f} -- corner alignment is likely broken"
        )
        print(
            f"  [famous-fight check] OK  {expected_favorite} favorite "
            f"({fav_implied:.3f}) in {name_x} vs {name_y} ({fights_row['date'].date()})"
        )

    if checked == 0:
        raise RuntimeError(
            "famous-fight alignment validation matched ZERO known fights -- "
            "cannot confirm corner alignment is correct, aborting"
        )


def to_features_convention(merged: pd.DataFrame) -> pd.DataFrame:
    """Map fights.parquet a/b odds onto features.parquet's a/b convention.

    `merged` already carries features.parquet's own `swapped` column (set by
    `mma.features.swap_corner` at feature-build time) -- reusing it directly,
    rather than recomputing the md5 hash here, keeps this in lockstep with
    whatever features.parquet actually shipped even if `swap_corner`'s
    implementation ever changes. `swapped=True` means features "a" is
    fights.parquet's "b", so market_implied_a follows the same flip.
    """
    out = merged.copy()
    swapped = out["swapped"].to_numpy(dtype=bool)
    out["market_implied_a"] = np.where(
        swapped, out["market_implied_fights_b"], out["market_implied_fights_a"]
    )
    out["market_implied_b"] = 1.0 - out["market_implied_a"]
    out["decimal_a"] = np.where(swapped, out["decimal_fights_b"], out["decimal_fights_a"])
    out["decimal_b"] = np.where(swapped, out["decimal_fights_a"], out["decimal_fights_b"])
    return out


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1], right=True), 0, n_bins - 1)
    table = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        table.append(
            {
                "bin": f"[{bins[b]:.1f}, {bins[b + 1]:.1f})",
                "count": count,
                "mean_pred": round(float(p[mask].mean()), 4) if count else None,
                "empirical_rate": round(float(y[mask].mean()), 4) if count else None,
            }
        )
    return table


def roi_sweep(df: pd.DataFrame) -> dict:
    """Flat-stake (1 unit) simulated ROI, both edge directions, at each threshold.

    "Favorite edge": bet fighter_a whenever the model's p(a) exceeds the
    devigged market implied p(a) by the threshold, settled at fighter_a's
    decimal odds. "Underdog edge" is the symmetric bet on fighter_b whenever
    the model's implied p(b) exceeds the market's by the threshold.
    This is an in-sample-of-the-market backtest over historical closing-ish
    lines, NOT a live betting-strategy claim (no bankroll management, no
    line-shopping/timing realism, no transaction costs).
    """
    results = {}
    for threshold in THRESHOLDS:
        edge_a = df["model_p_a"] - df["market_implied_a"]
        bets_a = df[edge_a > threshold]
        won_a = bets_a["y_winner"] == 1
        profit_a = np.where(won_a, bets_a["decimal_a"] - 1.0, -1.0)

        edge_b = (1.0 - df["model_p_a"]) - df["market_implied_b"]
        bets_b = df[edge_b > threshold]
        won_b = bets_b["y_winner"] == 0
        profit_b = np.where(won_b, bets_b["decimal_b"] - 1.0, -1.0)

        def _block(bets, profit):
            n = len(bets)
            staked = float(n)
            net = float(profit.sum()) if n else 0.0
            return {
                "n_bets": n,
                "staked": staked,
                "net": round(net, 3),
                "roi_pct": round(100.0 * net / staked, 3) if staked else None,
            }

        results[f"{threshold:.2f}"] = {
            "favorite_edge_on_a": _block(bets_a, profit_a),
            "underdog_edge_on_b": _block(bets_b, profit_b),
        }
    return results


def winner_metrics(y, p) -> dict:
    return {
        "accuracy": round(accuracy(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier_score(y, p), 4),
    }


def main() -> None:
    print("Loading fights/features/fighters parquet ...")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    features = pd.read_parquet(PROCESSED / "features.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")

    print(f"Downloading odds dataset ({ODDS_DATASET}) via kagglehub ...")
    odds_raw = load_odds_raw()
    print(f"  {len(odds_raw)} odds rows across {odds_raw['fight_id'].nunique()} fights")

    print("Aligning odds to fights.parquet corner convention ...")
    aligned, stats = align_odds_to_fights(odds_raw, fights, fighters)
    print(
        f"  aligned {len(aligned)} fights "
        f"(by id: {stats['n_id']}, by name: {stats['n_name']}, skipped: {stats['n_skipped']})"
    )
    if stats["skip_reasons"]:
        print("  skip reasons (top 5):")
        for reason, count in sorted(stats["skip_reasons"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {count}x {reason}")

    print("Validating alignment against famous fights ...")
    validate_famous_fights(fights, fighters, aligned)

    print("Computing the neural ensemble's predictions (committed checkpoints, no refit) ...")
    matched = features.merge(
        aligned[
            [
                "fight_id", "market_implied_fights_a", "market_implied_fights_b",
                "decimal_fights_a", "decimal_fights_b", "align_method", "n_books",
            ]
        ],
        on="fight_id", how="inner",
    )
    matched = to_features_convention(matched)
    ensemble = Ensemble.load()
    winner_probs = ensemble.predict(matched)["winner_prob"]
    matched["model_p_a"] = winner_probs
    matched["y_winner"] = matched["y_winner"].astype(float)

    headline = matched[matched["date"] >= HEADLINE_START].reset_index(drop=True)
    all_matched = matched.reset_index(drop=True)

    print(
        f"  matched fights: {len(all_matched)} total, "
        f"{len(headline)} in the {HEADLINE_START}+ validation+test era"
    )

    def _cut(df: pd.DataFrame) -> dict:
        y = df["y_winner"].to_numpy(dtype=float)
        p_model = df["model_p_a"].to_numpy(dtype=float)
        p_market = df["market_implied_a"].to_numpy(dtype=float)
        model_block = winner_metrics(y, p_model)
        market_block = winner_metrics(y, p_market)
        delta = {
            key: round(model_block[key] - market_block[key], 4) for key in model_block
        }
        return {
            "n_fights": int(len(df)),
            "model": model_block,
            "market": market_block,
            "delta_model_minus_market": delta,
            "calibration": {
                "model": calibration_table(y, p_model),
                "market": calibration_table(y, p_market),
            },
        }

    headline_block = _cut(headline)
    headline_block["roi"] = roi_sweep(headline)

    results = {
        "computed_once_on": pd.Timestamp.today().date().isoformat(),
        "odds_dataset": ODDS_DATASET,
        "note": (
            "Betting odds are an EVALUATION-ONLY comparator here, never a "
            "model feature. Headline comparison restricts to date >= "
            f"{HEADLINE_START} (the model's validation+test era, never seen "
            "in training); 'all_matched_fights_secondary' includes pre-2021 "
            "fights the model WAS trained on, so treat it as a looser sanity "
            "cut, not an honest out-of-sample comparison."
        ),
        "alignment": {
            "n_aligned_by_id": stats["n_id"],
            "n_aligned_by_name": stats["n_name"],
            "n_skipped": stats["n_skipped"],
        },
        "headline_2021_plus": headline_block,
        "all_matched_fights_secondary": _cut(all_matched),
    }

    out_path = MODELS / "market_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    print("\n=== SUMMARY (headline, date >= 2021-01-01) ===")
    print(f"n_fights: {headline_block['n_fights']}")
    print(f"{'':12s}{'accuracy':>10s}{'log_loss':>10s}{'brier':>10s}")
    for label in ("model", "market"):
        b = headline_block[label]
        print(f"{label:12s}{b['accuracy']:>10.4f}{b['log_loss']:>10.4f}{b['brier']:>10.4f}")
    print("delta (model - market):", headline_block["delta_model_minus_market"])

    gate_delta = headline_block["delta_model_minus_market"]["log_loss"]
    if gate_delta < -0.02:
        print(
            "\n*** HONESTY GATE WARNING: model log-loss beats market by "
            f"{-gate_delta:.4f} (> 0.02) -- this almost certainly means "
            "odds/corner misalignment or leakage. DO NOT trust these "
            "numbers without investigating. Reporting DONE_WITH_CONCERNS. ***"
        )
    else:
        print("\nHonesty gate OK: model does not implausibly dominate the market.")

    print("\nROI sweep (headline set):")
    for threshold, block in headline_block["roi"].items():
        fav = block["favorite_edge_on_a"]
        dog = block["underdog_edge_on_b"]
        print(
            f"  threshold={threshold}  "
            f"favorite-edge: n={fav['n_bets']} roi={fav['roi_pct']}%  "
            f"underdog-edge: n={dog['n_bets']} roi={dog['roi_pct']}%"
        )


if __name__ == "__main__":
    main()
