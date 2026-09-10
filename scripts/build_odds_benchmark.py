"""Build the "does the model beat the market?" evaluation benchmark.

EVALUATION-ONLY: betting odds are a COMPARATOR here, never a model feature.
This script downloads a Kaggle odds-history dataset via kagglehub, aligns it
to our fighter_a/fighter_b feature convention using the shared 16-hex
ufcstats fight id, and compares the model's pre-fight win probabilities
against the devigged market-implied probabilities on the same historical
fights.

TWO MODES, because the deployed model changed what "out of sample" means.

``--mode oof`` (the default, and the honest one) scores the WALK-FORWARD
OUT-OF-FOLD predictions dumped by ``scripts/run_walkforward.py
--dump-predictions``: fold year Y's probabilities come from a model fitted on
fights before Y-1 and early-stopped on Y-1, so no fight is scored by a model
that saw it. Those rows are joined to the aligned odds BY FIGHT ID ONLY, and
the intersection is the comparison set. Writes models/market_benchmark_oof.json.
It also carries, clearly labelled as a diagnostic and never as the headline,
the same comparison run with the DEPLOYED refit model's own predictions --
which is an in-sample model against an out-of-sample market, and is exactly
the number this mode exists to avoid reporting.

The dump is paired to the pooled rows POSITIONALLY, so it belongs to the
feature table it was written from and is retired the moment the weekly refresh
lands another event. This mode therefore resolves its own source: it takes the
freshest committed (report, dump) pair whose row count still matches the table
(see OOF_SOURCES), and refuses -- naming the command that writes a fresh dump
-- rather than scoring one that does not.

``--mode deployed`` is the original path: score whatever ``BlendedPredictor``
serves today over every odds-matched fight, restricting the headline to
2021+. That is how models/market_benchmark.json was computed on 2026-07-15,
when the deployed model trained on pre-2021 data only and 2021+ was genuinely
held out. It is no longer true -- ``models/torch/metrics_val.json`` records
``mode=refit_through`` with ``train_through=2026-08-08``, so every one of
those fights is in the deployed model's training set. The July artifact is
therefore FROZEN as a historical record, and this mode refuses to overwrite
it without ``--force``.

Since SP3 the deployed scorer is the hybrid (`mma.inference.SimulatorPredictor`),
but its winner marginal IS the blend member's own array -- returned unchanged,
verified element-wise over all 4,804 pooled rows in
models/walkforward/sp3_decision.json -- and the winner probability is the only
head this benchmark compares. So both modes read the winner probability
through the blend, and the walk-forward dump oof mode scores is the deployed
hybrid's own -- `hybrid_e2` as SP3 measured it, or the same recipe re-run on a
newer table by scripts/revalidate_recipe.py.

  .venv/bin/python scripts/build_odds_benchmark.py                 # OOF (default)
  .venv/bin/python scripts/build_odds_benchmark.py --mode deployed --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mma.evaluate import accuracy, brier_score, log_loss
from mma.inference import BlendedPredictor, deployed_training_mask, load_deployed_metrics
from mma.odds import consensus_odds, decimal_to_implied, devig_pair, extract_fight_id
from mma.prospective import build_name_index, match_fighter_id
from mma.versioning import model_version
from mma.walkforward import make_folds

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
WALKFORWARD = MODELS / "walkforward"

ODDS_DATASET = "jerzyszocik/ufc-betting-odds-daily-dataset"
HEADLINE_START = "2021-01-01"  # the JULY 2026 model's validation+test era
THRESHOLDS = (0.00, 0.05, 0.10)

#: The frozen July 2026 record. Never overwritten without --force: it
#: describes a model that no longer exists, and that is the point of keeping it.
FROZEN_BENCHMARK = MODELS / "market_benchmark.json"
#: The out-of-fold benchmark this script's default mode writes.
OOF_BENCHMARK = MODELS / "market_benchmark_oof.json"
#: The deployed hybrid's own walk-forward run, and the pooled predictions it
#: dumped. `hybrid_e2` is the report `models/simulator.json` names as its
#: source; the dump is row-aligned to `mma.walkforward.pool`'s order. It is
#: SP3's evidence and is never refreshed in place -- see OOF_SOURCES.
OOF_REPORT = WALKFORWARD / "hybrid_e2.json"
OOF_PREDICTIONS = WALKFORWARD / "preds" / "hybrid_e2.json"
#: The same recipe re-run on the CURRENT table by
#: `scripts/revalidate_recipe.py`, which re-measures the deployed recipe's
#: bars whenever the feature table has moved on. Its hybrid run dumps its
#: predictions for exactly this reader.
OOF_REVALIDATION_REPORT = WALKFORWARD / "revalidation" / "revalidation_hybrid.json"
OOF_REVALIDATION_PREDICTIONS = WALKFORWARD / "revalidation" / "preds" / "revalidation_hybrid.json"
#: The (report, dump) pairs this mode will score, in preference order.
#:
#: The pairing is POSITIONAL, so a dump is only joinable to the table it was
#: written from -- and the table grows every time the weekly refresh lands a
#: new event, which retires the previous dump. The freshest run comes first
#: and `hybrid_e2` is the fallback for a table that has not moved since SP3.
#: Refreshing `hybrid_e2` in place would be the shorter fix and the wrong
#: one: `models/simulator.json` names it `source_report`,
#: `models/walkforward/sp3_decision.json` rests on its numbers, and
#: `scripts/revalidate_recipe.py` exists on the promise that the reports the
#: original decisions rest on are never overwritten.
OOF_SOURCES = (
    (OOF_REVALIDATION_REPORT, OOF_REVALIDATION_PREDICTIONS),
    (OOF_REPORT, OOF_PREDICTIONS),
)

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


def roi_sweep(df: pd.DataFrame, model_col: str = "model_p_a") -> dict:
    """Flat-stake (1 unit) simulated ROI, both edge directions, at each threshold.

    "Favorite edge": bet fighter_a whenever the model's p(a) exceeds the
    devigged market implied p(a) by the threshold, settled at fighter_a's
    decimal odds. "Underdog edge" is the symmetric bet on fighter_b whenever
    the model's implied p(b) exceeds the market's by the threshold.
    This is an in-sample-of-the-market backtest over historical closing-ish
    lines, NOT a live betting-strategy claim (no bankroll management, no
    line-shopping/timing realism, no transaction costs).

    `model_col` names the probability column so the same sweep can be run on
    the out-of-fold predictions and on the deployed model's in-sample ones.
    """
    results = {}
    for threshold in THRESHOLDS:
        edge_a = df[model_col] - df["market_implied_a"]
        bets_a = df[edge_a > threshold]
        won_a = bets_a["y_winner"] == 1
        profit_a = np.where(won_a, bets_a["decimal_a"] - 1.0, -1.0)

        edge_b = (1.0 - df[model_col]) - df["market_implied_b"]
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


def compare_block(df: pd.DataFrame, model_col: str = "model_p_a") -> dict:
    """Model-vs-market metrics, deltas and 10-bin calibration for one row set.

    `model_col` is the model's P(fighter_a wins) in features.parquet's corner
    convention -- the out-of-fold column by default, or the deployed model's
    own predictions for the in-sample diagnostic. The market column is fixed:
    both are compared against the same devigged line on the same fights, so
    the only thing that changes between the two blocks is which model spoke.
    """
    y = df["y_winner"].to_numpy(dtype=float)
    p_model = df[model_col].to_numpy(dtype=float)
    p_market = df["market_implied_a"].to_numpy(dtype=float)
    model_block = winner_metrics(y, p_model)
    market_block = winner_metrics(y, p_market)
    return {
        "n_fights": int(len(df)),
        "model": model_block,
        "market": market_block,
        "delta_model_minus_market": {
            key: round(model_block[key] - market_block[key], 4) for key in model_block
        },
        "calibration": {
            "model": calibration_table(y, p_model),
            "market": calibration_table(y, p_market),
        },
    }


# --------------------------------------------------------------------------
# Walk-forward out-of-fold mode
# --------------------------------------------------------------------------


def load_features() -> pd.DataFrame:
    """features.parquet in `scripts/run_walkforward.py`'s row order.

    The harness sorts by date with a stable sort and resets the index before
    it builds a single fold, so any frame that wants to line up with a
    prediction dump has to start from the same sort.
    """
    return (
        pd.read_parquet(PROCESSED / "features.parquet")
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


def pooled_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Rebuild `mma.walkforward.pool`'s row order, carrying `fold_year`.

    `run_walkforward.prediction_dump` writes the pooled evaluation rows
    positionally -- fold year, y_winner and p_winner as three parallel lists
    with no fight id -- in the concatenation order of the folds. This walks
    the same folds in the same order over the same table, so row i here is
    row i there, which is what lets the dump be joined to anything at all.
    The fold masks are pairwise disjoint by construction (`make_folds` gives
    each year a half-open date window and the last fold absorbs the tail), so
    no fight appears twice.
    """
    frames = []
    for fold in make_folds(features["date"]):
        block = features.loc[fold.eval].copy()
        block["fold_year"] = fold.year
        frames.append(block)
    return pd.concat(frames).reset_index(drop=True)


def pooled_row_count(dates: pd.Series) -> int:
    """How many rows `pooled_frame` would build, from the date column alone.

    The same fold masks over the same table, counted rather than materialised,
    so a caller that only needs to know whether a dump still pairs -- the
    weekly staleness read -- does not have to load the whole feature table to
    find out. It must agree with `len(pooled_frame(features))` exactly, or it
    would clear a pairing `attach_oof_predictions` then rejects.
    """
    return int(sum(int(np.asarray(fold.eval, dtype=bool).sum())
                   for fold in make_folds(dates)))


def attach_oof_predictions(pooled: pd.DataFrame, dump: dict) -> pd.DataFrame:
    """`pooled` with the dump's out-of-fold winner probability as `model_p_a`.

    The dump is positional, so the pairing is only correct if the rebuilt
    frame IS the frame the dump was written from. That is checked, not
    assumed: the row count, the fold-year sequence and the realised outcome
    sequence must all match element for element. A silently mis-paired join
    would still produce a plausible-looking log-loss, which is precisely the
    failure this benchmark must not ship, so every mismatch raises.
    """
    n_dump = int(dump["n"])
    if len(pooled) != n_dump:
        raise ValueError(
            f"prediction dump has {n_dump} pooled rows but the rebuilt "
            f"walk-forward frame has {len(pooled)}; the dump was written from "
            "a different feature table or a different fold set"
        )
    for column, key in (("fold_year", "fold_year"), ("y_winner", "y_winner")):
        rebuilt = pooled[column].to_numpy(dtype=float)
        dumped = np.asarray(dump[key], dtype=float)
        if not np.array_equal(rebuilt, dumped):
            n_bad = int((rebuilt != dumped).sum())
            raise ValueError(
                f"rebuilt {column!r} disagrees with the prediction dump on "
                f"{n_bad} of {len(rebuilt)} rows; the positional pairing is "
                "not valid and no metric computed from it would be"
            )
    out = pooled.copy()
    out["model_p_a"] = np.asarray(dump["p_winner"], dtype=float)
    if not ((out["model_p_a"] > 0.0) & (out["model_p_a"] < 1.0)).all():
        raise ValueError("out-of-fold winner probabilities must lie strictly in (0, 1)")
    return out


def _rel(path) -> str:
    """Repo-relative when the path is in the repo, absolute otherwise."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def oof_source_status(pooled_rows: int, sources=OOF_SOURCES) -> list[dict]:
    """For each candidate (report, dump) pair: is it there, and does it pair?

    Reads each dump's own `n` and nothing else -- the row count is the first
    thing `attach_oof_predictions` checks and the only one that can be read
    without loading the predictions themselves, which is what keeps this cheap
    enough for the weekly staleness step to call.
    """
    rows = []
    for report, predictions in sources:
        exists = Path(report).exists() and Path(predictions).exists()
        n_dump = None
        if exists:
            n_dump = int(json.loads(Path(predictions).read_text())["n"])
        rows.append({
            "report": _rel(report),
            "predictions": _rel(predictions),
            "exists": bool(exists),
            "n_dump": n_dump,
            "n_pooled_walkforward_rows": int(pooled_rows),
            "pairs": bool(n_dump == pooled_rows),
        })
    return rows


def resolve_oof_source(pooled_rows: int, sources=OOF_SOURCES) -> tuple[Path, Path]:
    """The first (report, dump) pair whose dump still pairs with this table.

    A dump written from a different table cannot be joined to this one at all
    -- the join is positional -- so there is no benchmark to build and the
    only useful thing to do is say which command writes a dump that pairs.
    Falling back to the nearest-looking dump would produce a plausible number
    describing predictions no model made about these fights, which is the
    failure `attach_oof_predictions` was written to make impossible.
    """
    status = oof_source_status(pooled_rows, sources)
    for (report, predictions), row in zip(sources, status):
        if row["pairs"]:
            return Path(report), Path(predictions)
    lines = "\n".join(
        f"  {row['predictions']}: "
        + ("absent" if not row["exists"] else f"{row['n_dump']} pooled rows")
        for row in status
    )
    raise SystemExit(
        f"no walk-forward prediction dump pairs with the current feature table, "
        f"which rebuilds {pooled_rows} pooled walk-forward rows:\n{lines}\n"
        "The join is positional, so a dump from a different table cannot be "
        "scored against this one. Run `python scripts/revalidate_recipe.py` to "
        "re-run the deployed recipe on the current table -- it re-measures every "
        "bar and dumps the hybrid's predictions -- then re-run this script."
    )


def check_dump_reproduces_report(oof: pd.DataFrame, report: dict) -> dict:
    """Recompute the report's pooled winner metrics from the dump.

    The dump carries no metrics and the report carries no predictions, so
    this is the one place the two can be tied together. Disagreement means
    the dump and the report came from different runs, and the benchmark
    would then be describing a model no report vouches for.
    """
    y = oof["y_winner"].to_numpy(dtype=float)
    p = oof["model_p_a"].to_numpy(dtype=float)
    recomputed = {
        "n": int(len(oof)),
        "winner_log_loss": round(log_loss(y, p), 4),
        "accuracy": round(accuracy(y, p), 4),
        "brier": round(brier_score(y, p), 4),
    }
    pooled = report["pooled"]
    mismatched = {
        key: (value, pooled[key])
        for key, value in recomputed.items()
        if pooled.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "prediction dump does not reproduce the walk-forward report's "
            f"pooled winner metrics: {mismatched} (recomputed, reported)"
        )
    return recomputed


def join_odds_by_id(rows: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    """Inner-join predictions to aligned odds on `fight_id` and nothing else.

    No name matching, no date proximity, no positional guess: the 16-hex
    ufcstats fight id is present on both sides and is the whole key. Both
    sides must be unique on it -- a duplicate would multiply rows into the
    comparison and quietly reweight the metrics -- so that is enforced rather
    than left to `merge` to resolve.
    """
    odds_columns = [
        "fight_id", "market_implied_fights_a", "market_implied_fights_b",
        "decimal_fights_a", "decimal_fights_b", "align_method", "n_books",
    ]
    for label, frame in (("predictions", rows), ("odds", aligned)):
        if "fight_id" not in frame.columns:
            raise ValueError(f"{label} frame has no fight_id column to join on")
        n_duplicated = int(frame["fight_id"].duplicated().sum())
        if n_duplicated:
            raise ValueError(
                f"{label} frame has {n_duplicated} duplicate fight_id(s); an "
                "id-only join would multiply rows into the comparison"
            )
    return rows.merge(
        aligned[odds_columns], on="fight_id", how="inner", validate="one_to_one"
    ).reset_index(drop=True)


def fold_year_coverage(oof: pd.DataFrame, matched: pd.DataFrame) -> dict:
    """Per-fold-year odds coverage and out-of-fold log-loss, matched vs all.

    The intersection is not a random sample of the walk-forward rows -- odds
    coverage is thinner in the early years -- so the honest thing is to show
    what the model scores on the rows the market also priced AND on the rows
    it did not, year by year, rather than to assert the subset is neutral.
    """
    out = {}
    matched_ids = set(matched["fight_id"])
    for year, block in oof.groupby("fold_year"):
        in_odds = block["fight_id"].isin(matched_ids).to_numpy()
        row = {
            "n_walkforward": int(len(block)),
            "n_odds_matched": int(in_odds.sum()),
            "odds_coverage": round(float(in_odds.mean()), 4),
            "model_log_loss_matched": None,
            "model_log_loss_unmatched": None,
        }
        for label, mask in (("matched", in_odds), ("unmatched", ~in_odds)):
            if mask.any():
                row[f"model_log_loss_{label}"] = round(
                    log_loss(
                        block.loc[mask, "y_winner"].to_numpy(dtype=float),
                        block.loc[mask, "model_p_a"].to_numpy(dtype=float),
                    ),
                    4,
                )
        out[str(int(year))] = row
    return out


def july_comparison(oof_2021_plus: dict) -> dict | None:
    """The frozen July artifact's headline beside the same date cut, out of fold.

    Returns None when the July artifact is absent. Whether the two cuts cover
    the SAME fights is established rather than assumed: the walk-forward's
    fold years start at 2018, so every 2021+ odds-matched fight should also
    be a walk-forward evaluation row, and if that holds then the row count
    and all three market metrics -- which depend on the fights and the lines,
    not on the model -- must agree exactly. When they do, the two model
    numbers are directly subtractable; when they do not, this says so and the
    change is a direction, not a difference of differences.
    """
    if not FROZEN_BENCHMARK.exists():
        return None
    july = json.loads(FROZEN_BENCHMARK.read_text())["headline_2021_plus"]
    july_gap = july["delta_model_minus_market"]["log_loss"]
    now_gap = oof_2021_plus["delta_model_minus_market"]["log_loss"]
    change = round(now_gap - july_gap, 4)
    same_fights = (
        july["n_fights"] == oof_2021_plus["n_fights"]
        and july["market"] == oof_2021_plus["market"]
    )
    return {
        "july_2026_artifact": {
            "model": "the pre-refit torch ensemble, trained on pre-2021 data only",
            "n_fights": july["n_fights"],
            "model_log_loss": july["model"]["log_loss"],
            "market_log_loss": july["market"]["log_loss"],
            "gap_market_favour": july_gap,
        },
        "out_of_fold_2021_plus": {
            "model": "the deployed hybrid's winner marginal, out of fold",
            "n_fights": oof_2021_plus["n_fights"],
            "model_log_loss": oof_2021_plus["model"]["log_loss"],
            "market_log_loss": oof_2021_plus["market"]["log_loss"],
            "gap_market_favour": now_gap,
        },
        "change_in_gap": change,
        "direction": (
            "narrowed" if change < 0 else "widened" if change > 0 else "unchanged"
        ),
        "same_fight_set": bool(same_fights),
        "same_fight_set_evidence": (
            "Identical n and identical market accuracy/log-loss/Brier: the "
            "market block depends only on which fights are in the cut and "
            "what the lines were, so agreement to four decimals on all three "
            "means the two cuts are the same fights. Only the model changed."
            if same_fights else
            "The two cuts do NOT cover the same fights (n or the market "
            "metrics differ), so read the direction of the change rather "
            "than subtracting one gap from the other."
        ),
    }


def main_deployed(force: bool = False) -> None:
    """The original path: score whatever serves today over the matched fights.

    This is how models/market_benchmark.json was produced on 2026-07-15, when
    the model trained on pre-2021 data only. It does not describe the deployed
    model any more (which trains through the latest event), so it will not
    overwrite the frozen artifact unless the caller insists.
    """
    if FROZEN_BENCHMARK.exists() and not force:
        raise SystemExit(
            f"{FROZEN_BENCHMARK.relative_to(ROOT)} is the FROZEN July 2026 "
            "record of a model that no longer exists, and this mode would "
            "overwrite it with an in-sample comparison (the deployed model "
            "trains through the latest event). Run the default --mode oof for "
            "the honest recomputation, or pass --force if you really mean to "
            "replace the frozen artifact."
        )
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

    print("Computing the deployed blend's predictions (committed artifacts, no refit) ...")
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
    predictor = BlendedPredictor.load(ROOT)
    winner_probs = predictor.predict(matched)["winner_prob"]
    matched["model_p_a"] = winner_probs
    matched["y_winner"] = matched["y_winner"].astype(float)

    headline = matched[matched["date"] >= HEADLINE_START].reset_index(drop=True)
    all_matched = matched.reset_index(drop=True)

    print(
        f"  matched fights: {len(all_matched)} total, "
        f"{len(headline)} in the {HEADLINE_START}+ validation+test era"
    )

    headline_block = compare_block(headline)
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
        "all_matched_fights_secondary": compare_block(all_matched),
    }

    out_path = FROZEN_BENCHMARK
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


def main_oof(predictions: Path, report: Path, out_path: Path) -> None:
    """The honest recomputation: out-of-fold predictions against the market.

    Every probability here comes from a fold model that never saw the fight
    it is scoring. The deployed refit model's own predictions are computed on
    the SAME fights and reported alongside as a labelled diagnostic, because
    the size of the gap between the two blocks is the argument for why this
    mode exists.
    """
    print("Loading fights/features/fighters parquet ...")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    features = load_features()

    print(f"Downloading odds dataset ({ODDS_DATASET}) via kagglehub ...")
    odds_raw = load_odds_raw()
    print(f"  {len(odds_raw)} odds rows across {odds_raw['fight_id'].nunique()} fights")

    print("Aligning odds to fights.parquet corner convention ...")
    aligned, stats = align_odds_to_fights(odds_raw, fights, fighters)
    print(
        f"  aligned {len(aligned)} fights "
        f"(by id: {stats['n_id']}, by name: {stats['n_name']}, skipped: {stats['n_skipped']})"
    )

    print("Validating alignment against famous fights ...")
    validate_famous_fights(fights, fighters, aligned)

    print(f"Rebuilding the walk-forward pooled rows and pairing {predictions.name} ...")
    dump = json.loads(predictions.read_text())
    wf_report = json.loads(report.read_text())
    oof = attach_oof_predictions(pooled_frame(features), dump)
    reproduced = check_dump_reproduces_report(oof, wf_report)
    print(f"  {len(oof)} out-of-fold rows; dump reproduces the report's pooled metrics")

    print("Joining out-of-fold predictions to the aligned odds by fight id ...")
    matched = to_features_convention(join_odds_by_id(oof, aligned))
    matched["y_winner"] = matched["y_winner"].astype(float)
    print(
        f"  {len(matched)} fights are both walk-forward evaluation rows and "
        f"odds-matched ({100.0 * len(matched) / len(oof):.1f}% of the "
        f"{len(oof)} walk-forward rows)"
    )

    print("Computing the DEPLOYED model's own predictions on the same fights ...")
    print("  (in-sample by construction -- a labelled diagnostic, never the headline)")
    deployed = BlendedPredictor.load(ROOT)
    matched["deployed_p_a"] = deployed.predict(matched)["winner_prob"]
    metrics = load_deployed_metrics()
    trained_on = deployed_training_mask(matched, metrics)

    coverage = fold_year_coverage(oof, matched)
    thinnest_year = min(coverage, key=lambda year: coverage[year]["odds_coverage"])
    thinnest = {"fold_year": int(thinnest_year), **coverage[thinnest_year]}

    headline = compare_block(matched)
    headline["roi"] = roi_sweep(matched)
    since_2021 = matched[matched["date"] >= HEADLINE_START].reset_index(drop=True)
    in_sample = compare_block(matched, model_col="deployed_p_a")
    in_sample["roi"] = roi_sweep(matched, model_col="deployed_p_a")
    in_sample_gap = in_sample["delta_model_minus_market"]["log_loss"]

    results = {
        "computed_on": pd.Timestamp.today().date().isoformat(),
        "describes": (
            "The DEPLOYED model (the SP3 hybrid), evaluated out of fold. Its "
            "winner marginal is the deployed blend's own array, returned "
            "unchanged by the hybrid -- see models/walkforward/sp3_decision.json "
            "-- and the winner probability is the only head compared here."
        ),
        "odds_dataset": ODDS_DATASET,
        "note": (
            "Betting odds are an EVALUATION-ONLY comparator here, never a "
            "model feature. Every model probability in 'headline_out_of_fold' "
            "and 'out_of_fold_2021_plus' comes from a walk-forward fold model "
            "fitted on fights strictly before the fold's inner-validation "
            "year, so no fight is scored by a model that saw it. "
            "'in_sample_diagnostic' scores the SAME fights with the deployed "
            "refit model, which trained on all of them; it is reported to "
            "show how much an in-sample comparison would flatter the model, "
            "and must never be quoted as the model-vs-market result."
        ),
        "provenance": {
            "predictions": str(predictions.relative_to(ROOT)),
            "predictions_name": dump["name"],
            "walkforward_report": str(report.relative_to(ROOT)),
            "fold_years": [int(y) for y in wf_report["fold_years"]],
            "n_pooled_walkforward_rows": int(dump["n"]),
            "dump_reproduces_report_pooled": reproduced,
            "deployed_model_version": model_version(ROOT),
            "deployed_training_recipe": {
                "mode": metrics.get("mode"),
                "train_through": metrics.get("train_through"),
                "n_train": metrics.get("n_train"),
            },
            "frozen_predecessor": str(FROZEN_BENCHMARK.relative_to(ROOT)),
        },
        "alignment": {
            "n_aligned_by_id": stats["n_id"],
            "n_aligned_by_name": stats["n_name"],
            "n_skipped": stats["n_skipped"],
        },
        "intersection": {
            "n_walkforward_rows": int(len(oof)),
            "n_odds_aligned_fights": int(len(aligned)),
            "n_intersection": int(len(matched)),
            "join_key": "fight_id (the shared 16-hex ufcstats id), and nothing else",
            "n_intersection_rows_in_deployed_training_window": int(trained_on.sum()),
            "deployed_training_window_covers_the_whole_intersection": bool(trained_on.all()),
            "odds_coverage_of_walkforward_rows": round(len(matched) / len(oof), 4),
            "thinnest_fold_year": thinnest,
            "coverage_note": (
                "Odds coverage is not uniform across the fold years: the "
                f"dataset lags, so {thinnest['fold_year']} is only "
                f"{thinnest['odds_coverage']:.1%} covered. The pooled "
                "out-of-fold cut is therefore weighted toward the earlier "
                "fold years. See odds_coverage_by_fold_year for the "
                "model's log-loss on the covered and uncovered rows of each "
                "year, which is what says whether the covered subset is a "
                "harder or easier sample than the rest."
            ),
        },
        "headline_out_of_fold": headline,
        "out_of_fold_2021_plus": compare_block(since_2021),
        "in_sample_diagnostic": {
            "beats_the_market": bool(in_sample_gap < 0),
            "honesty_gate_would_have_caught_it": bool(in_sample_gap < -0.02),
            "honesty_gate_note": (
                "The gate in this script fires only when the model beats the "
                "market by more than 0.02 log-loss. An in-sample "
                "recomputation lands inside that tolerance, so the gate "
                "would have passed it. A leakage gate calibrated on 'is this "
                "edge implausibly large' does not catch 'this model was "
                "trained on the evaluation set' -- only using out-of-fold "
                "predictions does."
            ),
            "warning": (
                "NOT the headline. The deployed model trained on every fight "
                "scored here, so this compares an in-sample model against an "
                "out-of-sample market. It is the number the out-of-fold "
                "headline exists to replace."
            ),
            **in_sample,
        },
        "comparison_with_frozen_july_artifact": july_comparison(
            compare_block(since_2021)
        ),
        "odds_coverage_by_fold_year": coverage,
    }

    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out_path}")

    print("\n=== SUMMARY (out of fold, fold years "
          f"{results['provenance']['fold_years'][0]}-"
          f"{results['provenance']['fold_years'][-1]}) ===")
    print(f"n_fights: {headline['n_fights']}")
    print(f"{'':22s}{'accuracy':>10s}{'log_loss':>10s}{'brier':>10s}")
    for label, block in (("model (out of fold)", headline["model"]),
                         ("market", headline["market"]),
                         ("model (IN SAMPLE)", in_sample["model"])):
        print(f"{label:22s}{block['accuracy']:>10.4f}{block['log_loss']:>10.4f}{block['brier']:>10.4f}")
    print("delta out of fold (model - market):", headline["delta_model_minus_market"])
    print("delta in sample  (model - market):", in_sample["delta_model_minus_market"])

    gate_delta = headline["delta_model_minus_market"]["log_loss"]
    if gate_delta < -0.02:
        print(
            "\n*** HONESTY GATE WARNING: model log-loss beats market by "
            f"{-gate_delta:.4f} (> 0.02) -- this almost certainly means "
            "odds/corner misalignment or leakage. DO NOT trust these "
            "numbers without investigating. ***"
        )
    else:
        print("\nHonesty gate OK: model does not implausibly dominate the market.")

    july = results["comparison_with_frozen_july_artifact"]
    if july is not None:
        print(
            f"\nVersus the frozen July 2026 artifact, on 2021+ fights: the "
            f"market's log-loss edge {july['direction']} from "
            f"{july['july_2026_artifact']['gap_market_favour']:+.4f} "
            f"(n={july['july_2026_artifact']['n_fights']}) to "
            f"{july['out_of_fold_2021_plus']['gap_market_favour']:+.4f} "
            f"(n={july['out_of_fold_2021_plus']['n_fights']})."
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=("oof", "deployed"), default="oof",
        help="'oof' (default) scores the walk-forward out-of-fold predictions; "
             "'deployed' scores whatever serves today, which is in-sample for "
             "every fight in this dataset",
    )
    parser.add_argument("--predictions", type=Path, default=None,
                        help="oof mode: the --dump-predictions JSON to score "
                             "(default: the freshest committed dump that pairs "
                             "with the current feature table)")
    parser.add_argument("--report", type=Path, default=None,
                        help="oof mode: the walk-forward report those predictions "
                             "came from (default: resolved alongside --predictions)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (defaults to the mode's own artifact)")
    parser.add_argument("--force", action="store_true",
                        help="deployed mode: overwrite the frozen July 2026 artifact")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode == "deployed":
        if args.out is not None:
            raise SystemExit("--out is not supported in deployed mode")
        main_deployed(force=args.force)
        return
    if (args.predictions is None) != (args.report is None):
        raise SystemExit(
            "--predictions and --report name one run's two halves; pass both or "
            "neither (neither resolves the freshest pair that matches the table)"
        )
    if args.predictions is None:
        dates = pd.read_parquet(PROCESSED / "features.parquet", columns=["date"])["date"]
        report, predictions = resolve_oof_source(pooled_row_count(dates))
    else:
        report, predictions = args.report, args.predictions
    main_oof(predictions, report, args.out or OOF_BENCHMARK)


if __name__ == "__main__":
    main()

