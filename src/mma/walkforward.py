"""Expanding-window walk-forward evaluation.

Fold Y trains on fights before Y-1, early-stops on year Y-1, and is scored
on year Y (the last fold absorbs everything from 2025 onward). Pooled
metrics over all fold years are the headline; per-fold and slice metrics
are reported alongside. See the v3 design spec §4 SP1 / §5 for the locked
protocol and shipping bars.

Public functions: `make_folds` and `recency_weights` build the folds and
sample weights used to train each candidate; `score_rows` computes the
metric dict for one set of rows; `slice_masks` returns the named row
subsets (`debut`, `womens`, `five_round`, and `external_missing` when
present) reported for every candidate; `pool` concatenates per-fold
(mask, prediction) pairs into one row-aligned frame and prediction dict
for pooled scoring; `build_report` assembles the full JSON report for a
candidate; `bar_check` applies the pre-registered winner bar to a
candidate/incumbent report pair; `paired_delta` computes the plain B-minus-A
pooled/per-fold delta and the refit deployment-recipe gate (no bar, no
comparability check) shared by `scripts/refit_decision.py`. A report has the shape `{name, config,
fold_years, folds: {year: metrics}, pooled: metrics, slices: {name:
metrics}, fit_info: {key: [per-fold values]}}`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mma.evaluate import (
    accuracy, brier_score, expected_calibration_error, joint_outcome_log_loss,
    log_loss, macro_f1,
)

FOLD_YEARS = tuple(range(2018, 2026))
_DAYS_PER_YEAR = 365.25
MIN_FOLD_REGRESSION = 0.01
MIN_BAR = 0.003


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray
    inner_val: np.ndarray
    eval: np.ndarray

    @property
    def eval_start(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=1, day=1)


def make_folds(
    dates: pd.Series, fold_years=FOLD_YEARS, train_start: str | None = None
) -> list[Fold]:
    """Boolean masks (aligned to `dates`) for every fold year.

    The largest year in `fold_years` is unbounded above (it absorbs every
    later fight), which is how the default 2025 fold takes in the partial
    2026."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    start = pd.Timestamp(train_start) if train_start else None
    folds = []
    last = max(fold_years)
    for year in fold_years:
        val_start = pd.Timestamp(year=year - 1, month=1, day=1)
        eval_start = pd.Timestamp(year=year, month=1, day=1)
        eval_end = pd.Timestamp(year=year + 1, month=1, day=1)
        train = (d < val_start).to_numpy()
        if start is not None:
            train = train & (d >= start).to_numpy()
        inner_val = ((d >= val_start) & (d < eval_start)).to_numpy()
        eval_mask = (d >= eval_start).to_numpy()
        if year != last:
            eval_mask = eval_mask & (d < eval_end).to_numpy()
        folds.append(Fold(year, train, inner_val, eval_mask))
    return folds


def recency_weights(
    dates: pd.Series, reference: pd.Timestamp, half_life_years: float | None
) -> np.ndarray:
    """0.5 ** (age_in_years / half_life); uniform ones when half_life is None."""
    if half_life_years is None:
        return np.ones(len(dates), dtype=float)
    age_years = (reference - pd.to_datetime(dates)).dt.days.to_numpy() / _DAYS_PER_YEAR
    return np.power(0.5, np.clip(age_years, 0, None) / half_life_years)


def _labels(series: pd.Series) -> np.ndarray:
    """Object array with None for missing (works for pandas string dtype)."""
    return np.array([None if pd.isna(v) else str(v) for v in series], dtype=object)


def score_rows(features: pd.DataFrame, pred: dict, method_classes, round_classes) -> dict:
    """Metrics for one set of rows. `pred` has 'winner' (n,), and 'method'
    (n,3) / 'round' (n,4) or None when the candidate has no such head."""
    y = features["y_winner"].to_numpy(dtype=float)
    p = np.asarray(pred["winner"], dtype=float)
    out = {
        "n": int(len(y)),
        "winner_log_loss": round(log_loss(y, p), 4),
        "accuracy": round(accuracy(y, p), 4),
        "brier": round(brier_score(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "joint_log_loss": None, "method_macro_f1": None, "round_macro_f1": None,
        "n_method": int(features["y_method"].notna().sum()),
        "n_round": int(features["y_finish_round"].notna().sum()),
    }
    if pred.get("method") is not None and pred.get("round") is not None:
        y_m = _labels(features["y_method"])
        y_r = _labels(features["y_finish_round"])
        out["joint_log_loss"] = round(joint_outcome_log_loss(
            y, y_m, y_r, p, pred["method"], pred["round"], method_classes, round_classes), 4)
        known = features["y_method"].notna().to_numpy()
        if known.any():
            m_pred = [method_classes[i] for i in np.asarray(pred["method"])[known].argmax(axis=1)]
            out["method_macro_f1"] = round(macro_f1(list(y_m[known]), m_pred), 4)
        finish = features["y_finish_round"].notna().to_numpy()
        if finish.any():
            r_pred = [round_classes[i] for i in np.asarray(pred["round"])[finish].argmax(axis=1)]
            out["round_macro_f1"] = round(macro_f1(list(y_r[finish]), r_pred), 4)
    return out


def slice_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Named row subsets reported for every candidate (spec §4 SP1)."""
    masks = {
        "debut": (features["debut_a"].astype(bool) | features["debut_b"].astype(bool)).to_numpy(),
        "womens": features["weight_class"].astype("string").str.startswith("Women").fillna(False).to_numpy(dtype=bool),
        "five_round": (features["scheduled_rounds"].fillna(3) >= 5).to_numpy(dtype=bool),
    }
    if "external_missing" in features.columns:  # added by SP2's external-data block
        masks["external_missing"] = features["external_missing"].astype(bool).to_numpy()
    return masks


def pool(features: pd.DataFrame, fold_preds: list[tuple[np.ndarray, dict]]):
    """Concatenate (mask, pred) pairs into one row-aligned frame + prediction dict.

    Masks must be pairwise disjoint (a row scored twice would be double-counted
    in the pooled metrics). Output row order is the concatenation order of
    `fold_preds`, not the original order of `features`."""
    frames, winner, method, rounds = [], [], [], []
    has_heads = all(p.get("method") is not None and p.get("round") is not None for _, p in fold_preds)
    for mask, pred in fold_preds:
        frames.append(features.loc[mask])
        winner.append(np.asarray(pred["winner"], dtype=float))
        if has_heads:
            method.append(np.asarray(pred["method"], dtype=float))
            rounds.append(np.asarray(pred["round"], dtype=float))
    pooled = {
        "winner": np.concatenate(winner),
        "method": np.concatenate(method) if has_heads else None,
        "round": np.concatenate(rounds) if has_heads else None,
    }
    return pd.concat(frames).reset_index(drop=True), pooled


def _subset(pred: dict, mask: np.ndarray) -> dict:
    return {k: (np.asarray(v)[mask] if v is not None else None) for k, v in pred.items()}


def _to_python(value):
    """Recursively cast numpy scalars/arrays to plain Python so json.dumps works."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_python(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    return value


def build_report(name, config, features, fold_results, method_classes, round_classes) -> dict:
    """Assemble the JSON report. fold_results: list of (year, eval_mask, pred, fit_info).

    Folds whose eval mask selects no rows are skipped with a warning (they
    would otherwise contribute NaN metrics); fit_info values are cast to
    plain Python types so the report is JSON-serialisable."""
    folds, fit_info, kept = {}, {}, []
    for year, mask, pred, info in fold_results:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            print(f"build_report: fold {year} has no eval rows; skipping")
            continue
        kept.append((year, mask, pred, info))
        folds[str(year)] = score_rows(features.loc[mask], pred, method_classes, round_classes)
        for key, value in (info or {}).items():
            fit_info.setdefault(key, []).append(_to_python(value))
    fold_results = kept
    pooled_feats, pooled_pred = pool(features, [(m, p) for _, m, p, _ in fold_results])
    slices = {}
    for slice_name, mask in slice_masks(pooled_feats).items():
        if mask.any():
            slices[slice_name] = score_rows(pooled_feats.loc[mask], _subset(pooled_pred, mask),
                                            method_classes, round_classes)
    return {
        "name": name, "config": config, "fold_years": [int(y) for y, *_ in fold_results],
        "folds": folds,
        "pooled": score_rows(pooled_feats, pooled_pred, method_classes, round_classes),
        "slices": slices, "fit_info": fit_info,
    }


def bar_check(candidate: dict, incumbent: dict, sigma_seed: float) -> dict:
    """Pre-registered winner bar (spec §5): pooled Δ log-loss < −max(MIN_BAR, 2σ_seed)
    and no fold year worse than the incumbent by more than MIN_FOLD_REGRESSION.

    Deltas are candidate − incumbent (negative = better). Both the bar and the
    pooled delta are rounded to 6 dp before the strict comparison so that a
    candidate sitting exactly on the bar does not clear it on float noise. The pair is
    `comparable` only when both reports cover the same fold years and the
    same pooled `n`; a non-comparable pair never ships."""
    if sigma_seed is None or not np.isfinite(sigma_seed):
        raise ValueError("sigma_seed must be a finite number; run scripts/noise_floor.py first")
    bar = round(max(MIN_BAR, 2.0 * float(sigma_seed)), 6)
    delta = round(candidate["pooled"]["winner_log_loss"] - incumbent["pooled"]["winner_log_loss"], 6)
    cand_years, inc_years = set(candidate["folds"]), set(incumbent["folds"])
    missing_folds = sorted(cand_years ^ inc_years)
    comparable = (not missing_folds
                  and candidate["pooled"].get("n") == incumbent["pooled"].get("n"))
    fold_deltas = {
        year: candidate["folds"][year]["winner_log_loss"] - incumbent["folds"][year]["winner_log_loss"]
        for year in candidate["folds"] if year in incumbent["folds"]
    }
    worst = round(max(fold_deltas.values()), 6) if fold_deltas else 0.0  # 0.65-0.64 != 0.01 in floats
    clears = delta < -bar
    no_regression = worst <= MIN_FOLD_REGRESSION
    return {
        "bar": bar, "sigma_seed": float(sigma_seed), "delta": round(delta, 4),
        "fold_deltas": {k: round(v, 4) for k, v in fold_deltas.items()},
        "worst_fold_delta": round(worst, 4),
        "comparable": bool(comparable), "missing_folds": missing_folds,
        "clears_delta": bool(clears), "no_fold_regression": bool(no_regression),
        "ships": bool(comparable and clears and no_regression),
    }


def paired_delta(report_a: dict, report_b: dict, sigma_seed: float) -> dict:
    """Paired A/B pooled-metric delta with per-fold detail (spec §4 SP1).

    Delta is B minus A (negative = B better). ``B_not_worse_than_A_by_sigma``
    is the refit deployment-recipe gate: B ships as the recipe iff its pooled
    winner log-loss is not worse than A's by more than ``sigma_seed``. Unlike
    `bar_check`, this has no "ships as a challenger" bar or comparability
    check -- it is the shared arithmetic behind both the main A/B refit
    comparison and a fresh-seed re-score of the same pair."""
    a_pooled, b_pooled = report_a["pooled"], report_b["pooled"]
    delta = round(b_pooled["winner_log_loss"] - a_pooled["winner_log_loss"], 4)
    a_folds, b_folds = report_a["folds"], report_b["folds"]
    common_years = sorted(set(a_folds) & set(b_folds), key=int)
    fold_deltas = {
        year: round(b_folds[year]["winner_log_loss"] - a_folds[year]["winner_log_loss"], 4)
        for year in common_years
    }
    return {
        "A_pooled": a_pooled,
        "B_pooled": b_pooled,
        "delta_B_minus_A": delta,
        "sigma_seed": sigma_seed,
        "B_not_worse_than_A_by_sigma": bool(delta <= sigma_seed),
        "per_fold_B_minus_A": fold_deltas,
    }
