"""Shared evaluation metrics for binary win probability predictions, plus
calibration (expected calibration error) and joint-outcome (winner x method
x round) log-loss metrics."""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def log_loss(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p >= 0.5) == (y == 1.0)))


def brier_score(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p - y) ** 2))


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over the union of true and predicted labels."""
    true = list(y_true)
    pred = list(y_pred)
    labels = sorted(set(true) | set(pred))
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def expected_calibration_error(y_true, p_pred, n_bins: int = 10) -> float:
    """Count-weighted mean |mean prediction − empirical rate| over equal-width bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def reliability_curve(y_true, p_pred, n_bins: int = 10) -> list[dict]:
    """Per-bin count, mean prediction and empirical rate, in `expected_calibration_error`'s bins.

    Same equal-width edges and the same `np.digitize` placement, so
    `sum(row["weight"] * abs(row["gap"]) for row in reliability_curve(...))`
    reconstructs `expected_calibration_error(...)` exactly. That is the point:
    ECE is one number summarising this table, and a gate that compares two ECEs
    is really comparing two of these curves. `gap` is mean_pred - empirical_rate
    (positive = over-confident in the favourite's direction).

    Empty bins are reported with `n` 0 and null statistics rather than dropped,
    so two curves over different predictions stay row-comparable, and so a
    "miscalibration" carried by one nearly-empty bin is visible as such.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if len(y) != len(p):
        raise ValueError(f"y_true and p_pred must be the same length ({len(y)} vs {len(p)})")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        row = {"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
               "weight": float(mask.mean()) if len(y) else 0.0,
               "mean_pred": None, "empirical_rate": None, "gap": None}
        if n:
            row["mean_pred"] = float(p[mask].mean())
            row["empirical_rate"] = float(y[mask].mean())
            row["gap"] = float(p[mask].mean() - y[mask].mean())
        rows.append(row)
    return rows


def joint_outcome_log_loss(
    y_winner, y_method, y_round, p_winner, method_probs, round_probs,
    method_classes, round_classes,
) -> float:
    """Log-loss of the realised (winner, method, round) outcome under the
    composed distribution P(w)·P(m)·P(r | finish). Decisions have no round
    term. Rows with unknown method are skipped; a finish with unknown round
    is skipped too (it cannot be scored)."""
    y_w = np.asarray(y_winner, dtype=float)
    p_w = np.clip(np.asarray(p_winner, dtype=float), _EPS, 1 - _EPS)
    method_probs = np.asarray(method_probs, dtype=float)
    round_probs = np.asarray(round_probs, dtype=float)
    m_index = {label: i for i, label in enumerate(method_classes)}
    r_index = {label: i for i, label in enumerate(round_classes)}
    losses = []
    for i, (m, r) in enumerate(zip(y_method, y_round)):
        if m is None or m not in m_index:
            continue
        p = p_w[i] if y_w[i] == 1.0 else 1.0 - p_w[i]
        p *= method_probs[i, m_index[m]]
        if m != "decision":
            if r is None or r not in r_index:
                continue
            p *= round_probs[i, r_index[r]]
        losses.append(-np.log(max(p, _EPS)))
    return float(np.mean(losses)) if losses else float("nan")


# --------------------------------------------------------------------------
# joint outcome cells (SP3)
# --------------------------------------------------------------------------
# The simulator emits one distribution over the whole outcome space rather
# than three marginals, so it is scored by reading the realised cell's
# probability straight out of that distribution. The cell set is exactly the
# one the composed path implies -- (winner) x (finishing method) x (round),
# plus (winner) x decision -- so `joint_cell_log_loss` and
# `joint_outcome_log_loss` are both `-log P(realised cell)` over the same
# cells and are directly comparable. `tests/test_evaluate.py` pins that: a
# joint built by multiplying independent marginals scores exactly what the
# composed path scores for those marginals.
#
# Flat layout, given `method_classes` whose LAST entry is "decision" and
# `round_classes`:
#
#     index = winner * n_finish_methods * n_rounds + method * n_rounds + round
#     decision index = 2 * n_finish_methods * n_rounds + winner
#
# winner 0 is the feature table's corner A (`y_winner == 1`), winner 1 its
# corner B. With the harness's three methods and four round classes that is
# 16 finish cells followed by 2 decision cells, 18 in all.

DECISION = "decision"


def n_joint_cells(method_classes, round_classes) -> int:
    return 2 * (len(method_classes) - 1) * len(round_classes) + 2


def joint_cell_index(winner: int, method_index, round_index, method_classes, round_classes) -> int:
    """Flat index of one outcome cell. `method_index`/`round_index` are None
    for a decision, which has no round term."""
    n_rounds = len(round_classes)
    n_methods = len(method_classes) - 1
    if method_index is None or round_index is None:
        return 2 * n_methods * n_rounds + int(winner)
    return int(winner) * n_methods * n_rounds + int(method_index) * n_rounds + int(round_index)


def _realised_cells(y_winner, y_method, y_round, method_classes, round_classes):
    """(row, cell index) for every row whose realised cell is scorable.

    Skips exactly what `joint_outcome_log_loss` skips: an unknown method, and
    a finish with an unknown round. Both scorers walk this, so they always
    average over the same rows."""
    y_w = np.asarray(y_winner, dtype=float)
    m_index = {label: i for i, label in enumerate(method_classes) if label != DECISION}
    r_index = {label: i for i, label in enumerate(round_classes)}
    out = []
    for i, (m, r) in enumerate(zip(y_method, y_round)):
        winner = 0 if y_w[i] == 1.0 else 1
        if m == DECISION:
            out.append((i, joint_cell_index(winner, None, None, method_classes, round_classes)))
            continue
        if m is None or m not in m_index:
            continue
        if r is None or r not in r_index:
            continue
        out.append((i, joint_cell_index(winner, m_index[m], r_index[r], method_classes, round_classes)))
    return out


def joint_cell_log_loss(
    y_winner, y_method, y_round, cells, method_classes, round_classes,
) -> float:
    """Mean `-log P(realised cell)` read directly from a joint distribution.

    `cells` is `(n, n_joint_cells(...))` in the layout above. Zero mass is
    floored at `_EPS` exactly as the composed path floors its product, so a
    cell the simulator never landed in scores badly rather than infinitely --
    though the simulator's Laplace smoothing means it should never be zero."""
    cells = np.asarray(cells, dtype=float)
    expected = n_joint_cells(method_classes, round_classes)
    if cells.ndim != 2 or cells.shape[1] != expected:
        raise ValueError(f"cells must have shape (n, {expected}); got {cells.shape}")
    losses = [
        -np.log(max(cells[i, cell], _EPS))
        for i, cell in _realised_cells(y_winner, y_method, y_round, method_classes, round_classes)
    ]
    return float(np.mean(losses)) if losses else float("nan")


def realised_zero_mass_fraction(
    y_winner, y_method, y_round, zero_mass, method_classes, round_classes,
) -> float:
    """Fraction of scorable rows whose realised cell had zero raw simulated mass.

    The plan's pre-registered Monte Carlo adequacy check: more than ~1% means
    `n_runs` is too small for the cell set. `zero_mass` is a boolean array in
    the same layout as `cells`, carrying the raw (pre-smoothing) emptiness of
    each cell."""
    zero_mass = np.asarray(zero_mass, dtype=bool)
    rows = _realised_cells(y_winner, y_method, y_round, method_classes, round_classes)
    if not rows:
        return float("nan")
    return float(np.mean([bool(zero_mass[i, cell]) for i, cell in rows]))
