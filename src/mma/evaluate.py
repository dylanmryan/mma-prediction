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
