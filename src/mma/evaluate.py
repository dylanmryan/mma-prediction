"""Shared evaluation metrics for binary win probability predictions."""
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
