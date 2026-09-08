"""The blend's arithmetic, shared by the harness candidate and the served model.

SP2.2 ships a blend, so exactly one thing must be true: the scorer that serves
a prediction has to compute the same number the walk-forward harness measured.
The three operations that turn two members' outputs into one prediction --
the weighted average, the round-45 mask, and the post-average temperature --
therefore live here, and both `mma.candidates.BlendCandidate` (which measured
B1) and `mma.inference.BlendedPredictor` (which serves it) import them rather
than each carrying a copy that could drift.

Nothing here fits anything. Fitting the temperature is the harness's job
(`mma.models.train_loop.fit_temperature`, on the fold's inner-validation year);
deployment has no held-out year and applies a fixed value derived from those
per-fold fits, exactly as the refit recipe does for the torch member's own
temperature.
"""
from __future__ import annotations

import numpy as np

HEADS = ("winner", "method", "round")
# Guards log(0) in `logit`. Kept at the value BlendCandidate used when B1 was
# scored, because a different epsilon is a different (if barely) prediction.
LOGIT_EPS = 1e-9


def logit(p) -> np.ndarray:
    """log(p / (1 - p)), clipped away from 0 and 1 by `LOGIT_EPS`."""
    q = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(q / (1.0 - q))


def apply_temperature(p, temperature: float) -> np.ndarray:
    """Temperature-scale a probability: sigmoid(logit(p) / T).

    The blend's calibration step. It is applied AFTER averaging, on the
    averaged probability, because averaging two differently-calibrated
    probability streams is not itself calibrated -- SP2.1's uncalibrated blend
    measured ECE 0.0178 against the deployed model's 0.0088, and SP2.2's
    post-average temperature roughly halves it.
    """
    return 1.0 / (1.0 + np.exp(-logit(p) / float(temperature)))


def blend_heads(xgb_pred: dict, torch_pred: dict, weight: float) -> dict:
    """`weight` * the XGB member + (1 - weight) * the torch member, per head.

    `weight` is on the XGB member and is FIXED at 0.5 by SP2.2's
    pre-registration; it is never fitted. The model-v2 session (2026-07-15)
    found fitted stacking weights lose to a plain average, and fitting them on
    the same folds a candidate is judged on is the selection failure this
    project has been burned by before. The 0.3/0.7 cells in
    models/walkforward/blend_b0_w0{3,7}.json exist only to show the surface is
    flat near 0.5.
    """
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"blend weight must be in [0, 1] (got {weight!r})")
    return {
        head: weight * np.asarray(xgb_pred[head], dtype=float)
        + (1.0 - weight) * np.asarray(torch_pred[head], dtype=float)
        for head in HEADS
    }


def mask_round_45(probs: np.ndarray, three_round) -> np.ndarray:
    """Zero the '45' column for three-round fights and renormalise those rows.

    `MultiTaskNet.round_probs` does this inside the torch member (by masking
    the logit before the softmax) and the XGB member does not do it at all, so
    an average of the two puts mass on a round that cannot happen. Rows whose
    45 column is already exactly zero are left completely alone -- renormalising
    a row that already sums to one still perturbs its last bits, and a blend at
    weight 0.0 has to be bit-identical to the torch member.
    """
    out = np.array(probs, dtype=float, copy=True)
    three_round = np.asarray(three_round, dtype=bool)
    changed = three_round & (out[:, 3] != 0.0)
    out[three_round, 3] = 0.0
    if changed.any():
        out[changed] = out[changed] / out[changed].sum(axis=1, keepdims=True)
    return out
