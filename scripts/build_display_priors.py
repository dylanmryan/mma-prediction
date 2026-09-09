"""Build mean-matching display correction factors -> models/torch/display_priors.json.

The method/round heads are trained with class-weighted loss, so their raw
softmax outputs overstate rare classes. For each class c we compute

    factor(c) = empirical_prior(c) / mean_model_predicted(c)

over the rows the DEPLOYED model was trained on (mma.inference.
deployed_training_mask, read from models/torch/metrics_val.json): every
decisive fight through `train_through` under the refit_through recipe that
ships since SP1, or date < TRAIN_END for a split-protocol model. Method
factors come from rows with a known method, round factors from finishes
only, computed separately for 3-round and 5-round fights. Multiplying a
fight's predicted distribution by these factors and renormalizing
(mma.inference.apply_prior_correction) maps the model's aggregate
predictions onto the empirical base rates while preserving per-fight
relative signal.

`mean_model_predicted` is the DEPLOYED SCORER's mean prediction, which since
SP2.2 is the blend's (`mma.inference.BlendedPredictor`), not the torch
ensemble's. The two members' method/round heads are differently biased -- the
net's are class-weighted and the trees' are not -- so factors fitted against
the net alone would not mean-match what the app displays.

The factors are a property of the committed scorer on its own training rows,
so they must be rebuilt whenever EITHER member is retrained (the weekly
refresh and scripts/roll_window.py both do). They are NOT part of
`mma.versioning.model_version`: they are applied after the probability the
track record stores, so rebuilding them does not open a new track-record
section.

The output JSON is committed so the Streamlit app never has to load
features.parquet or run ensemble predictions at startup.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mma.inference import (
    BlendedPredictor,
    apply_prior_correction,
    compute_correction_factors,
    compute_display_priors,
    deployed_training_mask,
    load_deployed_metrics,
)
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "models" / "torch" / "display_priors.json"
XGB_METRICS = ROOT / "models" / "xgb_metrics_val.json"


def check_members_agree(torch_metrics: dict, xgb_metrics: dict) -> str | None:
    """Warning text when the blend's two members were refit through different
    dates, else None.

    The correction rows are `deployed_training_mask`, read from the TORCH
    member's metrics file. If the XGBoost member was refit through a different
    date then that mask is not both members' training set and the factors are
    being fitted partly out of sample for one of them -- which happens if a
    refresh runs one trainer and not the other.
    """
    if torch_metrics.get("mode") != "refit_through":
        return None
    if xgb_metrics.get("mode") != "refit_through":
        return None
    if torch_metrics.get("train_through") == xgb_metrics.get("train_through"):
        return None
    return (f"WARNING: the blend's members were refit through different dates "
            f"(torch {torch_metrics.get('train_through')}, xgb "
            f"{xgb_metrics.get('train_through')}); re-run both trainers")


def fit_factors(empirical: dict, raw_probs: np.ndarray, classes: list[str],
                max_iter: int = 100, tol: float = 1e-6) -> dict:
    """Fixed-point refinement of mean-matching factors.

    A single ratio empirical/mean_predicted matches the aggregate *before*
    per-row renormalization; renormalization then pulls the mean corrected
    distribution slightly off the base rates (up to ~1.6pp for round 45).
    So we iterate: recompute the mean of the actually-displayed (multiplied
    AND renormalized) distribution and rescale the factors by
    empirical/mean_corrected until the displayed aggregate matches the
    empirical prior to within `tol`.
    """
    mean_raw = dict(zip(classes, raw_probs.mean(axis=0)))
    factors = compute_correction_factors(empirical, mean_raw)
    for _ in range(max_iter):
        corrected = np.array([
            [d[c] for c in classes]
            for d in (
                apply_prior_correction(dict(zip(classes, row)), factors)
                for row in raw_probs
            )
        ])
        mean_corrected = dict(zip(classes, corrected.mean(axis=0)))
        update = compute_correction_factors(empirical, mean_corrected)
        factors = {c: factors[c] * update[c] for c in classes}
        if max(abs(mean_corrected[c] - empirical[c]) for c in classes) < tol:
            break
    return factors


def main() -> None:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    # The deployed SCORER, both members: the displayed method/round splits are
    # the blend's, so the factors have to be fitted against the blend's own
    # mean prediction.
    predictor = BlendedPredictor.load()
    metrics = load_deployed_metrics()
    priors = compute_display_priors(features, metrics)

    warning = check_members_agree(metrics, json.loads(XGB_METRICS.read_text())
                                  if XGB_METRICS.exists() else {})
    if warning:
        print(warning, file=sys.stderr)

    mask = deployed_training_mask(features, metrics)
    train = features[mask]
    print(f"deployed model mode={metrics.get('mode', 'split')}: correction rows = "
          f"{int(mask.sum())} of {len(features)} (through {train['date'].max().date()})")

    # Method: train rows with a known method.
    method_rows = train[train["y_method"].notna()]
    method_raw = predictor.predict(method_rows)["method_probs"]
    factors = {"method": fit_factors(priors["method"], method_raw, METHOD_CLASSES)}

    # Round: finishes only, split by scheduled length (same fillna(3) default
    # as the model's three-round mask).
    finishes = train[train["y_finish_round"].notna()]
    sched = finishes["scheduled_rounds"].fillna(3)
    for key, subset in (
        ("round_3", finishes[sched <= 3]),
        ("round_5", finishes[sched == 5]),
    ):
        round_raw = predictor.predict(subset)["round_probs"]
        factors[key] = fit_factors(priors[key], round_raw, ROUND_CLASSES)

    OUT.write_text(json.dumps(factors, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(factors, indent=2))


if __name__ == "__main__":
    main()
