"""Build mean-matching display correction factors -> models/torch/display_priors.json.

The method/round heads are trained with class-weighted loss, so their raw
softmax outputs overstate rare classes. For each class c we compute

    factor(c) = empirical_prior(c) / mean_model_predicted(c)

over TRAINING-split rows only (date < TRAIN_END): method factors from rows
with a known method, round factors from finishes only, computed separately
for 3-round and 5-round fights. Multiplying a fight's predicted distribution
by these factors and renormalizing (mma.inference.apply_prior_correction)
maps the model's aggregate predictions onto the empirical base rates while
preserving per-fight relative signal.

The output JSON is committed so the Streamlit app never has to load
features.parquet or run ensemble predictions at startup.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mma.inference import (
    TRAIN_END,
    Ensemble,
    apply_prior_correction,
    compute_correction_factors,
    compute_display_priors,
)
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "models" / "torch" / "display_priors.json"


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
    ensemble = Ensemble.load()
    priors = compute_display_priors(features)

    train = features[features["date"] < TRAIN_END]

    # Method: train rows with a known method.
    method_rows = train[train["y_method"].notna()]
    method_raw = ensemble.predict(method_rows)["method_probs"]
    factors = {"method": fit_factors(priors["method"], method_raw, METHOD_CLASSES)}

    # Round: finishes only, split by scheduled length (same fillna(3) default
    # as the model's three-round mask).
    finishes = train[train["y_finish_round"].notna()]
    sched = finishes["scheduled_rounds"].fillna(3)
    for key, subset in (
        ("round_3", finishes[sched <= 3]),
        ("round_5", finishes[sched == 5]),
    ):
        round_raw = ensemble.predict(subset)["round_probs"]
        factors[key] = fit_factors(priors[key], round_raw, ROUND_CLASSES)

    OUT.write_text(json.dumps(factors, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(factors, indent=2))


if __name__ == "__main__":
    main()
