"""Measure the deployed scorer's method/round marginals against base rates.

**What this replaced, and why.** Until SP3 this script fitted mean-matching
correction factors -- ``factor(c) = empirical_prior(c) / mean_model_predicted(c)``
-- and ``app.py`` multiplied every displayed method and round probability by
them. That correction existed because the class-weighted multi-task heads were
badly miscalibrated in aggregate: on their own training rows the blend
predicted P(rounds 4-5) for five-round fights at 2.5x the empirical rate and
P(round 3) for three-round fights at 1.7x, so the raw numbers could not be
shown.

SP3 replaced those heads with a Monte Carlo simulator whose marginals are read
off one coherent joint distribution, and the SP3 plan required measuring
whether the correction was still needed rather than assuming either way. It is
not. On the deployed training rows the simulator's aggregate marginals land
within a few points of the base rates by themselves -- at most 4 points on any
displayed class, against 25 for the blend's heads (the numbers this script
writes; see the committed artifact for the current ones).

So the correction is retired, on two grounds:

* it is not needed -- the factors it would fit are all near 1;
* applying it would now do harm. The displayed method and round splits are
  MARGINALS OF A JOINT: mean-matching each one separately and renormalising
  moves them off the joint the outcome table is drawn from, so the table's
  rows would stop summing to the method row above them. That contradiction
  between displayed numbers is exactly what the simulator was built to remove.

What remains is this measurement. It runs weekly beside the retrains, so a
retrain that quietly breaks the simulator's aggregate calibration shows up in
a committed diff rather than only in a fight nobody checked.
``models/display_calibration.json`` is NOT part of
``mma.versioning.model_version``: it records a property of the model, it does
not change a prediction.
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
    SimulatorPredictor,
    compute_correction_factors,
    compute_display_priors,
    deployed_training_mask,
    load_deployed_metrics,
)
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "models" / "display_calibration.json"
XGB_METRICS = ROOT / "models" / "xgb_metrics_val.json"
#: How far the deployed scorer's aggregate may drift from the base rates
#: before this script says so out loud, in PROBABILITY POINTS rather than as a
#: ratio: a rare class makes a ratio look alarming over a gap too small to see
#: in a displayed percentage (rounds 4-5 of a five-round fight sit near 0.16,
#: so a 4-point miss reads as a factor of 1.3). The retired correction was
#: fitted against gaps of 12 and 25 points, which is what 5 is set to catch.
TOLERANCE = 0.05


def check_members_agree(torch_metrics: dict, xgb_metrics: dict) -> str | None:
    """Warning text when the blend's two members were refit through different
    dates, else None.

    The measurement rows are `deployed_training_mask`, read from the TORCH
    member's metrics file. If the XGBoost member was refit through a different
    date then that mask is not both members' training set and the comparison
    is partly out of sample for one of them -- which happens if a refresh runs
    one trainer and not the other.
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


def compare(empirical: dict, predicted: np.ndarray, classes: list) -> dict:
    """One head's aggregate calibration: base rate, mean prediction, and the
    mean-matching factor that would be needed to reconcile them (1.0 = the
    model already matches the base rate in aggregate)."""
    mean_predicted = dict(zip(classes, predicted.mean(axis=0)))
    factors = compute_correction_factors(empirical, mean_predicted)
    return {
        "empirical": {cls: round(float(empirical[cls]), 4) for cls in classes},
        "mean_predicted": {cls: round(float(mean_predicted[cls]), 4) for cls in classes},
        "factor_needed": {cls: round(float(factors[cls]), 3) for cls in classes},
    }


def max_deviation(comparison: dict) -> float:
    """The largest gap, in probability points, between the mean prediction and
    the base rate over all classes.

    Every class counts, including one with a zero base rate: a model putting
    mass on rounds 4-5 of a three-round fight would be a real failure, and it
    shows up here as the full size of that mass."""
    return max(
        (abs(comparison["mean_predicted"][cls] - comparison["empirical"][cls])
         for cls in comparison["empirical"]),
        default=0.0,
    )


def measure(features: pd.DataFrame, hybrid, blend, metrics: dict) -> dict:
    """The whole comparison: the deployed hybrid against the base rates, with
    the retired blend heads' numbers beside it as the contrast that explains
    why the correction is gone."""
    priors = compute_display_priors(features, metrics)
    mask = deployed_training_mask(features, metrics)
    train = features[mask]

    method_rows = train[train["y_method"].notna()]
    finishes = train[train["y_finish_round"].notna()]
    sched = finishes["scheduled_rounds"].fillna(3)
    subsets = {
        "method": (method_rows, METHOD_CLASSES, "method_probs"),
        "round_3": (finishes[sched <= 3], ROUND_CLASSES, "round_probs"),
        "round_5": (finishes[sched == 5], ROUND_CLASSES, "round_probs"),
    }

    out = {}
    for key, (subset, classes, head) in subsets.items():
        print(f"  {key}: {len(subset)} rows")
        out[key] = {
            "n": int(len(subset)),
            "deployed_simulator": compare(priors[key], hybrid.predict(subset)[head], classes),
            "retired_blend_heads": compare(priors[key], blend.predict(subset)[head], classes),
        }
        out[key]["max_deviation_points"] = {
            "deployed_simulator": round(max_deviation(out[key]["deployed_simulator"]), 4),
            "retired_blend_heads": round(max_deviation(out[key]["retired_blend_heads"]), 4),
        }

    worst = max(block["max_deviation_points"]["deployed_simulator"]
                for block in out.values())
    return {
        "correction_applied": False,
        "reason": (
            "The displayed method and round splits are marginals of ONE joint "
            "distribution (mma.inference.SimulatorPredictor). Mean-matching them "
            "separately is unnecessary -- the factors below are near 1 -- and would "
            "move each marginal off the joint the app's outcome table is drawn from, "
            "reintroducing the contradiction between displayed numbers that SP3 "
            "removed. See this script's docstring."
        ),
        "tolerance": TOLERANCE,
        "within_tolerance": bool(worst <= TOLERANCE),
        "max_deviation_points": round(float(worst), 4),
        "measured_on": {
            "rows": int(mask.sum()),
            "of": int(len(features)),
            "train_through": str(train["date"].max().date()),
            "mode": metrics.get("mode", "split"),
        },
        **out,
    }


def main() -> None:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    metrics = load_deployed_metrics()
    warning = check_members_agree(
        metrics, json.loads(XGB_METRICS.read_text()) if XGB_METRICS.exists() else {})
    if warning:
        print(warning, file=sys.stderr)

    blend = BlendedPredictor.load()
    hybrid = SimulatorPredictor.load(blend=blend)
    print(f"measuring the deployed scorer against base rates "
          f"(mode={metrics.get('mode', 'split')}):")
    report = measure(features, hybrid, blend, metrics)

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT}")
    for key in ("method", "round_3", "round_5"):
        block = report[key]
        print(f"\n{key} (n={block['n']}): "
              f"simulator off the base rates by at most "
              f"{block['max_deviation_points']['deployed_simulator']:.3f}, "
              f"the retired blend heads by "
              f"{block['max_deviation_points']['retired_blend_heads']:.3f}")
        for cls in block["deployed_simulator"]["empirical"]:
            print(f"  {cls:<12} base {block['deployed_simulator']['empirical'][cls]:.4f}"
                  f"  simulator {block['deployed_simulator']['mean_predicted'][cls]:.4f}"
                  f"  (factor {block['deployed_simulator']['factor_needed'][cls]:.2f})")
    if not report["within_tolerance"]:
        print(f"\nWARNING: the deployed simulator's aggregate is off the base rates by "
              f"{report['max_deviation_points']:.3f}, past the {TOLERANCE} tolerance; "
              "investigate before trusting the displayed splits.", file=sys.stderr)


if __name__ == "__main__":
    main()
