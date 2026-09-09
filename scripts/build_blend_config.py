"""Build models/blend.json: the deployed blend's own configuration, committed.

The deployed scorer (`mma.inference.BlendedPredictor`) is parameterised by two
numbers that are not learned by either member's training script -- the
mixing weight between the XGBoost and torch members, and the post-average
temperature -- so neither `scripts/train_xgb.py` nor `scripts/train_torch.py`
can write them: each trainer knows only its own member. Before this script
existed, both numbers lived as module constants in `src/mma/inference.py`,
which `mma.versioning.MODEL_ARTIFACT_GLOBS` never hashed -- editing either
one silently changed every recorded probability while leaving `model_version`
byte-identical, exactly the failure the artifact-hash design replaced git-sha
stamping to prevent (see `src/mma/versioning.py`'s docstring).

This script makes the configuration a committed artifact instead, derived
from the walk-forward report the deployment decision shipped
(`models/walkforward/blend_b1.json`) rather than hand-typed, so it cannot
drift from the report it claims to come from:

* ``weight`` is read from the report's own ``config.blend_weight`` --
  pre-registered at 0.5 by SP2.2 and never fitted (`mma.blend.blend_heads`).
  Reading it from the report rather than hardcoding 0.5 means a report scored
  at a different weight would change this file, not silently mismatch it.
* ``temperature`` is the WALK-FORWARD temperature derived by
  `scripts/derive_blend_temperature.py` from the report and its committed
  per-row prediction dump: for fold Y, one temperature fitted on the pooled
  out-of-fold predictions of every fold before Y, and at serving time -- where
  every fold is "before" -- one fitted on all of them. Deployment has no
  held-out year, so it must apply a fixed value; this is the only rule for
  choosing that value that is validated without using the evaluation rows to
  choose it (`models/walkforward/blend_temperature.json` carries the
  comparison against the alternatives).

  It replaces the MEDIAN of the per-fold fits (0.80), which this file carried
  until 2026-09-09. That rule came from `scripts/run_walkforward.
  fixed_budget_from`, where a median selects a training BUDGET; a median
  temperature has no calibration justification, and re-scoring the dump under
  it measured pooled ECE 0.0177 against the walk-forward rule's 0.0108.

Usage:
    python scripts/build_blend_config.py
    python scripts/build_blend_config.py --report models/walkforward/blend_b1.json --out models/blend.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.derive_blend_temperature import (  # noqa: E402
    deployed_temperature, invert_per_fold, load_inputs, verify_round_trip,
)

BLEND_REPORT = ROOT / "models" / "walkforward" / "blend_b1.json"
BLEND_PREDICTIONS = ROOT / "models" / "walkforward" / "preds" / "blend_b1.json"
OUT = ROOT / "models" / "blend.json"

DERIVATION = (
    "weight is the pre-registered 0.5, read from the source report's own "
    "config.blend_weight (mma.blend.blend_heads: fixed by SP2.2's "
    "pre-registration, never fitted); temperature is the WALK-FORWARD "
    "temperature from scripts.derive_blend_temperature -- fitted by "
    "mma.models.train_loop.fit_temperature on the pooled out-of-fold "
    "pre-temperature predictions of every fold before the one being served, "
    "which at serving time is all of them, because deployment has no held-out "
    "year to fit one on directly. It supersedes the median of the per-fold "
    "fits (0.80), a rule borrowed from training-budget derivation that no "
    "harness run validated as a calibration rule; see "
    "models/walkforward/blend_temperature.json for the rule comparison and "
    "the deployed form's own pooled log-loss and ECE."
)


def build_config(report_path: Path = BLEND_REPORT,
                 predictions_path: Path = BLEND_PREDICTIONS,
                 root: Path = ROOT) -> dict:
    """The blend.json payload derived from ``report_path`` and its per-row
    prediction dump, pure apart from reading them, so a caller can diff it
    against the committed file.

    The dump is not optional: the temperature is derived by re-scoring the
    dumped predictions, and `verify_round_trip` refuses to derive anything if
    the dump does not reproduce the report's own pooled metrics.
    """
    report = json.loads(report_path.read_text())
    weight = float(report["config"]["blend_weight"])
    inputs = load_inputs(report_path, predictions_path)
    verify_round_trip(inputs)
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    temperature = deployed_temperature(pre, inputs["y"])
    return {
        "weight": weight,
        "temperature": temperature,
        "derivation": DERIVATION,
        "source_report": str(report_path.resolve().relative_to(root)),
        "source_predictions": str(predictions_path.resolve().relative_to(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=BLEND_REPORT,
                        help="walk-forward report to derive the config from")
    parser.add_argument("--predictions", type=Path, default=BLEND_PREDICTIONS,
                        help="that report's per-row prediction dump, which the "
                             "temperature is derived by re-scoring")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output path (default: models/blend.json)")
    args = parser.parse_args()

    config = build_config(args.report, args.predictions)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
