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
* ``temperature`` is recomputed from the report's per-fold fitted
  temperatures (``fit_info.temperature``) by the same median rule
  `scripts/run_walkforward.fixed_budget_from` uses for the torch member's own
  deployment temperature under the refit recipe -- deployment has no
  held-out year to fit a temperature on, so it applies a fixed value derived
  from the harness's per-fold fits instead (see `scripts/refit_decision.py`'s
  "b1" report-set note).

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

from scripts.run_walkforward import fixed_budget_from  # noqa: E402

BLEND_REPORT = ROOT / "models" / "walkforward" / "blend_b1.json"
OUT = ROOT / "models" / "blend.json"

DERIVATION = (
    "weight is the pre-registered 0.5, read from the source report's own "
    "config.blend_weight (mma.blend.blend_heads: fixed by SP2.2's "
    "pre-registration, never fitted); temperature is the median of the "
    "source report's per-fold fitted temperatures (fit_info.temperature), "
    "recomputed via scripts.run_walkforward.fixed_budget_from's median rule "
    "-- the same rule the refit recipe uses to derive the torch member's own "
    "deployment temperature, because deployment has no held-out year to fit "
    "one on directly."
)


def build_config(report_path: Path = BLEND_REPORT, root: Path = ROOT) -> dict:
    """The blend.json payload derived from ``report_path``, pure apart from
    reading the report, so a caller can diff it against the committed file."""
    report = json.loads(report_path.read_text())
    weight = float(report["config"]["blend_weight"])
    temperature = fixed_budget_from(report)["temperature"]
    return {
        "weight": weight,
        "temperature": temperature,
        "derivation": DERIVATION,
        "source_report": str(report_path.resolve().relative_to(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=BLEND_REPORT,
                        help="walk-forward report to derive the config from")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output path (default: models/blend.json)")
    args = parser.parse_args()

    config = build_config(args.report)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
