"""Measure how far the static external snapshot has decayed, and say so.

    python scripts/check_snapshot_coverage.py            # writes models/snapshot_coverage.json
    python scripts/check_snapshot_coverage.py --print    # ... and echoes it

**What this is for.** The `external`, `notice` and `context` blocks all read
one static source -- `ehan03/jds-mma-data`, UFC coverage ending 2024-12-14,
last upstream commit December 2025. A fighter who debuts after that date is
unmatched by construction, so the share of rows the snapshot has never seen
does not merely stop improving, it GROWS with every event. That is a decay,
and a decay with no alarm on it rots silently: the block's shipping number was
measured on folds whose coverage we will never have again, and nothing in the
weekly pipeline would notice the world drifting away from it.

This runs weekly beside `scripts/check_display_calibration.py`, in the same
shape and for the same reason: measurement, not correction. It changes no
prediction and gates nothing (exit 0 always, `continue-on-error` in the
Action). What it does is put the number in a committed diff and, past the
threshold, on stderr.

**The threshold is 0.40, and it is a DECISION-INVALIDATION threshold rather
than a data-quality one.** The 2026-09-09 decay decision
(`docs/superpowers/plans/2026-09-09-external-decay-decision.md`,
`models/walkforward/external_decay_decision.json`) turned on the recent-fold
behaviour of the snapshot-dependent columns, and the most recent fold it could
measure -- 2025 -- sits at 0.359 `external_missing`. A trailing-12-month rate
above 0.40 means the rows now being served are LESS covered than the worst
fold any of those numbers came from, so the measurement the decision rests on
has stopped describing the population it was about, and the decision is due to
be re-taken rather than merely re-read. Set below 0.359 the check would fire
on the state the decision already accounts for; set far above it, it would
fire only once the block had been quietly wrong for years.

The arithmetic lives in `mma.decay` (`coverage_by_year`, `trailing_coverage`)
and is unit-tested there; this file is the CLI, the artifact and the warning.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.decay import coverage_by_year, trailing_coverage  # noqa: E402

FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "models" / "snapshot_coverage.json"
#: The fight-level either-corner-unmapped flag, kept in the table (and out of
#: every model matrix) precisely so this stays measurable.
FLAG = "external_missing"
#: See the module docstring: the worst fold the decay decision was measured on
#: is 2025 at 0.359, and above 0.40 the served rows are less covered than that.
THRESHOLD = 0.40
#: Trailing window. Twelve months is one full event calendar, so the number is
#: not a seasonal artifact of which divisions happened to fight recently.
WINDOW_MONTHS = 12


def measure(features: pd.DataFrame, *, threshold: float = THRESHOLD,
            months: int = WINDOW_MONTHS, as_of=None) -> dict:
    """The coverage report for a feature table.

    Pure but for the table it is handed, so the artifact is a function of the
    committed data and re-running reproduces it exactly. `as_of` defaults to
    the table's own last date rather than to today, for the same reason -- see
    `mma.decay.trailing_coverage`.
    """
    if FLAG not in features.columns:
        raise SystemExit(
            f"{FLAG!r} is not in the feature table, so snapshot coverage cannot be "
            "measured. It is kept in the table (and out of both model matrices) for "
            "exactly this reason -- see mma.tensors.DROPPED."
        )
    dates, missing = features["date"], features[FLAG]
    trailing = trailing_coverage(dates, missing, months=months, as_of=as_of)
    breached = bool(trailing["n"] and trailing["share"] > threshold)
    return {
        "flag": FLAG,
        "threshold": threshold,
        "window_months": months,
        "trailing": trailing,
        "breached": breached,
        "by_year": coverage_by_year(dates, missing),
        "overall": {
            "n": int(len(features)),
            "missing": int(missing.astype(bool).sum()),
            "share": round(float(missing.astype(bool).mean()), 4),
        },
        "source": {
            "snapshot": "ehan03/jds-mma-data",
            "commit": "ec77f53737829c51fbf15ba140b46d36dee65d22",
            "ufc_coverage_end": "2024-12-14",
            "static": True,
        },
        "why": (
            "The snapshot is static, so this share grows with every event. Past "
            "the threshold the rows being served are less covered than the 2025 "
            "fold the decay decision was measured on, and that decision "
            "(docs/superpowers/plans/2026-09-09-external-decay-decision.md) is "
            "due to be re-taken."
        ),
    }


def warning_text(report: dict) -> str | None:
    """The stderr warning for a breached report, else None."""
    if not report["breached"]:
        return None
    trailing = report["trailing"]
    return (
        f"WARNING: the external snapshot has not seen at least one corner of "
        f"{trailing['share']:.3f} of the {trailing['n']} fights in the trailing "
        f"{trailing['months']} months ({trailing['window_start']} .. "
        f"{trailing['window_end']}), past the {report['threshold']} threshold. "
        "The snapshot-dependent columns were last judged against a 2025 fold at "
        "0.359 missing, so that judgement no longer describes the rows being "
        "served: re-take the decay decision "
        "(docs/superpowers/plans/2026-09-09-external-decay-decision.md) or "
        "refresh the source (scripts/build_external.py)."
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--months", type=int, default=WINDOW_MONTHS)
    parser.add_argument("--as-of", default=None,
                        help="end of the trailing window (default: the table's last date)")
    parser.add_argument("--print", dest="echo", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report = measure(pd.read_parquet(args.features), threshold=args.threshold,
                     months=args.months, as_of=args.as_of)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    trailing = report["trailing"]
    print(f"external snapshot coverage ({report['source']['snapshot']}, "
          f"UFC coverage to {report['source']['ufc_coverage_end']}):")
    print(f"  overall           {report['overall']['share']:.4f} missing "
          f"of {report['overall']['n']} rows")
    print(f"  trailing {trailing['months']:>2}m     {trailing['share']:.4f} missing "
          f"of {trailing['n']} rows ({trailing['window_start']} .. "
          f"{trailing['window_end']}), threshold {report['threshold']}")
    for year, block in sorted(report["by_year"].items()):
        print(f"  {year}              {block['share']:.4f}  (n={block['n']})")
    print(f"wrote {args.out}")
    if args.echo:
        print(json.dumps(report, indent=2))
    warning = warning_text(report)
    if warning:
        print(f"\n{warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
