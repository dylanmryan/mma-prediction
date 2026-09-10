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
than a data-quality one.** It was pre-registered in
`docs/superpowers/plans/2026-09-09-external-decay-decision.md` §8 before any
candidate had been run, on the reasoning that the most recent fold the decay
decision could measure sits below it, so a trailing-12-month rate above 0.40
means the rows now being served are LESS covered than the evidence the
decision rests on. Set lower, the check would fire on a state the decision
already accounts for; set far higher, it would fire only once the block had
been quietly wrong for years.

**Two corrections to that reasoning, both made after the fact and neither
moving the threshold.** The pre-registration quoted the 2025 fold's coverage
as 0.359. That is the 2025 CALENDAR YEAR; the 2025 FOLD is unbounded above and
absorbs 2026 too, so the rows the recent-fold clause was actually measured on
run at **0.4225** missing (`mma.decay.coverage_by_fold`, recorded in the
decision artifact). And the trailing twelve months already stand at 0.51, so
**this check fires the day it is written**. The threshold is deliberately left
where it was pre-registered rather than raised to suit the number it produced:
moving a bar after seeing what it says about you is the one thing a
pre-registration exists to prevent.

What that costs is that the warning is permanent, and a permanent warning is
noise unless it says something a reader can act on. So it names the STANDING
DECISION -- its date, its verdict and the coverage it was taken at, read out
of `models/walkforward/external_decay_decision.json` rather than retyped. The
weekly message is therefore not "coverage is bad" but "coverage is X; the
standing answer was taken at Y and said Z" -- which is actionable exactly when
X has moved away from Y.

The arithmetic lives in `mma.decay` (`coverage_by_year`, `coverage_by_fold`,
`trailing_coverage`) and is unit-tested there; this file is the CLI, the
artifact and the warning.
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
#: Pre-registered in the decay plan §8 and deliberately never moved; see the
#: module docstring for the two after-the-fact corrections to its reasoning.
THRESHOLD = 0.40
#: The standing decision this check keeps honest. Optional: absent it, the
#: warning still fires, it just cannot say what the last answer was.
DECISION = ROOT / "models" / "walkforward" / "external_decay_decision.json"
#: Trailing window. Twelve months is one full event calendar, so the number is
#: not a seasonal artifact of which divisions happened to fight recently.
WINDOW_MONTHS = 12


def standing_decision(path: Path = DECISION) -> dict | None:
    """Date, verdict and coverage-at-the-time of the last decay decision.

    Read from the artifact rather than retyped, so the weekly warning cannot
    go on quoting an answer that has since been re-taken.
    """
    if not path.exists():
        return None
    artifact = json.loads(path.read_text())
    coverage = artifact.get("coverage") or {}
    return {
        "artifact": str(path.name),
        "verdict": artifact.get("decision", {}).get("outcome"),
        "per_group": artifact.get("decision", {}).get("confirmed_per_group"),
        "trailing_12m_at_decision": (coverage.get("trailing_12m") or {}).get("share"),
        "most_recent_fold_at_decision": coverage.get("most_recent_fold"),
    }


def measure(features: pd.DataFrame, *, threshold: float = THRESHOLD,
            months: int = WINDOW_MONTHS, as_of=None,
            decision: Path = DECISION) -> dict:
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
    standing = standing_decision(decision)
    drift = None
    if standing and standing["trailing_12m_at_decision"] is not None and trailing["n"]:
        drift = round(trailing["share"] - standing["trailing_12m_at_decision"], 4)
    return {
        "flag": FLAG,
        "threshold": threshold,
        "window_months": months,
        "trailing": trailing,
        "breached": breached,
        "standing_decision": standing,
        "drift_since_decision": drift,
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
    """The stderr warning for a breached report, else None.

    It states the standing decision alongside the number, because this
    threshold fires permanently on the current data and a permanent warning
    with no answer attached is what people learn to scroll past.
    """
    if not report["breached"]:
        return None
    trailing, standing = report["trailing"], report.get("standing_decision")
    lines = [
        f"WARNING: the external snapshot has not seen at least one corner of "
        f"{trailing['share']:.3f} of the {trailing['n']} fights in the trailing "
        f"{trailing['months']} months ({trailing['window_start']} .. "
        f"{trailing['window_end']}), past the {report['threshold']} threshold."
    ]
    if standing and standing.get("verdict"):
        at = standing["trailing_12m_at_decision"]
        lines.append(
            f"  Standing answer ({standing['artifact']}): {standing['verdict']}"
            + (f", taken at {at:.3f} trailing coverage." if at is not None else ".")
        )
        if report.get("drift_since_decision") is not None:
            lines.append(
                f"  Coverage has moved {report['drift_since_decision']:+.4f} since then. "
                "Re-take the decision (docs/superpowers/plans/"
                "2026-09-09-external-decay-decision.md) once that drift is large "
                "enough to matter, or refresh the source (scripts/build_external.py)."
            )
    else:
        lines.append(
            "  No decision artifact on disk: take the decay decision "
            "(docs/superpowers/plans/2026-09-09-external-decay-decision.md) or "
            "refresh the source (scripts/build_external.py)."
        )
    return "\n".join(lines)


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
    parser.add_argument("--decision", type=Path, default=DECISION)
    parser.add_argument("--print", dest="echo", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report = measure(pd.read_parquet(args.features), threshold=args.threshold,
                     months=args.months, as_of=args.as_of, decision=args.decision)
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
