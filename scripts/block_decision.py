"""Apply the pre-registered bar to one block's walk-forward report.

    python scripts/block_decision.py --candidate models/walkforward/xgb_in_fight.json \
        --incumbent models/walkforward/xgb_v1.json

Prints the bar_check block (bar, delta, per-fold deltas, ships) and a slice
table so a regression concentrated in one slice is visible. Writes nothing:
the decision is recorded by a human in the SP2 plan's results table, and a
block's code is reverted when it does not ship.

sigma_seed comes from models/walkforward/noise_floor.json (the cached seed
noise floor from scripts/noise_floor.py); --sigma overrides it. This is a
report, not a gate: it exits 0 whatever the decision, and non-zero only when
it cannot produce one (no cached noise floor, or a report with a non-finite
pooled metric).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.walkforward import bar_check, slice_comparison  # noqa: E402

WF = ROOT / "models" / "walkforward"
NOISE_FLOOR = WF / "noise_floor.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidate", type=Path, required=True,
                        help="walk-forward report for the block being judged")
    parser.add_argument("--incumbent", type=Path, required=True,
                        help="walk-forward report it must beat")
    parser.add_argument("--sigma", type=float, default=None,
                        help=f"override sigma_seed (default: from {NOISE_FLOOR.name})")
    parser.add_argument("--noise-floor", type=Path, default=NOISE_FLOOR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    sigma = args.sigma
    if sigma is None:
        if not args.noise_floor.exists():
            print(
                f"no cached seed noise floor at {args.noise_floor} -- run "
                "scripts/noise_floor.py first, or pass --sigma",
                file=sys.stderr,
            )
            raise SystemExit(1)
        sigma = json.loads(args.noise_floor.read_text())["sigma_seed"]

    candidate = json.loads(args.candidate.read_text())
    incumbent = json.loads(args.incumbent.read_text())

    print(f"candidate: {candidate.get('name', args.candidate.stem)} ({args.candidate})")
    print(f"incumbent: {incumbent.get('name', args.incumbent.stem)} ({args.incumbent})")
    print()
    print("bar_check:")
    print(json.dumps(bar_check(candidate, incumbent, sigma), indent=2))
    print()
    print("slices (delta = candidate - incumbent, negative = candidate better):")
    print(json.dumps(slice_comparison(candidate, incumbent), indent=2))


if __name__ == "__main__":
    main()
