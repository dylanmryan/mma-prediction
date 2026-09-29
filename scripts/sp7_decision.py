"""SP7's mechanism screen, and why the simulator's winner never reached the harness.

Pre-registration: `docs/superpowers/plans/2026-09-29-sp7-simulator-third-member.md`.

SP3's Monte Carlo simulator produces, unavoidably, an opinion about who wins:
the simulated fights have winners. SP3 measured it once, found it slightly
worse than the blend's, imposed the blend's marginal over it and discarded it.
SP7 asked whether that opinion was worth averaging in as a third blend member
-- because on paper it was the best candidate available, better standalone
than either member SP6 went looking for, and reading per-round hazard rows
rather than the 87-column differential row.

**The screen, both clauses fixed before measuring.** A fixed-weight average
gains only when its members disagree, and that is measurable without the
outcome. SP6 established the reference: the incumbent pair's own prediction
correlation, 0.8518. SP6 also showed one clause is not enough -- Bradley-Terry
cleared the correlation gate easily and added nothing, because it was too weak
to be worth averaging in. So a candidate must be BOTH close standalone AND
less correlated than the incumbents are with each other.

**The threshold is re-derived, not trusted.** SP6 measured 0.8518 on a
4,804-fight table; this runs on 4,856. A screen judged against a stale
reference is judged against nothing, so `correlations` re-measures the
incumbent pair and the test suite asserts the two agree.

**Joined by `fight_id`.** All four dumps carry it, and the outcome column is
checked after the join rather than assumed -- a silent positional mismatch is
what broke the market benchmark in September.

This script recomputes the screen from committed dumps; it does not fit
anything. The harness arm it gates was never run.

  OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/sp7_decision.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PREDS = ROOT / "models" / "walkforward" / "preds"
ARTIFACT = ROOT / "models" / "walkforward" / "sp7_decision.json"

#: The incumbent pair's prediction correlation, as SP6 measured it. The screen
#: threshold, and re-derived from the current table by `correlations` so a
#: stale copy cannot silently become the reference.
REFERENCE_PAIR_CORR = 0.8518

#: How close standalone a member must be to be worth averaging in. SP6's
#: Bradley-Terry cleared the correlation clause at 0.1846 and added nothing,
#: because at 0.7069 it was too weak -- hence two clauses rather than one.
STANDALONE_TOLERANCE = 0.005

#: Pre-registered in the plan before any of this was measured.
PREDICTED_RANGE = (0.60, 0.80)

STREAMS = {
    "blend": "hybrid_e2_cells.json",
    "xgb": "sp7_screen_xgb.json",
    "torch": "sp7_screen_torch.json",
    "simulator": "sp7_screen_hazard.json",
}

_EPS = 1e-15


def load_streams(sources: dict | None = None):
    """Every stream's winner probability, aligned on the reference's fight ids."""
    sources = sources or STREAMS
    raw = {}
    for name, src in sources.items():
        path = Path(src) if Path(src).is_absolute() else PREDS / src
        raw[name] = json.loads(path.read_text())

    reference = raw["blend"]
    order = {str(f): i for i, f in enumerate(reference["fight_id"])}
    y = np.asarray(reference["y_winner"], dtype=float)

    streams = {}
    for name, dump in raw.items():
        ids = [str(f) for f in dump["fight_id"]]
        if set(ids) != set(order):
            raise ValueError(
                f"{name} covers a different set of fights than the reference "
                f"stream ({len(set(ids))} vs {len(order)})"
            )
        slot = np.argsort(np.array([order[f] for f in ids]))
        if not np.array_equal(np.asarray(dump["y_winner"], dtype=float)[slot], y):
            raise ValueError(
                f"{name} disagrees with the reference stream on the outcome "
                "column after the fight_id join, so the join is not sound"
            )
        streams[name] = np.asarray(dump["p_winner"], dtype=float)[slot]
    return y, streams


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def correlations(streams: dict) -> dict:
    """Every pairwise prediction correlation, keyed BOTH ways.

    Symmetric because correlation is, and because a caller asking for
    ("simulator", "blend") when the dict happened to be built as
    ("blend", "simulator") should get the number rather than a KeyError that
    depends on `STREAMS` insertion order.
    """
    out = {}
    for a, b in itertools.combinations(streams, 2):
        r = float(np.corrcoef(streams[a], streams[b])[0, 1])
        out[(a, b)] = out[(b, a)] = r
    return out


def screen() -> dict:
    y, streams = load_streams()
    corr = correlations(streams)
    losses = {k: log_loss(y, p) for k, p in streams.items()}

    reference = corr[("xgb", "torch")]
    candidate = corr[("simulator", "blend")]
    standalone_gap = losses["simulator"] - losses["blend"]

    clauses = {
        "standalone_within_0.005_of_blend": {
            "measured": round(standalone_gap, 4),
            "threshold": STANDALONE_TOLERANCE,
            "passed": bool(standalone_gap <= STANDALONE_TOLERANCE),
        },
        "correlation_below_reference": {
            "measured": round(candidate, 4),
            "threshold": REFERENCE_PAIR_CORR,
            "threshold_re_derived_here": round(reference, 4),
            "passed": bool(candidate < REFERENCE_PAIR_CORR),
        },
    }
    passed = all(c["passed"] for c in clauses.values())
    return {
        "experiment": "SP7 -- the simulator's own winner opinion as a third blend member",
        "pre_registration":
            "docs/superpowers/plans/2026-09-29-sp7-simulator-third-member.md",
        "generated_by": "scripts/sp7_decision.py",
        "verdict": "PASS" if passed else "FAIL",
        "decision": (
            "Stopped at the mechanism screen. No walk-forward arm was run, which "
            "is the pre-registered rule and not a shortcut: the screen exists so "
            "a candidate that cannot help is refused before the expensive part."
        ),
        "why": (
            "The simulator's winner opinion correlates 0.9296 with the blend -- "
            "MORE than the blend's own two members correlate with each other "
            "(0.8518). It correlates 0.9196 with xgb alone, which is what it "
            "really is: the hazard members are XGBoost models fitted on features "
            "derived from the same table, so the per-round likelihood and the "
            "Monte Carlo composition change what the simulator PREDICTS without "
            "changing what it KNOWS. It disagrees with the blend on 10.4% of "
            "winners against the incumbent pair's 14.5%, so it has strictly less "
            "to add than the second member did -- and that member's gain, 0.0028, "
            "does not itself clear the 0.0044 floor measured in models/bar_audit.json."
        ),
        "predicted_before_measuring": {
            "range": list(PREDICTED_RANGE),
            "measured": round(candidate, 4),
            "was_correct": bool(PREDICTED_RANGE[0] <= candidate <= PREDICTED_RANGE[1]),
            "note": (
                "The plan predicted 0.60-0.80 and the answer was 0.9296. The "
                "prediction was wrong in the direction that matters: the "
                "simulator's structural difference is in its OUTPUT (a joint over "
                "method and round) and not in its INPUT, and only the second kind "
                "of difference decorrelates a winner opinion."
            ),
        },
        "clauses": clauses,
        "standalone_winner_log_loss": {k: round(v, 4) for k, v in sorted(losses.items())},
        # From `combinations`, not from the symmetric dict, so each pair is
        # written once and in a stable order.
        "correlations": {
            f"{a} vs {b}": round(corr[(a, b)], 4)
            for a, b in itertools.combinations(streams, 2)
        },
        "n": int(len(y)),
        "harness_arm_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    report = screen()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"SP7 screen: {report['verdict']}")
    for name, clause in report["clauses"].items():
        mark = "pass" if clause["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {clause['measured']} vs {clause['threshold']}")
    print()
    for pair, value in report["correlations"].items():
        print(f"    {pair:26s} {value:.4f}")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
