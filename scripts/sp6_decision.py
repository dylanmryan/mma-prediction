"""Score SP6's three pre-registered candidates and apply the rules as written.

The candidates, the bar, the equal weighting and the mechanism secondary were
fixed in `docs/superpowers/plans/2026-09-15-sp6-third-blend-member.md` before
anything was built. This applies them mechanically and deploys nothing.

**The mechanism secondary is the interesting part, because it came first.** A
fixed-weight average can only gain when its members disagree, and disagreement
is measurable without the outcome. SP6 therefore pre-registered the incumbent
pair's own prediction correlation -- 0.8518 -- as the reference a third member
has to beat, and that number was in the plan before either new member existed.
Unlike SP5's mechanism, which could only be checked after the fact, this one
predicts the result in advance. It did.

  OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/sp6_decision.py \
      --run-dir <dir with the arm reports> --pred-dir <dir with the dumps>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PLAN = "docs/superpowers/plans/2026-09-15-sp6-third-blend-member.md"
OUT_PATH = ROOT / "models" / "walkforward" / "sp6_decision.json"

BAR = 0.003
SIGMA_BLEND = 0.0001
#: The incumbent pair's own prediction correlation, measured before either new
#: member was built and written into the plan. A third member more correlated
#: than this has less to offer than the second member did.
INCUMBENT_PAIR_CORRELATION = 0.8518
N_CANDIDATES = 3

CANDIDATES = (
    ("C1", "c1_logistic", ("logistic",), "equal-weight xgb + torch + logistic"),
    ("C2", "c2_bt", ("bt",), "equal-weight xgb + torch + bt"),
    ("C3", "c3_both", ("logistic", "bt"), "equal-weight xgb + torch + logistic + bt"),
)
INCUMBENT = "c0_incumbent"
MEMBER_DUMPS = ("torch", "xgb", "logistic", "bt")


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def member_matrix(pred_dir: Path) -> dict:
    """Standalone quality and the full pairwise correlation, paired on fight_id."""
    streams, shared = {}, None
    for name in MEMBER_DUMPS:
        path = pred_dir / f"{name}.json"
        if not path.exists():
            continue
        dump = json.loads(path.read_text())
        streams[name] = {str(f): (y, q) for f, y, q in zip(
            dump["fight_id"], dump["y_winner"], dump["p_winner"])}
        keys = set(streams[name])
        shared = keys if shared is None else shared & keys
    if not streams:
        return {}
    ids = sorted(shared)
    y = np.array([streams[MEMBER_DUMPS[0]][i][0] for i in ids], dtype=float)
    p = {k: np.array([v[i][1] for i in ids], dtype=float) for k, v in streams.items()}
    names = list(p)
    return {
        "n": len(ids),
        "reference": {
            "incumbent_pair_correlation": INCUMBENT_PAIR_CORRELATION,
            "meaning": ("a member more correlated with the incumbents than they "
                        "are with each other has less to add than the second "
                        "member did"),
        },
        "standalone": {
            k: {"winner_log_loss": round(log_loss(y, p[k]), 4),
                "accuracy": round(float(((p[k] > 0.5) == (y == 1)).mean()), 4)}
            for k in names
        },
        "prediction_correlation": {
            a: {b: round(float(np.corrcoef(p[a], p[b])[0, 1]), 4) for b in names}
            for a in names
        },
        "winner_disagreement_rate": {
            a: {b: round(float(np.mean((p[a] > 0.5) != (p[b] > 0.5))), 4)
                for b in names if b != a}
            for a in names
        },
    }


def assess(run_dir: Path, pred_dir: Path | None) -> dict:
    def report(stem):
        path = run_dir / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing report: {path}")
        return json.loads(path.read_text())

    inc = report(INCUMBENT)
    base = inc["pooled"]["winner_log_loss"]
    bar = max(BAR, 2 * SIGMA_BLEND)
    rows = []
    for arm, stem, extra, what in CANDIDATES:
        cand = report(stem)
        value = cand["pooled"]["winner_log_loss"]
        delta = value - base
        rows.append({
            "arm": arm, "extra_members": list(extra), "what": what,
            "report": f"{stem}.json",
            "pooled_winner_log_loss": value,
            "incumbent_pooled_winner_log_loss": base,
            "delta": round(delta, 5), "bar": round(bar, 5),
            "clears_bar": bool(delta < -bar),
            "accuracy": cand["pooled"]["accuracy"],
            "ece": cand["pooled"]["ece"],
            "per_fold_delta": {
                year: round(cand["folds"][year]["winner_log_loss"]
                            - inc["folds"][year]["winner_log_loss"], 4)
                for year in sorted(cand["folds"]) if year in inc["folds"]
            },
        })

    shipped = [r["arm"] for r in rows if r["clears_bar"]]
    best = min(rows, key=lambda r: r["delta"])
    out = {
        "experiment": "SP6 -- a structurally different third blend member",
        "pre_registration": PLAN,
        "generated_by": "scripts/sp6_decision.py",
        "primary_endpoint": "pooled out-of-fold winner log-loss, walk-forward",
        "paired_incumbent": {
            "report": f"{INCUMBENT}.json",
            "what": "the deployed two-member blend, xgb + torch",
            "pooled_winner_log_loss": base,
        },
        "n_candidates": N_CANDIDATES,
        "expected_best_of_n_under_the_null": round(
            SIGMA_BLEND * np.sqrt(2 * np.log(N_CANDIDATES)), 5),
        "arms": rows,
        "best_arm": {"arm": best["arm"], "delta": best["delta"],
                     "clears": best["clears_bar"]},
        "arms_clearing_the_bar": shipped,
        "fresh_seed_confirmation": (
            "not run: required only of an arm that clears the bar, and none did"
            if not shipped else "REQUIRED -- see the plan, section 4"),
    }
    if pred_dir is not None:
        out["mechanism_secondary"] = member_matrix(pred_dir)
    return out


def render(report: dict) -> None:
    base = report["paired_incumbent"]["pooled_winner_log_loss"]
    print(f"SP6: {report['n_candidates']} pre-registered candidates, paired "
          f"against the deployed blend at {base:.4f}\n")
    print(f"  {'arm':5s} {'members added':22s} {'pooled':>8s} {'delta':>9s} "
          f"{'acc':>7s}  verdict")
    for r in report["arms"]:
        print(f"  {r['arm']:5s} {'+'.join(r['extra_members']):22s} "
              f"{r['pooled_winner_log_loss']:8.4f} {r['delta']:+9.5f} "
              f"{r['accuracy']:7.4f}  {'SHIPS' if r['clears_bar'] else 'fails'}")
    print(f"\n  bar {report['arms'][0]['bar']}, selection optimism for three "
          f"candidates {report['expected_best_of_n_under_the_null']}")

    m = report.get("mechanism_secondary")
    if m:
        print(f"\n  MECHANISM SECONDARY (n={m['n']}), reference "
              f"{m['reference']['incumbent_pair_correlation']}")
        print(f"    {'member':10s} {'standalone':>11s}  correlation with the incumbents")
        for name, row in m["standalone"].items():
            corr = m["prediction_correlation"][name]
            with_inc = ", ".join(f"{k} {corr[k]:.4f}" for k in ("torch", "xgb")
                                 if k in corr and k != name)
            print(f"    {name:10s} {row['winner_log_loss']:11.4f}  {with_inc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    report = assess(args.run_dir, args.pred_dir)
    render(report)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
