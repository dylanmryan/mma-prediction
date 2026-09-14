"""Score SP5's six pre-registered arms and apply the rules as written.

The arms, the bar, the fresh-seed requirement and the mechanism secondary were
all fixed in `docs/superpowers/plans/2026-09-13-sp5-scorecard-label.md` before
anything was built. This script reads the reports those arms wrote and applies
the rules mechanically; it decides nothing that the plan did not already
decide, and it deploys nothing.

**The mechanism secondary is the part worth keeping when the primary fails.**
SP5's claim was not "margins help" but "margins help where the model is
currently silent" -- 79% of out-of-fold predictions sit in bands where it is
52-59% accurate, and it cannot tell a genuine toss-up from a fight it failed
to read. So the plan pre-registered the improvement restricted to
|p - 0.5| < 0.10. An arm clearing the bar without that concentration would
have shipped anyway, with the notes recording that the stated mechanism was
wrong. An arm that fails the bar still answers a different question: whether
the mechanism was WRONG, or merely too small to pay for.

  OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/sp5_decision.py \
      --run-dir <dir with the arm reports>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PLAN = "docs/superpowers/plans/2026-09-13-sp5-scorecard-label.md"
OUT_PATH = ROOT / "models" / "walkforward" / "sp5_decision.json"

#: Fixed by the plan, section 5.
BAR = 0.003
SIGMA_TORCH = 0.0003464101615137373
SIGMA_XGB = 0.0012463279397226486
#: Section 6: where the gain was predicted to land.
COINFLIP_BAND = 0.10

#: The plan's section 5 fixed N = 6 for the selection-optimism arithmetic,
#: counting T0 among the arms. Five COMPARISONS are actually made (T0 and X0
#: are the paired incumbents, not candidates), so computing it from the arm
#: table would give a slightly SMALLER correction than was registered.
#: Reporting the registered figure keeps the correction on the conservative
#: side of the difference rather than quietly relaxing it after the fact.
PREREGISTERED_N_ARMS = 6

#: (arm, member, report stem, incumbent stem, what it changed)
ARMS = (
    ("T1", "torch", "t_margin_0.1", "t0_incumbent_check", "margin head, lambda=0.1"),
    ("T2", "torch", "t_margin_0.3", "t0_incumbent_check", "margin head, lambda=0.3"),
    ("T3", "torch", "t_margin_1.0", "t0_incumbent_check", "margin head, lambda=1.0"),
    ("X1", "xgb", "x_soft_t2", "x0_hard", "soft label, T=2"),
    ("X2", "xgb", "x_soft_t4", "x0_hard", "soft label, T=4"),
)


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def per_row_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def banded_comparison(incumbent: dict, candidate: dict) -> dict:
    """The mechanism secondary: where does the difference actually land?

    Paired on `fight_id`, never on row position -- this repo has mis-paired
    two prediction dumps that way already.
    """
    a = {str(f): (y, p) for f, y, p in zip(
        incumbent["fight_id"], incumbent["y_winner"], incumbent["p_winner"])}
    b = {str(f): (y, p) for f, y, p in zip(
        candidate["fight_id"], candidate["y_winner"], candidate["p_winner"])}
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("the two dumps share no fight_id")
    y = np.array([a[f][0] for f in shared], dtype=float)
    p_inc = np.array([a[f][1] for f in shared], dtype=float)
    p_cand = np.array([b[f][1] for f in shared], dtype=float)
    if not np.allclose(y, [b[f][0] for f in shared]):
        raise SystemExit("the dumps disagree about who won")

    loss_inc, loss_cand = per_row_loss(y, p_inc), per_row_loss(y, p_cand)
    # the incumbent's own confidence defines the band, so the split does not
    # move with the candidate being tested
    conf = np.abs(p_inc - 0.5)
    bands = {}
    for label, mask in (
        (f"coin-flip |p-0.5| < {COINFLIP_BAND}", conf < COINFLIP_BAND),
        (f"decided |p-0.5| >= {COINFLIP_BAND}", conf >= COINFLIP_BAND),
    ):
        if not mask.any():
            continue
        bands[label] = {
            "n": int(mask.sum()),
            "share_of_rows": round(float(mask.mean()), 4),
            "incumbent_log_loss": round(float(loss_inc[mask].mean()), 4),
            "candidate_log_loss": round(float(loss_cand[mask].mean()), 4),
            "delta": round(float((loss_cand - loss_inc)[mask].mean()), 5),
        }
    pooled = float((loss_cand - loss_inc).mean())
    coin = bands.get(f"coin-flip |p-0.5| < {COINFLIP_BAND}", {})
    decided = bands.get(f"decided |p-0.5| >= {COINFLIP_BAND}", {})
    return {
        "n_paired": len(shared),
        "paired_on": "fight_id",
        "pooled_delta": round(pooled, 5),
        "bands": bands,
        "mechanism_predicted": (
            "the gain concentrates in the coin-flip band, where the model "
            "currently cannot tell a toss-up from a fight it failed to read"
        ),
        "gain_concentrates_in_the_coinflip_band": bool(
            coin and decided and coin.get("delta", 0) < decided.get("delta", 0)
            and coin.get("delta", 0) < 0
        ),
    }


def assess(run_dir: Path) -> dict:
    def report(stem):
        path = run_dir / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing arm report: {path}")
        return json.loads(path.read_text())

    rows = []
    for arm, member, stem, incumbent_stem, what in ARMS:
        cand, inc = report(stem), report(incumbent_stem)
        c, i = cand["pooled"]["winner_log_loss"], inc["pooled"]["winner_log_loss"]
        delta = c - i
        sigma = SIGMA_TORCH if member == "torch" else SIGMA_XGB
        bar = max(BAR, 2 * sigma)
        rows.append({
            "arm": arm, "member": member, "what_it_changed": what,
            "report": f"{stem}.json", "paired_incumbent": f"{incumbent_stem}.json",
            "pooled_winner_log_loss": c,
            "incumbent_pooled_winner_log_loss": i,
            "delta": round(delta, 5),
            "bar": round(bar, 5),
            "sigma_seed": sigma,
            "clears_bar": bool(delta < -bar),
            "accuracy": cand["pooled"]["accuracy"],
            "ece": cand["pooled"]["ece"],
            "per_fold_delta": {
                year: round(cand["folds"][year]["winner_log_loss"]
                            - inc["folds"][year]["winner_log_loss"], 4)
                for year in sorted(cand["folds"])
                if year in inc["folds"]
            },
        })

    shipped = [r["arm"] for r in rows if r["clears_bar"]]
    best = min(rows, key=lambda r: r["delta"])
    return {
        "experiment": "SP5 -- the scorecard label",
        "pre_registration": PLAN,
        "generated_by": "scripts/sp5_decision.py",
        "primary_endpoint": "pooled out-of-fold winner log-loss, walk-forward",
        "n_arms_preregistered": PREREGISTERED_N_ARMS,
        "n_comparisons_made": len(ARMS),
        "expected_best_of_n_under_the_null": {
            "torch": round(SIGMA_TORCH * np.sqrt(2 * np.log(PREREGISTERED_N_ARMS)), 5),
            "xgb": round(SIGMA_XGB * np.sqrt(2 * np.log(PREREGISTERED_N_ARMS)), 5),
            "note": ("sigma * sqrt(2 ln N) at the pre-registered N = 6, the "
                     "selection optimism a best-of-six search carries before "
                     "any real effect. Five comparisons were actually made; "
                     "the registered N is reported because recomputing it "
                     "downward after the fact would weaken the correction."),
        },
        "arms": rows,
        "best_arm": {"arm": best["arm"], "delta": best["delta"],
                     "bar": best["bar"], "clears": best["clears_bar"]},
        "arms_clearing_the_bar": shipped,
        "fresh_seed_confirmation": (
            "not run: required only for an arm that clears the bar, and none did"
            if not shipped else "REQUIRED -- see the plan, section 5"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--incumbent-dump", type=Path)
    parser.add_argument("--candidate-dump", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    report = assess(args.run_dir)
    if args.incumbent_dump and args.candidate_dump:
        report["mechanism_secondary"] = banded_comparison(
            json.loads(args.incumbent_dump.read_text()),
            json.loads(args.candidate_dump.read_text()),
        )

    print(f"SP5: {report['n_arms_preregistered']} pre-registered arms, "
          f"{report['n_comparisons_made']} comparisons\n")
    print(f"  {'arm':5s} {'member':7s} {'what':22s} {'pooled':>8s} {'delta':>9s} "
          f"{'bar':>8s}  verdict")
    for r in report["arms"]:
        print(f"  {r['arm']:5s} {r['member']:7s} {r['what_it_changed']:22s} "
              f"{r['pooled_winner_log_loss']:8.4f} {r['delta']:+9.5f} "
              f"{r['bar']:8.5f}  {'SHIPS' if r['clears_bar'] else 'fails'}")
    best = report["best_arm"]
    print(f"\n  best: {best['arm']} at {best['delta']:+.5f} against a bar of "
          f"{best['bar']:.5f}")
    print(f"  selection optimism for six arms: torch "
          f"{report['expected_best_of_n_under_the_null']['torch']:.5f}, xgb "
          f"{report['expected_best_of_n_under_the_null']['xgb']:.5f}")

    if "mechanism_secondary" in report:
        m = report["mechanism_secondary"]
        print(f"\n  MECHANISM SECONDARY (n={m['n_paired']}, paired on fight_id)")
        for label, band in m["bands"].items():
            print(f"    {label:28s} n={band['n']:5d} ({band['share_of_rows']:5.1%})  "
                  f"delta {band['delta']:+.5f}")
        print(f"    -> gain concentrates where predicted: "
              f"{m['gain_concentrates_in_the_coinflip_band']}")

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
