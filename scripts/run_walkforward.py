"""Score one candidate on the walk-forward harness and write a JSON report.

Examples:
  python scripts/run_walkforward.py --candidate elo --name elo
  python scripts/run_walkforward.py --candidate xgb --name xgb_v1
  python scripts/run_walkforward.py --candidate torch --name torch_v1 --seeds 0,1,2,3,4
  python scripts/run_walkforward.py --candidate torch --name torch_v1_seeds5 --seeds 5,6,7,8,9
  python scripts/run_walkforward.py --candidate xgb --name xgb_hl4 --half-life 4 --train-start 2005-01-01
  python scripts/run_walkforward.py --candidate torch --name torch_refit --fixed-budget-from models/walkforward/torch_v1.json

Reports land in models/walkforward/<name>.json. Nothing here touches the
deployed artifacts under models/torch or models/xgb_*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.candidates import EloCandidate, TorchCandidate, XGBCandidate  # noqa: E402
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES  # noqa: E402
from mma.walkforward import build_report, make_folds, recency_weights  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models" / "walkforward"


def fixed_budget_from(report: dict) -> dict:
    """Median fit budget (and temperature) across the folds of a reference report.

    ``best_iteration`` is either a per-fold int (old single-budget XGB reports)
    or a per-fold dict of per-head best_iteration (winner/method/round); the
    latter yields a per-head median dict in ``fixed_rounds``.
    """
    info = report["fit_info"]
    budget = {}
    if "best_iteration" in info:
        entries = info["best_iteration"]
        if entries and isinstance(entries[0], dict):
            budget["fixed_rounds"] = {
                head: int(np.median([entry[head] for entry in entries]))
                for head in ("winner", "method", "round")
            }
        else:
            budget["fixed_rounds"] = int(np.median(entries))
    if "best_epoch" in info:
        per_fold = [float(np.median(e)) if isinstance(e, list) else float(e) for e in info["best_epoch"]]
        budget["fixed_epochs"] = int(np.median(per_fold)) + 1
    if "temperature" in info:
        per_fold = [float(np.median(t)) if isinstance(t, list) else float(t) for t in info["temperature"]]
        budget["temperature"] = round(float(np.median(per_fold)), 2)
    return budget


def build_candidate(kind: str, name: str, seeds: str, config: dict, budget: dict | None):
    """``budget`` is None without --fixed-budget-from; otherwise the dict from
    fixed_budget_from, which must carry the key this learner consumes
    (fixed_rounds for xgb, fixed_epochs for torch) -- a budget derived from
    the wrong learner's report, or an elo candidate, is a usage error."""
    if kind == "elo":
        if budget is not None:
            raise SystemExit("--fixed-budget-from does not apply to the elo candidate (nothing is fit)")
        return EloCandidate()
    needed = "fixed_rounds" if kind == "xgb" else "fixed_epochs"
    if budget is not None and needed not in budget:
        raise SystemExit(
            f"--fixed-budget-from report has no {needed!r} budget for the {kind} candidate "
            f"(found {sorted(budget) or 'nothing'}); point it at a {kind} walk-forward report"
        )
    budget = budget or {}
    if kind == "xgb":
        return XGBCandidate(name=name, params=config, fixed_rounds=budget.get("fixed_rounds"))
    return TorchCandidate(name=name, seeds=tuple(int(s) for s in seeds.split(",")), config=config,
                         fixed_epochs=budget.get("fixed_epochs"), temperature=budget.get("temperature"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", choices=["elo", "xgb", "torch"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--config-json", default=None, help="JSON dict of XGB params / torch config")
    parser.add_argument("--train-start", default=None, help="drop training fights before this date")
    parser.add_argument("--half-life", type=float, default=None, help="recency half-life in years")
    parser.add_argument("--fixed-budget-from", type=Path, default=None,
                        help="reference report; train on train+inner_val with its median budget, no early stopping")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    features = (
        pd.read_parquet(PROCESSED / "features.parquet")
        .sort_values("date", kind="stable").reset_index(drop=True)
    )
    config = json.loads(args.config_json) if args.config_json else {}
    budget = fixed_budget_from(json.loads(args.fixed_budget_from.read_text())) if args.fixed_budget_from else None
    candidate = build_candidate(args.candidate, args.name, args.seeds, config, budget)
    budget = budget or {}  # report shape: always a dict

    fold_results = []
    started = time.time()
    for fold in make_folds(features["date"], train_start=args.train_start):
        weights = recency_weights(features["date"], fold.eval_start, args.half_life) if args.half_life else None
        pred, info = candidate.fit_predict(features, fold, weights)
        fold_results.append((fold.year, fold.eval, pred, info))
        print(f"fold {fold.year}: n_train={int(fold.train.sum())} n_eval={int(fold.eval.sum())} info={info}")

    run_config = {
        "candidate": args.candidate, "seeds": args.seeds if args.candidate == "torch" else None,
        "config": config, "train_start": args.train_start, "half_life": args.half_life,
        "fixed_budget_from": str(args.fixed_budget_from) if args.fixed_budget_from else None,
        "budget": budget, "n_feature_rows": int(len(features)),
        "features_max_date": str(features["date"].max().date()),
        "runtime_sec": round(time.time() - started, 1),
    }
    report = build_report(args.name, run_config, features, fold_results, METHOD_CLASSES, ROUND_CLASSES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pooled": report["pooled"],
                      "folds": {y: f["winner_log_loss"] for y, f in report["folds"].items()}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
