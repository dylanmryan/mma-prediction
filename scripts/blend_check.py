"""Score the equal-weight average of the xgb and torch candidates out-of-fold.

    python scripts/blend_check.py --name blend_a1_xgb_torch \
        --drop-columns external_missing,same_country,notice_unknown,home_country_a,home_country_b

This is a MEASUREMENT, not a ship candidate. SP2.1's pre-registration names
five arms and a blend is not one of them, so nothing here can ship under it;
the number exists so the human call on "which scorer should be deployed" is
made with the third option quantified rather than assumed. model-v2 already
found stacking lost to a simple average, which is why the average is the
form measured.

Both candidates are fitted on the same folds of the same table with their
deployed configurations, and their per-head probabilities are averaged
row-wise before scoring -- so the report is directly comparable to the
single-candidate reports in models/walkforward/.
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

from mma.candidates import TorchCandidate, XGBCandidate  # noqa: E402
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES  # noqa: E402
from mma.walkforward import build_report, make_folds  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models" / "walkforward"


def average_predictions(preds: list[dict]) -> dict:
    """Row-wise mean of each head across members.

    A head is averaged only when every member produced it; if any member
    returned None for a head (the elo floor does, though it is not a member
    here) the blend has nothing to average and the head is None.
    """
    out = {}
    for head in ("winner", "method", "round"):
        parts = [p[head] for p in preds]
        out[head] = None if any(part is None for part in parts) else np.mean(parts, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4", help="torch member's seed ensemble")
    parser.add_argument("--drop-columns", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    drop_columns = tuple(n.strip() for n in (args.drop_columns or "").split(",") if n.strip())
    features = (
        pd.read_parquet(PROCESSED / "features.parquet")
        .sort_values("date", kind="stable").reset_index(drop=True)
    )
    unknown = [n for n in drop_columns if n not in features.columns]
    if unknown:
        raise SystemExit(f"--drop-columns names column(s) absent from the feature table: {unknown}")

    members = [
        XGBCandidate(name="xgb", drop_columns=drop_columns),
        TorchCandidate(name="torch", seeds=tuple(int(s) for s in args.seeds.split(",")),
                       drop_columns=drop_columns),
    ]

    fold_results = []
    started = time.time()
    for fold in make_folds(features["date"]):
        preds, infos = [], {}
        for member in members:
            pred, info = member.fit_predict(features, fold, None)
            preds.append(pred)
            infos[member.name] = info
        fold_results.append((fold.year, fold.eval, average_predictions(preds), infos))
        print(f"fold {fold.year}: n_eval={int(fold.eval.sum())}")

    run_config = {
        "candidate": "blend(xgb,torch)", "members": ["xgb", "torch"], "weights": [0.5, 0.5],
        "seeds": args.seeds, "config": {}, "drop_columns": list(drop_columns),
        "n_feature_rows": int(len(features)),
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
