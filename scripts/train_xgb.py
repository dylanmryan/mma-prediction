"""Train XGBoost baselines; report validation-years metrics only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mma.evaluate import accuracy, brier_score, log_loss, macro_f1
from mma.models.xgb import feature_frame, train_binary, train_multiclass

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
TRAIN_END = "2021-01-01"
VAL_START, VAL_END = "2021-01-01", "2023-12-31"
METHOD_CLASSES = ["ko_tko", "submission", "decision"]
ROUND_CLASSES = ["1", "2", "3", "45"]


def main() -> None:
    # --train-end/--val-start/--val-end/--models-dir default to the module
    # constants above (the standard pre-2021/2021-2023 split used by every
    # other script and by app.py). scripts/roll_window.py --execute passes
    # different values to re-validate a walk-forward retrain on a newer
    # slice without touching this script's normal weekly-refresh behavior.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--val-start", default=VAL_START)
    parser.add_argument("--val-end", default=VAL_END)
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    args = parser.parse_args()

    features = pd.read_parquet(PROCESSED / "features.parquet")
    x = feature_frame(features)
    train = features["date"] < args.train_end
    val = (features["date"] >= args.val_start) & (features["date"] <= args.val_end)
    models_dir = args.models_dir
    models_dir.mkdir(exist_ok=True, parents=True)
    metrics = {}

    # winner
    y = features["y_winner"]
    winner = train_binary(x[train], y[train], x[val], y[val])
    p_val = winner.predict_proba(x[val])[:, 1]
    metrics["winner"] = {
        "n_val": int(val.sum()),
        "accuracy": round(accuracy(y[val], p_val), 4),
        "log_loss": round(log_loss(y[val], p_val), 4),
        "brier": round(brier_score(y[val], p_val), 4),
        "best_iteration": int(winner.best_iteration),
    }
    winner.save_model(models_dir / "xgb_winner.json")

    # method (rows with known method)
    known = features["y_method"].notna()
    y = features["y_method"]
    method = train_multiclass(
        x[train & known], y[train & known], x[val & known], y[val & known],
        METHOD_CLASSES,
    )
    pred = [METHOD_CLASSES[i] for i in method.predict(x[val & known])]
    truth = list(y[val & known])
    majority = y[train & known].mode()[0]
    metrics["method"] = {
        "n_val": int((val & known).sum()),
        "accuracy": round(float(sum(p == t for p, t in zip(pred, truth)) / len(truth)), 4),
        "macro_f1": round(macro_f1(truth, pred), 4),
        "majority_baseline_accuracy": round(float(sum(t == majority for t in truth) / len(truth)), 4),
    }
    method.save_model(models_dir / "xgb_method.json")

    # finish round (finishes only)
    finish = features["y_finish_round"].notna()
    y = features["y_finish_round"]
    rounds = train_multiclass(
        x[train & finish], y[train & finish], x[val & finish], y[val & finish],
        ROUND_CLASSES,
    )
    pred = [ROUND_CLASSES[i] for i in rounds.predict(x[val & finish])]
    truth = list(y[val & finish])
    majority = y[train & finish].mode()[0]
    metrics["finish_round"] = {
        "n_val": int((val & finish).sum()),
        "accuracy": round(float(sum(p == t for p, t in zip(pred, truth)) / len(truth)), 4),
        "macro_f1": round(macro_f1(truth, pred), 4),
        "majority_baseline_accuracy": round(float(sum(t == majority for t in truth) / len(truth)), 4),
    }
    rounds.save_model(models_dir / "xgb_round.json")

    (models_dir / "xgb_metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    importances = pd.Series(
        winner.feature_importances_, index=x.columns
    ).sort_values(ascending=False)
    print("\ntop 15 winner-model features:")
    print(importances.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
