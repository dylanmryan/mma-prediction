"""ONE-TIME held-out test evaluation on 2024+ fights.

The test years (2024+) were never read by any training, tuning, or
calibration code anywhere in this project. This script evaluates the
locked, committed artifacts on them exactly once and reports whatever
comes out. Nothing is refit here: the XGBoost models, the 5-seed torch
checkpoints, their per-seed temperatures (fit on 2021-2023 validation),
and the preprocessor are all loaded as committed.

Do NOT change any model, feature, threshold, or calibration in response
to these numbers.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from mma.elo import expected_score
from mma.evaluate import accuracy, brier_score, log_loss, macro_f1
from mma.inference import Ensemble
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
from mma.models.xgb import feature_frame

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
TEST_START = "2024-01-01"
TRAIN_END = "2021-01-01"  # majority baselines come from the training split


def winner_metrics(y, p) -> dict:
    return {
        "accuracy": round(accuracy(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier_score(y, p), 4),
    }


def elo_test_predictions() -> pd.DataFrame:
    """Same construction as scripts/build_ratings.py::_predictions, on test rows.

    One row per rated fight with a decisive winner: p(A wins) from the
    committed pre-fight overall Elo ratings (built with parameters tuned
    on pre-2020 fights only).
    """
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    a = ratings[ratings["corner"] == "a"][["fight_id", "pre_overall"]]
    b = ratings[ratings["corner"] == "b"][["fight_id", "pre_overall"]]
    merged = (
        fights[fights["winner"].isin(["a", "b"])][["fight_id", "date", "winner"]]
        .merge(a, on="fight_id")
        .merge(b, on="fight_id", suffixes=("_a", "_b"))
    )
    merged["p_a"] = [
        expected_score(ra, rb)
        for ra, rb in zip(merged["pre_overall_a"], merged["pre_overall_b"])
    ]
    merged["y"] = (merged["winner"] == "a").astype(float)
    return merged[merged["date"] >= TEST_START]


def multiclass_block(truth: list, pred: list, majority: str) -> dict:
    majority_pred = [majority] * len(truth)
    return {
        "accuracy": round(float(np.mean([p == t for p, t in zip(pred, truth)])), 4),
        "macro_f1": round(macro_f1(truth, pred), 4),
        "majority_baseline": {
            "class": majority,
            "accuracy": round(float(np.mean([t == majority for t in truth])), 4),
            "macro_f1": round(macro_f1(truth, majority_pred), 4),
        },
    }


def main() -> None:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    test = (features["date"] >= TEST_START).to_numpy()
    train = (features["date"] < TRAIN_END).to_numpy()
    y_test = features.loc[test, "y_winner"].to_numpy(dtype=float)
    n_test = int(test.sum())

    results: dict = {
        "computed_once_on": datetime.date.today().isoformat(),
        "test_split": f"date >= {TEST_START}",
        "winner": {},
    }

    # 1a. coin flip
    results["winner"]["coin_flip"] = {
        "n_test": n_test,
        **winner_metrics(y_test, np.full(n_test, 0.5)),
    }

    # 1b + 2. Elo expected-score baseline and higher-Elo-wins dummy
    elo = elo_test_predictions()
    results["winner"]["higher_elo_dummy"] = {
        "n_test": int(len(elo)),
        "accuracy": round(
            accuracy(elo["y"], (elo["p_a"] > 0.5).astype(float)), 4
        ),
        "log_loss": None,  # hard 0/1 predictions; log-loss not meaningful
        "brier": None,
    }
    results["winner"]["elo"] = {
        "n_test": int(len(elo)),
        **winner_metrics(elo["y"], elo["p_a"]),
    }

    # 3. XGBoost winner (committed models/xgb_winner.json, no refit)
    x = feature_frame(features)
    xgb_winner = xgb.XGBClassifier(enable_categorical=True)
    xgb_winner.load_model(MODELS / "xgb_winner.json")
    p_xgb = xgb_winner.predict_proba(x[test])[:, 1]
    results["winner"]["xgboost"] = {
        "n_test": n_test,
        **winner_metrics(y_test, p_xgb),
    }

    # 4. Torch 5-seed ensemble with COMMITTED per-seed temperatures (no refit)
    ensemble = Ensemble.load()
    torch_out = ensemble.predict(features.loc[test])
    p_torch = torch_out["winner_prob"]
    results["winner"]["torch_ensemble"] = {
        "n_test": n_test,
        **winner_metrics(y_test, p_torch),
    }

    # method of victory (rows with known method)
    method_known = test & features["y_method"].notna().to_numpy()
    method_truth = list(features.loc[method_known, "y_method"])
    method_majority = features.loc[
        train & features["y_method"].notna().to_numpy(), "y_method"
    ].mode()[0]

    xgb_method = xgb.XGBClassifier(enable_categorical=True)
    xgb_method.load_model(MODELS / "xgb_method.json")
    method_pred_xgb = [METHOD_CLASSES[i] for i in xgb_method.predict(x[method_known])]

    method_out = ensemble.predict(features.loc[method_known])
    method_pred_torch = [
        METHOD_CLASSES[i] for i in method_out["method_probs"].argmax(axis=1)
    ]

    results["method"] = {
        "n_test": int(method_known.sum()),
        "xgboost": multiclass_block(method_truth, method_pred_xgb, method_majority),
        "torch_ensemble": multiclass_block(
            method_truth, method_pred_torch, method_majority
        ),
    }

    # finish round (finishes only)
    finish_known = test & features["y_finish_round"].notna().to_numpy()
    round_truth = list(features.loc[finish_known, "y_finish_round"])
    round_majority = features.loc[
        train & features["y_finish_round"].notna().to_numpy(), "y_finish_round"
    ].mode()[0]

    xgb_round = xgb.XGBClassifier(enable_categorical=True)
    xgb_round.load_model(MODELS / "xgb_round.json")
    round_pred_xgb = [ROUND_CLASSES[i] for i in xgb_round.predict(x[finish_known])]

    results["finish_round"] = {
        "n_test": int(finish_known.sum()),
        "xgboost": multiclass_block(round_truth, round_pred_xgb, round_majority),
    }

    results["n_test"] = {
        "winner_features": n_test,
        "winner_elo": int(len(elo)),
        "method": int(method_known.sum()),
        "finish_round": int(finish_known.sum()),
    }

    (MODELS / "final_test_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
