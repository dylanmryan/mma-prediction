"""ONE-TIME held-out test evaluation on 2024+ fights -- FROZEN.

This ran exactly once, on 2026-07-13, against the pre-refit July 2026
models (trained on pre-2021 fights, temperatures fit on 2021-2023
validation) and wrote models/final_test_metrics.json. At that time the
test years (2024+) had never been read by any training, tuning, or
calibration code, which is what made the number a genuine holdout.

That is no longer true: the deployed models now follow the refit_through
recipe (models/walkforward/refit_decision.json) and train on every
decisive fight through the latest event, so 2024+ is training data, and
the walk-forward harness (scripts/run_walkforward.py) scores those years
as expanding-window folds instead. Re-running this script would report
in-sample numbers under a "held-out" label, so main() refuses to run
whenever models/torch/metrics_val.json says the incumbent is a refit
model. The committed artifact is kept as the historical record; the
prospective track record (predictions/track_record.json) is the only
true holdout now.

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
from mma.inference import BlendedPredictor, Ensemble
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
from mma.models.xgb import feature_frame

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def _load_seed_ensemble(head: str) -> list:
    """The committed XGBoost seed ensemble for one head, in seed order."""
    paths = sorted(MODELS.glob(f"xgb_{head}_seed*.json"),
                   key=lambda q: int(q.stem.rsplit("seed", 1)[1]))
    if not paths:
        raise FileNotFoundError(
            f"no xgb_{head}_seed*.json under {MODELS}; run scripts/train_xgb.py"
        )
    models = []
    for path in paths:
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(path)
        models.append(model)
    return models


def _mean_proba(models: list, x) -> np.ndarray:
    """The ensemble's prediction: the mean of the members' predict_proba."""
    return np.mean([model.predict_proba(x) for model in models], axis=0)
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


def refuse_if_refit_incumbent(torch_metrics_path: Path) -> None:
    """SystemExit when the deployed torch ensemble was trained with the
    refit_through recipe: the 2024+ years are then training data and the
    held-out evaluation cannot be repeated."""
    if not torch_metrics_path.exists():
        return
    metrics = json.loads(torch_metrics_path.read_text())
    if metrics.get("mode") == "refit_through":
        raise SystemExit(
            "final_test_metrics.json is frozen from the pre-refit (July 2026) "
            "model; the 2024+ years are now training data (the deployed "
            f"ensemble is refit through {metrics.get('train_through')}) -- do "
            "not re-run. The walk-forward harness (scripts/run_walkforward.py, "
            "models/walkforward/) is the evaluation of record for those years."
        )


def main() -> None:
    refuse_if_refit_incumbent(MODELS / "torch" / "metrics_val.json")
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

    # 3. XGBoost winner: the committed 5-seed ensemble, no refit
    x = feature_frame(features)
    xgb_winner = _load_seed_ensemble("winner")
    p_xgb = _mean_proba(xgb_winner, x[test])[:, 1]
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

    # 5. The DEPLOYED SCORER: the blend of 3 and 4, temperature-scaled after
    # averaging. The two members above are reported as its components.
    blended = BlendedPredictor.load(ROOT)
    p_blend = blended.predict(features.loc[test])["winner_prob"]
    results["winner"]["blend_deployed"] = {
        "n_test": n_test,
        **winner_metrics(y_test, p_blend),
    }

    # method of victory (rows with known method)
    method_known = test & features["y_method"].notna().to_numpy()
    method_truth = list(features.loc[method_known, "y_method"])
    method_majority = features.loc[
        train & features["y_method"].notna().to_numpy(), "y_method"
    ].mode()[0]

    method_pred_xgb = [
        METHOD_CLASSES[i]
        for i in _mean_proba(_load_seed_ensemble("method"), x[method_known]).argmax(axis=1)
    ]

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

    round_pred_xgb = [
        ROUND_CLASSES[i]
        for i in _mean_proba(_load_seed_ensemble("round"), x[finish_known]).argmax(axis=1)
    ]

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
