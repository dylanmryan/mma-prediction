"""Train the 5-seed ensemble; report validation-years metrics only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mma.evaluate import accuracy, brier_score, log_loss, macro_f1
from mma.models.net import MultiTaskNet
from mma.models.train_loop import (
    METHOD_CLASSES, ROUND_CLASSES, encode_targets, fit_temperature, predict, train_one,
)
from mma.tensors import Preprocessor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "models" / "torch"
TRAIN_END = "2021-01-01"
VAL_START, VAL_END = "2021-01-01", "2023-12-31"
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    train = (features["date"] < TRAIN_END).to_numpy()
    val = (
        (features["date"] >= VAL_START) & (features["date"] <= VAL_END)
    ).to_numpy()

    prep = Preprocessor.fit(features, train_mask=train)
    x, wc = prep.transform(features)
    targets = encode_targets(features)

    def sliced(mask):
        return {key: value[torch.tensor(mask)] for key, value in targets.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    prep.save(OUT / "preprocess.json")

    y_val = targets["y_winner"][torch.tensor(val)].numpy()
    per_seed, winner_probs = [], []
    method_prob_sum = None
    for seed in SEEDS:
        net, info = train_one(
            seed, x[train], wc[train], sliced(train), x[val], wc[val], sliced(val),
            n_weight_classes=prep.n_weight_classes,
        )
        raw = predict(net, x[val], wc[val])
        temperature = fit_temperature(raw["winner_logits"], y_val)
        calibrated = predict(net, x[val], wc[val], temperature=temperature)
        per_seed.append(
            {
                "seed": seed,
                **info,
                "temperature": temperature,
                "val_log_loss_calibrated": round(
                    log_loss(y_val, calibrated["winner"]), 4
                ),
            }
        )
        winner_probs.append(calibrated["winner"])
        method_prob_sum = (
            calibrated["method"]
            if method_prob_sum is None
            else method_prob_sum + calibrated["method"]
        )
        torch.save(
            {"state_dict": net.state_dict(), "temperature": temperature,
             "n_features": x.shape[1], "n_weight_classes": prep.n_weight_classes},
            OUT / f"net_seed{seed}.pt",
        )

    ensemble = np.mean(winner_probs, axis=0)
    spread = np.max(winner_probs, axis=0) - np.min(winner_probs, axis=0)
    metrics = {
        "winner_ensemble": {
            "n_val": int(val.sum()),
            "accuracy": round(accuracy(y_val, ensemble), 4),
            "log_loss": round(log_loss(y_val, ensemble), 4),
            "brier": round(brier_score(y_val, ensemble), 4),
            "mean_seed_spread": round(float(spread.mean()), 4),
        },
        "per_seed": per_seed,
    }

    method_known = (targets["y_method"][torch.tensor(val)] >= 0).numpy()
    method_pred = [
        METHOD_CLASSES[i] for i in method_prob_sum[method_known].argmax(axis=1)
    ]
    method_true = [
        METHOD_CLASSES[i]
        for i in targets["y_method"][torch.tensor(val)][method_known].tolist()
    ]
    metrics["method_ensemble"] = {
        "n_val": int(method_known.sum()),
        "accuracy": round(
            float(np.mean([p == t for p, t in zip(method_pred, method_true)])), 4
        ),
        "macro_f1": round(macro_f1(method_true, method_pred), 4),
    }

    (OUT / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
