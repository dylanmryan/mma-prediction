"""Train the 5-seed MultiTaskNet ensemble and write metrics.

Two protocols, selected by the flags given (see scripts/train_xgb.py for the
same split in the XGBoost trainer):

* split -- fit the preprocessor and each seed on ``date < --train-end``,
  early-stop on the ``[--val-start, --val-end]`` slice, fit each seed's
  temperature on that slice, and report the slice's ensemble metrics in
  ``metrics_val.json``. This is the original pre-2021 / 2021-2023 recipe and
  what ``scripts/roll_window.py --execute`` drives with explicit dates.
* ``--refit-through DATE`` -- fit the preprocessor and each seed on every
  decisive fight dated ``<= DATE`` (``latest`` = the newest fight in
  features.parquet) for exactly ``--budget`` epochs with no validation set,
  and stamp every seed with the ``--temperature`` the harness derived. There
  is no held-out slice, so the metrics file instead carries the pooled
  walk-forward numbers of the harness report given by ``--report`` (the
  experiment that chose the budget), under the same keys as the split
  layout, with nulls where no walk-forward equivalent exists. Torch is
  pinned to one intra-op thread in this mode, as the harness pins it, so
  the deployed checkpoints do not depend on the training host.

Checkpoint payloads (state_dict, temperature, n_features, n_weight_classes)
are identical in both modes; ``mma.inference.Ensemble.load`` reads either.
The two flag families are mutually exclusive. With neither, DEFAULT_MODE
applies.
"""
from __future__ import annotations

import argparse
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

MODE_SPLIT, MODE_REFIT = "split", "refit_through"
DEFAULT_MODE = MODE_SPLIT
REFIT_THROUGH = "latest"
TEMPERATURE = 1.0


def parse_budget(spec) -> int:
    """Fixed epoch count from a CLI string (or int); must be positive."""
    if isinstance(spec, bool) or isinstance(spec, float):
        raise ValueError(f"budget must be a whole number of epochs, got {spec!r}")
    budget = int(spec)
    if budget <= 0:
        raise ValueError(f"budget must be a positive number of epochs, got {budget}")
    return budget


def refit_metrics(train_through: str, n_train: int, budget: int, temperature: float,
                  report_path: str, pooled: dict, per_seed: list[dict]) -> dict:
    """metrics_val.json for the refit mode: the ensemble blocks are filled
    from the harness report's pooled walk-forward metrics (no held-out slice
    exists), null where the walk-forward has no equivalent."""
    source = f"walk-forward pooled ({report_path})"
    return {
        "mode": MODE_REFIT,
        "train_through": train_through,
        "n_train": int(n_train),
        "budget": int(budget),
        "temperature": float(temperature),
        "harness_report": report_path,
        "walkforward_pooled": pooled,
        "winner_ensemble": {
            "n_val": pooled["n"],
            "accuracy": pooled["accuracy"],
            "log_loss": pooled["winner_log_loss"],
            "brier": pooled["brier"],
            "mean_seed_spread": None,
            "source": source + "; mean_seed_spread has no walk-forward equivalent (null)",
        },
        "per_seed": [
            {
                "seed": int(entry["seed"]),
                "best_epoch": int(entry["best_epoch"]),
                "best_val_log_loss": None,
                "temperature": float(entry["temperature"]),
                "epochs_run": int(entry["epochs_run"]),
            }
            for entry in per_seed
        ],
        "method_ensemble": {
            "n_val": pooled["n_method"],
            "accuracy": None,
            "macro_f1": pooled["method_macro_f1"],
            "source": source + "; accuracy has no walk-forward equivalent (null)",
        },
    }


def resolve_mode(args) -> str:
    split_given = any(v is not None for v in (args.train_end, args.val_start, args.val_end))
    if args.refit_through is not None and split_given:
        raise SystemExit("--refit-through cannot be combined with --train-end/--val-start/--val-end")
    if args.refit_through is not None:
        return MODE_REFIT
    if split_given:
        return MODE_SPLIT
    return DEFAULT_MODE


def refit_cutoff(features: pd.DataFrame, spec: str) -> pd.Timestamp:
    """'latest' -> newest fight date; otherwise the ISO date given."""
    return pd.Timestamp(features["date"].max()) if spec == "latest" else pd.Timestamp(spec)


def display_path(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def save_checkpoint(net, temperature: float, n_features: int, n_weight_classes: int, path: Path) -> None:
    torch.save(
        {"state_dict": net.state_dict(), "temperature": temperature,
         "n_features": n_features, "n_weight_classes": n_weight_classes},
        path,
    )


def run_split(features: pd.DataFrame, args, out_dir: Path) -> dict:
    train = (features["date"] < args.train_end).to_numpy()
    val = (
        (features["date"] >= args.val_start) & (features["date"] <= args.val_end)
    ).to_numpy()

    prep = Preprocessor.fit(features, train_mask=train)
    x, wc = prep.transform(features)
    targets = encode_targets(features)

    def sliced(mask):
        return {key: value[torch.tensor(mask)] for key, value in targets.items()}

    prep.save(out_dir / "preprocess.json")

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
        save_checkpoint(net, temperature, x.shape[1], prep.n_weight_classes,
                        out_dir / f"net_seed{seed}.pt")

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
    return metrics


def run_refit(features: pd.DataFrame, args, out_dir: Path) -> dict:
    torch.set_num_threads(1)  # as the harness that chose the budget does
    cutoff = refit_cutoff(features, args.refit_through)
    budget = parse_budget(args.budget)
    temperature = float(args.temperature)
    pooled = json.loads(Path(args.report).read_text())["pooled"]
    train = (features["date"] <= cutoff).to_numpy()
    print(f"refit through {cutoff.date()}: n_train={int(train.sum())} "
          f"epochs={budget} temperature={temperature}")

    prep = Preprocessor.fit(features, train_mask=train)
    x, wc = prep.transform(features)
    targets = encode_targets(features)
    train_targets = {key: value[torch.tensor(train)] for key, value in targets.items()}
    prep.save(out_dir / "preprocess.json")

    per_seed = []
    for seed in SEEDS:
        net, info = train_one(
            seed, x[train], wc[train], train_targets, None, None, None,
            n_weight_classes=prep.n_weight_classes, fixed_epochs=budget,
        )
        per_seed.append({"seed": seed, **info, "temperature": temperature})
        save_checkpoint(net, temperature, x.shape[1], prep.n_weight_classes,
                        out_dir / f"net_seed{seed}.pt")

    return refit_metrics(str(cutoff.date()), int(train.sum()), budget, temperature,
                         display_path(args.report), pooled, per_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-end", default=None, help=f"split mode (default {TRAIN_END})")
    parser.add_argument("--val-start", default=None, help=f"split mode (default {VAL_START})")
    parser.add_argument("--val-end", default=None, help=f"split mode (default {VAL_END})")
    parser.add_argument("--refit-through", default=None, metavar="DATE",
                        help="refit mode: train on every fight dated <= DATE ('latest' = newest fight)")
    parser.add_argument("--budget", default=None, help="refit mode: epochs per seed, e.g. 14")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"refit mode: per-seed temperature (default {TEMPERATURE})")
    parser.add_argument("--report", type=Path, default=None,
                        help="refit mode: walk-forward report whose pooled metrics fill the metrics file")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    mode = resolve_mode(args)
    if mode == MODE_SPLIT:
        args.train_end = args.train_end or TRAIN_END
        args.val_start = args.val_start or VAL_START
        args.val_end = args.val_end or VAL_END
    else:
        args.refit_through = args.refit_through or REFIT_THROUGH
        if args.budget is None or args.report is None:
            raise SystemExit("--refit-through needs --budget and --report")

    features = pd.read_parquet(PROCESSED / "features.parquet")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_split if mode == MODE_SPLIT else run_refit
    metrics = run(features, args, out_dir)

    (out_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
