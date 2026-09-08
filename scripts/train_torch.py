"""Train the 5-seed MultiTaskNet ensemble and write metrics.

Two protocols, selected by the flags given (see scripts/train_xgb.py for the
same split in the XGBoost trainer):

* refit-through (DEFAULT; a bare ``python scripts/train_torch.py``, which is
  what the weekly refresh Action runs) -- fit the preprocessor and each seed
  on every decisive fight dated ``<= --refit-through`` (``latest`` = the
  newest fight in features.parquet) for exactly ``--budget`` epochs with no
  validation set, and stamp every seed with the ``--temperature`` the
  harness derived. There is no held-out slice, so the metrics file instead
  carries the pooled walk-forward numbers of the harness report given by
  ``--report`` (the experiment that chose the budget), under the same keys
  as the split layout, with nulls where no walk-forward equivalent exists.
  Torch is pinned to one intra-op thread in this mode, as the harness pins
  it, so the deployed checkpoints do not depend on the training host.
* split (``--train-end`` / ``--val-start`` / ``--val-end``; passing any one
  of them selects it) -- fit the preprocessor and each seed on ``date <
  --train-end``, early-stop on the ``[--val-start, --val-end]`` slice, fit
  each seed's temperature on that slice, and report the slice's ensemble
  metrics. This is the original pre-2021 / 2021-2023 recipe and what
  ``scripts/roll_window.py --execute`` drives with explicit dates.

Why refit is the default: the walk-forward harness (scripts/run_walkforward.py)
compared early-stopping on a held-out year against a fixed budget on all
data through the newest year, on the same 2018-2025 eval folds; the fixed
budget was not worse by more than the seed noise floor, and its pre-
registered rule then ships it (models/walkforward/refit_decision_b1.json,
``deployment_recipe: refit_through_latest``: B -0.0002 against A, inside
sigma_seed, and confirmed on fresh seeds 5-9). The deployed ensemble thereby
trains on ~5 more years of fights than the pre-2021 split. BUDGET,
TEMPERATURE, REPORT and REFIT_THROUGH below are that decision's numbers;
re-derive them via ``scripts/refit_decision.py --reports b1`` and
``run_walkforward.py --fixed-budget-from`` rather than editing them by hand.
The budget belongs to a FEATURE TABLE, not to the recipe: these are the SP2.2
shipped S1 table's numbers (refit_decision_b1.json); the SP2 `base,external`
table's -- 10 epochs at temperature 1.07 -- are kept in refit_decision_v3.json
and the SP1 46-column table's -- 14 epochs at 1.1 -- in refit_decision.json,
for the record.

This trains the BLEND's neural member. The XGBoost member is
``scripts/train_xgb.py`` and the two are combined by
``mma.inference.BlendedPredictor``; retraining one without the other leaves a
blend whose halves saw different data, which
``scripts/build_display_priors.py`` warns about.

Checkpoint payloads (state_dict, temperature, n_features, n_weight_classes)
are identical in both modes; ``mma.inference.Ensemble.load`` reads either.
The two flag families are mutually exclusive.
"""
from __future__ import annotations

import argparse
import json
import sys
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
DEFAULT_MODE = MODE_REFIT
# Refit-mode defaults: models/walkforward/refit_decision_b1.json -> torch.budget.
# NOTE these are the TORCH MEMBER's per-seed temperatures. The blend applies a
# SECOND, post-average temperature on top of them
# (`mma.inference.BLEND_TEMPERATURE`), which this script knows nothing about
# because it is a property of the two members combined, not of either one.
REFIT_THROUGH = "latest"
BUDGET = 6
TEMPERATURE = 1.15
REPORT = ROOT / "models" / "walkforward" / "torch_a1_combined_refit.json"


def parse_budget(spec) -> int:
    """Fixed epoch count from a CLI string (or int); must be positive."""
    if isinstance(spec, bool) or isinstance(spec, float):
        raise ValueError(f"budget must be a whole number of epochs, got {spec!r}")
    budget = int(spec)
    if budget <= 0:
        raise ValueError(f"budget must be a positive number of epochs, got {budget}")
    return budget


def refit_metrics(train_through: str, n_train: int, budget: int, temperature: float,
                  report_path: str, pooled: dict, per_seed: list[dict],
                  harness_features_max_date: str, harness_fold_years: list) -> dict:
    """metrics_val.json for the refit mode: the ensemble blocks are filled
    from the harness report's pooled walk-forward metrics (no held-out slice
    exists), null where the walk-forward has no equivalent.
    `harness_features_max_date` (the report's config.features_max_date) and
    `harness_fold_years` record which data the evidence was computed on, so
    a metrics file whose train_through has moved past the harness is
    detectably stale."""
    source = f"walk-forward pooled ({report_path})"
    return {
        "mode": MODE_REFIT,
        "train_through": train_through,
        "n_train": int(n_train),
        "budget": int(budget),
        "temperature": float(temperature),
        "harness_report": report_path,
        "harness_features_max_date": harness_features_max_date,
        "harness_fold_years": [int(year) for year in harness_fold_years],
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


def stale_harness_warning(train_through: str, harness_features_max_date: str) -> str | None:
    """Warning text when the refit trains on fights newer than the harness
    report ever saw, else None."""
    if pd.Timestamp(train_through) <= pd.Timestamp(harness_features_max_date):
        return None
    return (f"WARNING: harness evidence predates this training data (harness "
            f"features_max_date {harness_features_max_date} < train_through {train_through}); "
            "re-run scripts/run_walkforward.py to refresh")


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
    report = json.loads(Path(args.report).read_text())
    pooled = report["pooled"]
    harness_max_date = report["config"]["features_max_date"]
    train = (features["date"] <= cutoff).to_numpy()
    print(f"refit through {cutoff.date()}: n_train={int(train.sum())} "
          f"epochs={budget} temperature={temperature}")
    warning = stale_harness_warning(str(cutoff.date()), harness_max_date)
    if warning:
        print(warning, file=sys.stderr)

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
                         display_path(args.report), pooled, per_seed,
                         harness_max_date, report["fold_years"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-end", default=None, help=f"split mode (default {TRAIN_END})")
    parser.add_argument("--val-start", default=None, help=f"split mode (default {VAL_START})")
    parser.add_argument("--val-end", default=None, help=f"split mode (default {VAL_END})")
    parser.add_argument("--refit-through", default=None, metavar="DATE",
                        help=f"refit mode: train on every fight dated <= DATE ('latest' = newest fight; "
                             f"default {REFIT_THROUGH})")
    parser.add_argument("--budget", default=BUDGET, help=f"refit mode: epochs per seed (default {BUDGET})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"refit mode: per-seed temperature (default {TEMPERATURE})")
    parser.add_argument("--report", type=Path, default=REPORT,
                        help=f"refit mode: walk-forward report whose pooled metrics fill the metrics file "
                             f"(default {REPORT.relative_to(ROOT)})")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    mode = resolve_mode(args)
    if mode == MODE_SPLIT:
        args.train_end = args.train_end or TRAIN_END
        args.val_start = args.val_start or VAL_START
        args.val_end = args.val_end or VAL_END
    else:
        args.refit_through = args.refit_through or REFIT_THROUGH

    features = pd.read_parquet(PROCESSED / "features.parquet")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_split if mode == MODE_SPLIT else run_refit
    metrics = run(features, args, out_dir)

    (out_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
