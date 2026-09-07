"""Train the XGBoost heads (winner, method, finish round) and write metrics.

Two protocols, selected by the flags given:

* refit-through (DEFAULT; a bare ``python scripts/train_xgb.py``, which is
  what the weekly refresh Action runs) -- train on every decisive fight dated
  ``<= --refit-through`` (``latest`` = the newest fight in features.parquet)
  for a fixed per-head round budget (``--budget``) with no validation set and
  no early stopping. There is no held-out slice in this mode, so the metrics
  file instead carries the pooled walk-forward numbers of the harness report
  given by ``--report`` (the experiment that chose the budget), under the
  same inner key names the README reads, with nulls where no walk-forward
  equivalent exists.
* split (``--train-end`` / ``--val-start`` / ``--val-end``; passing any one
  of them selects it) -- train on ``date < --train-end``, early-stop each
  head on the ``[--val-start, --val-end]`` slice, and report that slice's
  metrics. This is the original pre-2021 / 2021-2023 recipe and what
  ``scripts/roll_window.py --execute`` drives with explicit dates.

Why refit is the default: the walk-forward harness (scripts/run_walkforward.py)
compared early-stopping on a held-out year against a fixed budget on all
data through the newest year, on the same 2018-2025 eval folds; the fixed
budget was not worse by more than the seed noise floor, and its pre-
registered rule then ships it (models/walkforward/refit_decision.json,
``deployment_recipe: refit_through_latest``). The deployed models thereby
train on ~5 more years of fights than the pre-2021 split. BUDGET, REPORT and
REFIT_THROUGH below are that decision's numbers; re-derive them via
run_walkforward.py --fixed-budget-from rather than editing them by hand.

The two flag families are mutually exclusive.
"""
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
HEADS = ("winner", "method", "round")

MODE_SPLIT, MODE_REFIT = "split", "refit_through"
DEFAULT_MODE = MODE_REFIT
# Refit-mode defaults: models/walkforward/refit_decision.json -> xgb.budget.
REFIT_THROUGH = "latest"
BUDGET = {"winner": 82, "method": 80, "round": 76}
REPORT = ROOT / "models" / "walkforward" / "xgb_refit.json"


def parse_budget(spec) -> dict:
    """Per-head fixed round budget from a CLI string (or an int / dict).

    ``"81"`` applies 81 rounds to every head; ``'{"winner": 81, "method": 79,
    "round": 75}'`` sets each head. All three heads are required, no extras."""
    if isinstance(spec, str):
        spec = json.loads(spec)
    if isinstance(spec, bool) or not isinstance(spec, (int, dict)):
        raise ValueError(f"budget must be an int or a per-head dict, got {spec!r}")
    if isinstance(spec, int):
        spec = {head: spec for head in HEADS}
    missing = [head for head in HEADS if head not in spec]
    extra = sorted(set(spec) - set(HEADS))
    if missing or extra:
        raise ValueError(f"budget needs exactly the heads {list(HEADS)}; missing {missing}, extra {extra}")
    budget = {head: int(spec[head]) for head in HEADS}
    if any(rounds <= 0 for rounds in budget.values()):
        raise ValueError(f"every head budget must be positive, got {budget}")
    return budget


def refit_metrics(train_through: str, n_train: int, budget: dict, report_path: str, pooled: dict) -> dict:
    """xgb_metrics_val.json for the refit mode: the README's winner/method/
    finish_round blocks are filled from the harness report's pooled
    walk-forward metrics (no held-out slice exists), null where the
    walk-forward has no equivalent."""
    source = f"walk-forward pooled ({report_path})"
    no_equivalent = "; accuracy and majority_baseline_accuracy have no walk-forward equivalent (null)"
    return {
        "mode": MODE_REFIT,
        "train_through": train_through,
        "n_train": int(n_train),
        "budget": dict(budget),
        "harness_report": report_path,
        "walkforward_pooled": pooled,
        "winner": {
            "n_val": pooled["n"],
            "accuracy": pooled["accuracy"],
            "log_loss": pooled["winner_log_loss"],
            "brier": pooled["brier"],
            "best_iteration": budget["winner"],
            "source": source + "; best_iteration is the fixed budget, not an early-stopping result",
        },
        "method": {
            "n_val": pooled["n_method"],
            "accuracy": None,
            "macro_f1": pooled["method_macro_f1"],
            "majority_baseline_accuracy": None,
            "source": source + no_equivalent,
        },
        "finish_round": {
            "n_val": pooled["n_round"],
            "accuracy": None,
            "macro_f1": pooled["round_macro_f1"],
            "majority_baseline_accuracy": None,
            "source": source + no_equivalent,
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


def run_split(features: pd.DataFrame, x: pd.DataFrame, args, models_dir: Path) -> dict:
    train = features["date"] < args.train_end
    val = (features["date"] >= args.val_start) & (features["date"] <= args.val_end)
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
    return metrics, winner


def run_refit(features: pd.DataFrame, x: pd.DataFrame, args, models_dir: Path) -> dict:
    cutoff = refit_cutoff(features, args.refit_through)
    budget = parse_budget(args.budget)
    pooled = json.loads(Path(args.report).read_text())["pooled"]
    train = features["date"] <= cutoff
    print(f"refit through {cutoff.date()}: n_train={int(train.sum())} budget={budget}")

    y = features["y_winner"]
    winner = train_binary(x[train], y[train], None, None, fixed_rounds=budget["winner"])
    winner.save_model(models_dir / "xgb_winner.json")

    known = features["y_method"].notna()
    y = features["y_method"]
    method = train_multiclass(x[train & known], y[train & known], None, None,
                              METHOD_CLASSES, fixed_rounds=budget["method"])
    method.save_model(models_dir / "xgb_method.json")

    finish = features["y_finish_round"].notna()
    y = features["y_finish_round"]
    rounds = train_multiclass(x[train & finish], y[train & finish], None, None,
                              ROUND_CLASSES, fixed_rounds=budget["round"])
    rounds.save_model(models_dir / "xgb_round.json")

    metrics = refit_metrics(str(cutoff.date()), int(train.sum()), budget,
                            display_path(args.report), pooled)
    return metrics, winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-end", default=None, help=f"split mode (default {TRAIN_END})")
    parser.add_argument("--val-start", default=None, help=f"split mode (default {VAL_START})")
    parser.add_argument("--val-end", default=None, help=f"split mode (default {VAL_END})")
    parser.add_argument("--refit-through", default=None, metavar="DATE",
                        help=f"refit mode: train on every fight dated <= DATE ('latest' = newest fight; "
                             f"default {REFIT_THROUGH})")
    parser.add_argument("--budget", default=BUDGET,
                        help=f'refit mode: rounds per head as JSON or a single int (default {json.dumps(BUDGET)})')
    parser.add_argument("--report", type=Path, default=REPORT,
                        help=f"refit mode: walk-forward report whose pooled metrics fill the metrics file "
                             f"(default {REPORT.relative_to(ROOT)})")
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    args = parser.parse_args()
    mode = resolve_mode(args)
    if mode == MODE_SPLIT:
        args.train_end = args.train_end or TRAIN_END
        args.val_start = args.val_start or VAL_START
        args.val_end = args.val_end or VAL_END
    else:
        args.refit_through = args.refit_through or REFIT_THROUGH

    features = pd.read_parquet(PROCESSED / "features.parquet")
    x = feature_frame(features)
    models_dir = args.models_dir
    models_dir.mkdir(exist_ok=True, parents=True)

    run = run_split if mode == MODE_SPLIT else run_refit
    metrics, winner = run(features, x, args, models_dir)

    (models_dir / "xgb_metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    importances = pd.Series(
        winner.feature_importances_, index=x.columns
    ).sort_values(ascending=False)
    print("\ntop 15 winner-model features:")
    print(importances.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
