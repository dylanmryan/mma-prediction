"""Train the XGBoost heads (winner, method, finish round) and write metrics.

Since SP2.2 each head is a FIVE-SEED ENSEMBLE, not a single fit: one model per
``random_state`` in ``SEEDS``, averaged over ``predict_proba``. XGBoost has no
seed ensemble of its own -- its stochasticity is ``subsample``/
``colsample_bytree`` under one seed -- and SP2.1 measured the price of scoring
a single fit at sigma 0.00087-0.00125 pooled winner log-loss, three times the
torch ensemble's 0.000346. Half of that noise would land in the deployed blend,
so the ensemble is required rather than optional; the artifacts are
``models/xgb_<head>_seed<seed>.json`` and ``mma.inference.BlendedPredictor``
loads all of them. This is exactly what ``mma.candidates.XGBCandidate(seeds=
(0, 1, 2, 3, 4))`` does in the harness, which is what B1 was measured with.

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
  metrics. This is the original pre-2021 / 2021-2023 recipe, and it is what
  the walk-forward harness's protocol A measures. Nothing deploys it: the
  script that used to drive it with explicit dates
  (``scripts/roll_window.py --execute``) is retired, and its aborts are
  half the reason why.

Why refit is the default: the walk-forward harness (scripts/run_walkforward.py)
compared early-stopping on a held-out year against a fixed budget on all
data through the newest year, on the same 2018-2025 eval folds; the fixed
budget was not worse by more than the seed noise floor, and its pre-
registered rule then ships it (models/walkforward/refit_decision_b1.json,
``deployment_recipe: refit_through_latest``: B -0.0007 against A, inside
sigma_seed). The deployed models thereby train on ~5 more years of fights than
the pre-2021 split. BUDGET, REPORT and REFIT_THROUGH below are that decision's
numbers; re-derive them via ``scripts/refit_decision.py --reports b1`` and
``run_walkforward.py --fixed-budget-from`` rather than editing them by hand.

The two flag families are mutually exclusive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
# The seed ensemble (SP2.2). Each seed is an xgboost ``random_state``; the head's
# prediction is the mean of the five models' ``predict_proba``. The same five
# seeds the torch ensemble uses and the same five ``mma.candidates.BlendCandidate``
# gave its XGB member when B1 was measured.
SEEDS = (0, 1, 2, 3, 4)

MODE_SPLIT, MODE_REFIT = "split", "refit_through"
DEFAULT_MODE = MODE_REFIT
# Refit-mode defaults: models/walkforward/refit_decision_b1.json -> xgb.budget.
# The budget belongs to a FEATURE TABLE *and* to a fit shape: these are the
# SP2.2 S1 table's numbers for the FIVE-SEED ensemble. The SP2 `base,external`
# single-fit numbers (105/61/75) stay in refit_decision_v3.json and the SP1
# 46-column table's (82/80/76) in refit_decision.json, for the record.
REFIT_THROUGH = "latest"
BUDGET = {"winner": 109, "method": 73, "round": 71}
REPORT = ROOT / "models" / "walkforward" / "xgb_ens5_s1_refit.json"


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


def refit_metrics(train_through: str, n_train: int, budget: dict, report_path: str, pooled: dict,
                  harness_features_max_date: str, harness_fold_years: list) -> dict:
    """xgb_metrics_val.json for the refit mode: the README's winner/method/
    finish_round blocks are filled from the harness report's pooled
    walk-forward metrics (no held-out slice exists), null where the
    walk-forward has no equivalent. `harness_features_max_date` (the
    report's config.features_max_date) and `harness_fold_years` record
    which data the evidence was computed on, so a metrics file whose
    train_through has moved past the harness is detectably stale."""
    source = f"walk-forward pooled ({report_path})"
    no_equivalent = "; accuracy and majority_baseline_accuracy have no walk-forward equivalent (null)"
    return {
        "mode": MODE_REFIT,
        "train_through": train_through,
        "n_train": int(n_train),
        "budget": dict(budget),
        "harness_report": report_path,
        "harness_features_max_date": harness_features_max_date,
        "harness_fold_years": [int(year) for year in harness_fold_years],
        "seeds": list(SEEDS),
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


def head_path(models_dir: Path, head: str, seed: int) -> Path:
    """models/xgb_<head>_seed<seed>.json -- one artifact per head per seed.

    `mma.versioning.MODEL_ARTIFACT_GLOBS` hashes exactly this shape, so the
    deployed model version changes when any member of the ensemble does.
    """
    return Path(models_dir) / f"xgb_{head}_seed{int(seed)}.json"


def mean_proba(models, x) -> "np.ndarray":
    """The seed ensemble's prediction: the mean of the members' predict_proba.

    Identical to `mma.candidates._mean_over_members`, including its "a single
    member is returned untouched" behaviour, so a one-seed deployment is
    bit-identical to that seed's own fit.
    """
    if len(models) == 1:
        return models[0].predict_proba(x)
    return np.mean([model.predict_proba(x) for model in models], axis=0)


def run_split(features: pd.DataFrame, x: pd.DataFrame, args, models_dir: Path) -> tuple[dict, list]:
    train = features["date"] < args.train_end
    val = (features["date"] >= args.val_start) & (features["date"] <= args.val_end)
    metrics = {}

    known = features["y_method"].notna()
    finish = features["y_finish_round"].notna()
    winners, methods, rounds = [], [], []
    for seed in SEEDS:
        params = {"random_state": int(seed)}
        winner = train_binary(x[train], features.loc[train, "y_winner"],
                              x[val], features.loc[val, "y_winner"], params=params)
        winner.save_model(head_path(models_dir, "winner", seed))
        winners.append(winner)

        method = train_multiclass(
            x[train & known], features.loc[train & known, "y_method"],
            x[val & known], features.loc[val & known, "y_method"],
            METHOD_CLASSES, params=params,
        )
        method.save_model(head_path(models_dir, "method", seed))
        methods.append(method)

        rounds_model = train_multiclass(
            x[train & finish], features.loc[train & finish, "y_finish_round"],
            x[val & finish], features.loc[val & finish, "y_finish_round"],
            ROUND_CLASSES, params=params,
        )
        rounds_model.save_model(head_path(models_dir, "round", seed))
        rounds.append(rounds_model)

    # winner -- the ensemble's mean probability, not any one member's
    y = features["y_winner"]
    p_val = mean_proba(winners, x[val])[:, 1]
    metrics["winner"] = {
        "n_val": int(val.sum()),
        "accuracy": round(accuracy(y[val], p_val), 4),
        "log_loss": round(log_loss(y[val], p_val), 4),
        "brier": round(brier_score(y[val], p_val), 4),
        # one per seed: early stopping is per member, so there is no single
        # "the" best iteration for an ensemble
        "best_iteration": [int(model.best_iteration) for model in winners],
    }

    # method (rows with known method)
    y = features["y_method"]
    pred = [METHOD_CLASSES[i] for i in mean_proba(methods, x[val & known]).argmax(axis=1)]
    truth = list(y[val & known])
    majority = y[train & known].mode()[0]
    metrics["method"] = {
        "n_val": int((val & known).sum()),
        "accuracy": round(float(sum(p == t for p, t in zip(pred, truth)) / len(truth)), 4),
        "macro_f1": round(macro_f1(truth, pred), 4),
        "majority_baseline_accuracy": round(float(sum(t == majority for t in truth) / len(truth)), 4),
    }

    # finish round (finishes only)
    y = features["y_finish_round"]
    pred = [ROUND_CLASSES[i] for i in mean_proba(rounds, x[val & finish]).argmax(axis=1)]
    truth = list(y[val & finish])
    majority = y[train & finish].mode()[0]
    metrics["finish_round"] = {
        "n_val": int((val & finish).sum()),
        "accuracy": round(float(sum(p == t for p, t in zip(pred, truth)) / len(truth)), 4),
        "macro_f1": round(macro_f1(truth, pred), 4),
        "majority_baseline_accuracy": round(float(sum(t == majority for t in truth) / len(truth)), 4),
    }
    metrics["seeds"] = list(SEEDS)
    return metrics, winners


def run_refit(features: pd.DataFrame, x: pd.DataFrame, args, models_dir: Path) -> tuple[dict, list]:
    cutoff = refit_cutoff(features, args.refit_through)
    budget = parse_budget(args.budget)
    report = json.loads(Path(args.report).read_text())
    pooled = report["pooled"]
    harness_max_date = report["config"]["features_max_date"]
    train = features["date"] <= cutoff
    print(f"refit through {cutoff.date()}: n_train={int(train.sum())} budget={budget} "
          f"seeds={list(SEEDS)}")
    warning = stale_harness_warning(str(cutoff.date()), harness_max_date)
    if warning:
        print(warning, file=sys.stderr)

    known = features["y_method"].notna()
    finish = features["y_finish_round"].notna()
    winners = []
    for seed in SEEDS:
        params = {"random_state": int(seed)}
        winner = train_binary(x[train], features.loc[train, "y_winner"], None, None,
                              params=params, fixed_rounds=budget["winner"])
        winner.save_model(head_path(models_dir, "winner", seed))
        winners.append(winner)

        method = train_multiclass(x[train & known], features.loc[train & known, "y_method"],
                                  None, None, METHOD_CLASSES, params=params,
                                  fixed_rounds=budget["method"])
        method.save_model(head_path(models_dir, "method", seed))

        rounds = train_multiclass(x[train & finish], features.loc[train & finish, "y_finish_round"],
                                  None, None, ROUND_CLASSES, params=params,
                                  fixed_rounds=budget["round"])
        rounds.save_model(head_path(models_dir, "round", seed))

    metrics = refit_metrics(str(cutoff.date()), int(train.sum()), budget,
                            display_path(args.report), pooled,
                            harness_max_date, report["fold_years"])
    return metrics, winners


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
    metrics, winners = run(features, x, args, models_dir)

    (models_dir / "xgb_metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # Averaged over the ensemble, because no single member is the model now.
    importances = pd.Series(
        np.mean([model.feature_importances_ for model in winners], axis=0),
        index=x.columns,
    ).sort_values(ascending=False)
    print(f"\ntop 15 winner-model features (mean over {len(winners)} seeds):")
    print(importances.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
