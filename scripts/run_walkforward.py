"""Score one candidate on the walk-forward harness and write a JSON report.

Examples:
  python scripts/run_walkforward.py --candidate elo --name elo
  python scripts/run_walkforward.py --candidate xgb --name xgb_v1
  python scripts/run_walkforward.py --candidate torch --name torch_v1 --seeds 0,1,2,3,4
  python scripts/run_walkforward.py --candidate torch --name torch_v1_seeds5 --seeds 5,6,7,8,9
  python scripts/run_walkforward.py --candidate xgb --name xgb_hl4 --half-life 4 --train-start 2005-01-01
  python scripts/run_walkforward.py --candidate torch --name torch_refit --fixed-budget-from models/walkforward/torch_v1.json
  python scripts/run_walkforward.py --candidate torch --name torch_refit_recent --fixed-epochs 18 --temperature 0.98
  python scripts/run_walkforward.py --candidate torch --name torch_v1_extslice --drop-columns external_missing,same_country
  python scripts/run_walkforward.py --candidate xgb --name xgb_v1_seed1 --model-seed 1
  python scripts/run_walkforward.py --candidate xgb --name xgb_ens5 --seeds 0,1,2,3,4
  python scripts/run_walkforward.py --candidate blend --name blend_b0 --seeds 0,1,2,3,4 --blend-weight 0.5

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

from mma.candidates import BlendCandidate, EloCandidate, TorchCandidate, XGBCandidate  # noqa: E402
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES  # noqa: E402
from mma.walkforward import build_report, make_folds, recency_weights  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models" / "walkforward"
# The ensemble the torch candidate has always run, and the one SP2.2's blend
# gives both of its members.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
SEEDED = ("torch", "blend")


def fixed_budget_from(report: dict) -> dict:
    """Median fit budget (and temperature) across the folds of a reference report.

    ``best_iteration`` is either a per-fold int (old single-budget XGB reports)
    or a per-fold dict of per-head best_iteration (winner/method/round); the
    latter yields a per-head median dict in ``fixed_rounds``.
    """
    # xgboost's best_iteration is a 0-based index (early stopping at
    # best_iteration=81 means 82 trees were kept), so the fixed round count
    # is median(best_iteration) + 1 -- matching the torch epoch rule below.
    info = report["fit_info"]
    budget = {}
    if "best_iteration" in info:
        entries = info["best_iteration"]
        if entries and isinstance(entries[0], dict):
            budget["fixed_rounds"] = {
                head: int(np.median([entry[head] for entry in entries])) + 1
                for head in ("winner", "method", "round")
            }
        else:
            budget["fixed_rounds"] = int(np.median(entries)) + 1
    if "best_epoch" in info:
        per_fold = [float(np.median(e)) if isinstance(e, list) else float(e) for e in info["best_epoch"]]
        budget["fixed_epochs"] = int(np.median(per_fold)) + 1
    if "temperature" in info:
        per_fold = [float(np.median(t)) if isinstance(t, list) else float(t) for t in info["temperature"]]
        budget["temperature"] = round(float(np.median(per_fold)), 2)
    return budget


def resolve_budget(args: argparse.Namespace) -> dict | None:
    """The fit budget for this run, or None to early-stop on the inner val.

    Either derived from a reference report (--fixed-budget-from) or stated
    outright (--fixed-epochs/--temperature, torch only); the two sources are
    mutually exclusive. A temperature without a budget is a usage error --
    outside fixed-budget mode the temperature is fit on the inner val.
    """
    explicit = args.fixed_epochs is not None or args.temperature is not None
    if args.fixed_budget_from is not None:
        if explicit:
            raise SystemExit(
                "--fixed-budget-from is mutually exclusive with --fixed-epochs/--temperature"
            )
        return fixed_budget_from(json.loads(args.fixed_budget_from.read_text()))
    if not explicit:
        return None
    if args.candidate != "torch":
        raise SystemExit(
            f"--fixed-epochs/--temperature apply to the torch candidate only (got {args.candidate!r})"
        )
    if args.fixed_epochs is None:
        raise SystemExit(
            "--temperature applies to fixed-budget mode only; pass --fixed-epochs too "
            "(without it the temperature is fit on the inner validation year)"
        )
    if args.fixed_epochs < 1:
        raise SystemExit(f"--fixed-epochs must be at least 1 (got {args.fixed_epochs})")
    budget = {"fixed_epochs": int(args.fixed_epochs)}
    if args.temperature is not None:
        if not args.temperature > 0:
            raise SystemExit(f"--temperature must be positive (got {args.temperature})")
        budget["temperature"] = float(args.temperature)
    return budget


def resolve_drop_columns(args: argparse.Namespace) -> tuple[str, ...]:
    """Feature columns to hold out of the model matrix, in the order given.

    The columns stay in the feature table -- only the learner stops seeing
    them -- so `walkforward.slice_masks` still reports a slice keyed on a
    dropped column. That is the point: it makes a *paired* incumbent
    computable on the same table as the candidate, which is the only way a
    slice delta means "what the block did to those rows" rather than "how
    easy those rows are". Names are deduplicated in order; whitespace and
    empty entries are ignored, but a flag that names nothing is a usage
    error rather than a silent no-op.
    """
    if args.drop_columns is None:
        return ()
    names: list[str] = []
    for name in args.drop_columns.split(","):
        name = name.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise SystemExit(f"--drop-columns names no columns (got {args.drop_columns!r})")
    if args.candidate == "elo":
        raise SystemExit(
            "--drop-columns applies to the fitted candidates only; the elo "
            "candidate reads elo_diff directly and fits nothing"
        )
    return tuple(names)


def apply_model_seed(args: argparse.Namespace, config: dict) -> dict:
    """Fold ``--model-seed`` into the XGB param override as ``random_state``.

    XGBoost has no seed ensemble the way the torch candidate does -- its
    stochasticity lives in ``subsample``/``colsample_bytree`` under a single
    ``random_state`` (0 in ``mma.models.xgb.BASE_PARAMS``). Re-running with a
    different value is therefore the XGB analogue of torch's fresh seed set,
    and is what the SP2.1 fresh-seed confirmation needs. Torch already takes
    ``--seeds``, so pointing this flag at it would be two seed knobs for one
    ensemble; the elo candidate fits nothing. Both are usage errors rather
    than silent no-ops. Returns a new dict; ``config`` is left alone.
    """
    if args.model_seed is None:
        return dict(config)
    if args.candidate != "xgb":
        raise SystemExit(
            f"--model-seed applies to the xgb candidate only (got {args.candidate!r}); "
            "the torch ensemble is seeded with --seeds"
        )
    if "random_state" in config:
        raise SystemExit(
            "--model-seed and --config-json both set random_state; pass one of them"
        )
    return {**config, "random_state": int(args.model_seed)}


def check_drop_columns(drop_columns, features: pd.DataFrame) -> None:
    """Fail loudly on a column that is not in the table.

    A typo would otherwise drop nothing and the run would silently be an
    ordinary one wearing an ablation's name.
    """
    unknown = [name for name in drop_columns if name not in features.columns]
    if unknown:
        raise SystemExit(
            f"--drop-columns names column(s) absent from the feature table: {unknown}"
        )


def resolve_seeds(args: argparse.Namespace) -> tuple[int, ...] | None:
    """The seed ensemble for this run, or None for a single-fit candidate.

    torch and blend default to ``DEFAULT_SEEDS``. XGBoost does NOT: every
    committed XGB report is a single fit at ``mma.models.xgb.BASE_PARAMS``'
    ``random_state``, and defaulting the new seed-ensembling path on would
    silently change what ``--candidate xgb`` means. It becomes an ensemble only
    when ``--seeds`` names one, and ``--seeds 0`` is bit-identical to the
    single fit. ``--model-seed`` is the other way to set the XGB seed, so the
    two together are a usage error rather than a silent winner. The elo floor
    fits nothing.
    """
    if args.candidate == "xgb" and args.seeds is not None and args.model_seed is not None:
        raise SystemExit(
            "--seeds and --model-seed both set the xgb random_state; pass one of them "
            "(--seeds for a seed ensemble, --model-seed for one fit at one seed)"
        )
    if args.seeds is None:
        return DEFAULT_SEEDS if args.candidate in SEEDED else None
    if args.candidate == "elo":
        raise SystemExit(
            "--seeds applies to the fitted candidates only; the elo candidate "
            "reads elo_diff directly and fits nothing"
        )
    seeds: list[int] = []
    for token in args.seeds.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value not in seeds:
            seeds.append(value)
    if not seeds:
        raise SystemExit(f"--seeds names no seeds (got {args.seeds!r})")
    return tuple(seeds)


def resolve_blend(args: argparse.Namespace) -> dict | None:
    """The blend's fixed weight and calibration switch, or None for other candidates.

    The weight is on the XGB member and is FIXED at 0.5 by SP2.2's
    pre-registration; the flag exists so the 0.3/0.7 flatness diagnostic and
    the uncalibrated diagnostic can be run and reported, not so a better cell
    can be shipped.
    """
    if args.candidate != "blend":
        if args.blend_weight is not None:
            raise SystemExit(
                f"--blend-weight applies to the blend candidate only (got {args.candidate!r})"
            )
        if args.no_blend_calibration:
            raise SystemExit(
                f"--no-blend-calibration applies to the blend candidate only (got {args.candidate!r})"
            )
        return None
    weight = 0.5 if args.blend_weight is None else float(args.blend_weight)
    if not 0.0 <= weight <= 1.0:
        raise SystemExit(f"--blend-weight must be in [0, 1] (got {args.blend_weight})")
    return {"blend_weight": weight, "blend_calibrated": not args.no_blend_calibration}


def build_candidate(kind: str, name: str, seeds, config: dict, budget: dict | None,
                    drop_columns: tuple[str, ...] = (), blend: dict | None = None):
    """``budget`` is None in early-stopping mode; otherwise the dict from
    resolve_budget, which must carry the key this learner consumes
    (fixed_rounds for xgb, fixed_epochs for torch) -- a budget derived from
    the wrong learner's report, or an elo candidate, is a usage error."""
    if kind == "elo":
        if budget is not None:
            raise SystemExit("--fixed-budget-from does not apply to the elo candidate (nothing is fit)")
        return EloCandidate()  # resolve_drop_columns already rejected --drop-columns here
    if kind == "blend":
        if budget is not None:
            raise SystemExit(
                "fixed-budget mode does not apply to the blend candidate: its post-average "
                "temperature is fitted on the inner-validation year, which a fixed budget "
                "trains on"
            )
        if config:
            raise SystemExit(
                "--config-json does not apply to the blend candidate: it has two members and "
                "the flag cannot say which one it configures; both run their deployed configs"
            )
        return BlendCandidate(name=name, seeds=tuple(seeds), weight=blend["blend_weight"],
                              calibrate=blend["blend_calibrated"], drop_columns=drop_columns)
    needed = "fixed_rounds" if kind == "xgb" else "fixed_epochs"
    if budget is not None and needed not in budget:
        raise SystemExit(
            f"--fixed-budget-from report has no {needed!r} budget for the {kind} candidate "
            f"(found {sorted(budget) or 'nothing'}); point it at a {kind} walk-forward report"
        )
    budget = budget or {}
    if kind == "xgb":
        return XGBCandidate(name=name, params=config, fixed_rounds=budget.get("fixed_rounds"),
                            drop_columns=drop_columns,
                            seeds=None if seeds is None else tuple(seeds))
    return TorchCandidate(name=name, seeds=tuple(seeds), config=config,
                         fixed_epochs=budget.get("fixed_epochs"), temperature=budget.get("temperature"),
                         drop_columns=drop_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", choices=["elo", "xgb", "torch", "blend"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", default=None,
                        help="seed ensemble (default 0,1,2,3,4 for torch and blend; xgb is a "
                             "single fit unless this names seeds)")
    parser.add_argument("--config-json", default=None, help="JSON dict of XGB params / torch config")
    parser.add_argument("--train-start", default=None, help="drop training fights before this date")
    parser.add_argument("--half-life", type=float, default=None, help="recency half-life in years")
    parser.add_argument("--fixed-budget-from", type=Path, default=None,
                        help="reference report; train on train+inner_val with its median budget, no early stopping")
    parser.add_argument("--fixed-epochs", type=int, default=None,
                        help="torch fit budget stated outright (mutually exclusive with --fixed-budget-from)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="calibration temperature for --fixed-epochs mode (default 1.0)")
    parser.add_argument("--model-seed", type=int, default=None,
                        help="xgb random_state (subsample/colsample draws); the XGB analogue "
                             "of torch's --seeds, used for fresh-seed re-scoring")
    parser.add_argument("--blend-weight", type=float, default=None,
                        help="blend candidate: weight on the XGB member (default 0.5, the "
                             "pre-registered value; 0.3/0.7 are a flatness diagnostic only)")
    parser.add_argument("--no-blend-calibration", action="store_true",
                        help="blend candidate: skip the post-average temperature (diagnostic)")
    parser.add_argument("--drop-columns", default=None,
                        help="comma-separated feature columns to hold out of the model matrix "
                             "(they stay in the table, so slices keyed on them still report)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    features = (
        pd.read_parquet(PROCESSED / "features.parquet")
        .sort_values("date", kind="stable").reset_index(drop=True)
    )
    config = apply_model_seed(args, json.loads(args.config_json) if args.config_json else {})
    budget = resolve_budget(args)
    drop_columns = resolve_drop_columns(args)
    check_drop_columns(drop_columns, features)
    seeds = resolve_seeds(args)
    blend = resolve_blend(args)
    candidate = build_candidate(args.candidate, args.name, seeds, config, budget, drop_columns, blend)
    budget = budget or {}  # report shape: always a dict

    fold_results = []
    started = time.time()
    for fold in make_folds(features["date"], train_start=args.train_start):
        weights = recency_weights(features["date"], fold.eval_start, args.half_life) if args.half_life else None
        pred, info = candidate.fit_predict(features, fold, weights)
        fold_results.append((fold.year, fold.eval, pred, info))
        print(f"fold {fold.year}: n_train={int(fold.train.sum())} n_eval={int(fold.eval.sum())} info={info}")

    run_config = {
        "candidate": args.candidate,
        "seeds": None if seeds is None else ",".join(str(s) for s in seeds),
        "model_seed": args.model_seed,
        "config": config, "train_start": args.train_start, "half_life": args.half_life,
        "fixed_budget_from": str(args.fixed_budget_from) if args.fixed_budget_from else None,
        "budget": budget, "drop_columns": list(drop_columns),
        "n_feature_rows": int(len(features)),
        "features_max_date": str(features["date"].max().date()),
        "runtime_sec": round(time.time() - started, 1),
    }
    if blend is not None:  # only the blend carries these, so other reports keep their shape
        run_config.update(blend)
    report = build_report(args.name, run_config, features, fold_results, METHOD_CLASSES, ROUND_CLASSES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pooled": report["pooled"],
                      "folds": {y: f["winner_log_loss"] for y, f in report["folds"].items()}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
