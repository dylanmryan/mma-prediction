"""Train and persist the fight simulator's two members (SP3 deployment).

The deployed scorer since SP3 is the HYBRID: the blend supplies P(A wins) and
the Monte Carlo simulator supplies P(method, round | winner). The simulator is
two XGBoost models, and this script is what puts them on disk:

* ``models/xgb_hazard_seed<seed>.json`` -- a 5-class model over
  ``mma.hazard.HAZARD_CLASSES`` fitted on the per-round hazard rows
  (``mma.hazard.build_hazard_rows``: one row per fight per round actually
  fought);
* ``models/xgb_decision_seed<seed>.json`` -- a binary model over the fights
  that went to the cards (``mma.hazard.build_decision_rows``).

Five seeds each, the same five ``scripts/train_xgb.py`` and
``scripts/train_torch.py`` use and the same five
``mma.candidates.HazardCandidate`` was measured with, averaged over
``predict_proba`` by ``mma.simulator.mean_over_seeds`` at serving time.

**Why a separate script rather than a fifth and sixth head in
``scripts/train_xgb.py``.** That script's two members are per-FIGHT models
over one model matrix; these two are fitted on frames a fight-level table
cannot express -- the hazard rows expand one fight into one row per round and
carry a ``round_no`` column, and both frames need ``fights.parquet`` for
``finish_round``, which the feature table only carries bucketed as '45'. Its
``--budget`` also validates exactly the three heads it owns, its split mode
has no simulator equivalent, and its metrics file is the one the README reads. Folding all of that into one script
would make both harder to read than keeping the simulator's training beside
the simulator's own modules.

**Refit-through only.** The simulator has no split-protocol form: it was never
measured in one, and the split-slice gate that once wanted one
(``scripts/roll_window.py``, retired) was explicitly not a promotion gate for
the served scorer. So the only mode here is the deployed
recipe -- train on every decisive fight dated ``<= --refit-through`` for a
fixed per-member round budget, with no validation set and no early stopping.

``BUDGET`` is the median-of-folds rule ``scripts/run_walkforward.py``'s
``fixed_budget_from`` applies, computed over ``REPORT``'s per-fold, per-seed
best iterations by ``budget_from_report`` below -- and
``tests/test_train_scripts.py`` asserts the constant still equals what that
function reads off the committed report, so the two cannot drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mma.hazard import HAZARD_CLASSES, build_decision_rows, build_hazard_rows
from mma.models.xgb import feature_frame, train_binary, train_multiclass

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
HEADS = ("hazard", "decision")
# The same five seeds every other deployed ensemble uses, and the five the
# hybrid was measured with.
SEEDS = (0, 1, 2, 3, 4)
MODE_REFIT = "refit_through"
REFIT_THROUGH = "latest"
# The walk-forward report the deployed simulator's evidence comes from: the
# hybrid at the deployed seeds (models/walkforward/sp3_decision.json).
REPORT = ROOT / "models" / "walkforward" / "hybrid_e2.json"
# median over folds of (median over seeds of best_iteration), + 1 -- see
# budget_from_report, and scripts/run_walkforward.fixed_budget_from, which is
# the same rule for the blend's members.
BUDGET = {"hazard": 138, "decision": 114}
METRICS = MODELS / "hazard_metrics.json"


def parse_budget(spec) -> dict:
    """Per-member fixed round budget from a CLI string (or an int / dict).

    ``"120"`` applies 120 rounds to both members; ``'{"hazard": 138,
    "decision": 114}'`` sets each. Both members are required, no extras --
    the same contract ``scripts/train_xgb.parse_budget`` applies to its
    three heads."""
    if isinstance(spec, str):
        spec = json.loads(spec)
    if isinstance(spec, bool) or not isinstance(spec, (int, dict)):
        raise ValueError(f"budget must be an int or a per-member dict, got {spec!r}")
    if isinstance(spec, int):
        spec = {head: spec for head in HEADS}
    missing = [head for head in HEADS if head not in spec]
    extra = sorted(set(spec) - set(HEADS))
    if missing or extra:
        raise ValueError(
            f"budget needs exactly the members {list(HEADS)}; missing {missing}, extra {extra}")
    budget = {head: int(spec[head]) for head in HEADS}
    if any(rounds <= 0 for rounds in budget.values()):
        raise ValueError(f"every member budget must be positive, got {budget}")
    return budget


def budget_from_report(report: dict) -> dict:
    """The fixed budget implied by a hybrid/hazard walk-forward report.

    ``scripts/run_walkforward.fixed_budget_from``'s rule, applied to the
    simulator's two members: xgboost's ``best_iteration`` is a 0-based index,
    so the budget is median(best_iteration) + 1 -- over seeds within a fold
    first (a five-seed ensemble has five of them), then over folds.
    """
    info = report["fit_info"]
    entries = info["hazard"] if "hazard" in info and isinstance(info["hazard"], list) else None
    if entries is None:
        raise ValueError(
            "report has no per-fold hazard fit_info; pass a hybrid or hazard "
            "walk-forward report")
    budget = {}
    for head in HEADS:
        per_fold = [float(np.median(fold[head]["best_iteration"])) for fold in entries]
        budget[head] = int(np.median(per_fold)) + 1
    return budget


def head_path(models_dir: Path, head: str, seed: int) -> Path:
    """models/xgb_<member>_seed<seed>.json -- one artifact per member per seed.

    ``mma.versioning.MODEL_ARTIFACT_GLOBS`` hashes exactly this shape, so the
    deployed model version changes when any member of the simulator does.
    """
    return Path(models_dir) / f"xgb_{head}_seed{int(seed)}.json"


def stale_harness_warning(train_through: str, harness_features_max_date: str) -> str | None:
    """Warning text when the refit trains on fights newer than the harness
    report ever saw, else None."""
    if pd.Timestamp(train_through) <= pd.Timestamp(harness_features_max_date):
        return None
    return (f"WARNING: harness evidence predates this training data (harness "
            f"features_max_date {harness_features_max_date} < train_through {train_through}); "
            "re-run scripts/run_walkforward.py to refresh")


def refit_cutoff(features: pd.DataFrame, spec: str) -> pd.Timestamp:
    """'latest' -> newest fight date; otherwise the ISO date given."""
    return pd.Timestamp(features["date"].max()) if spec == "latest" else pd.Timestamp(spec)


def display_path(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def refit_metrics(train_through: str, counts: dict, budget: dict, report_path: str,
                  pooled: dict, harness_features_max_date: str,
                  harness_fold_years: list) -> dict:
    """models/hazard_metrics.json: how these two members were fitted, and the
    walk-forward evidence behind the budget that fitted them.

    There is no held-out slice in refit mode, so the quality numbers are the
    HYBRID's pooled walk-forward metrics -- the simulator's contribution is a
    joint distribution, and the joint log-loss is the number the pre-registered
    bar was applied to. The winner block is deliberately absent: the simulator
    does not supply a winner, the blend does.
    """
    return {
        "mode": MODE_REFIT,
        "train_through": train_through,
        "n_train_fights": int(counts["fights"]),
        "n_hazard_rows": int(counts["hazard"]),
        "n_decision_rows": int(counts["decision"]),
        "budget": dict(budget),
        "harness_report": report_path,
        "harness_features_max_date": harness_features_max_date,
        "harness_fold_years": [int(year) for year in harness_fold_years],
        "seeds": list(SEEDS),
        "walkforward_pooled": pooled,
        "joint": {
            "n": pooled.get("n"),
            "joint_log_loss": pooled.get("joint_log_loss"),
            "method_macro_f1": pooled.get("method_macro_f1"),
            "round_macro_f1": pooled.get("round_macro_f1"),
            "zero_mass_cell_fraction": pooled.get("zero_mass_cell_fraction"),
            "source": (
                f"walk-forward pooled ({report_path}); these are the HYBRID's numbers -- "
                "the simulator supplies P(method, round | winner) and the blend supplies "
                "the winner, so neither member has a walk-forward number of its own"
            ),
        },
    }


def run_refit(features: pd.DataFrame, fights: pd.DataFrame, args,
              models_dir: Path) -> tuple[dict, dict]:
    """Fit both members on every fight through the cutoff; return (metrics, models)."""
    cutoff = refit_cutoff(features, args.refit_through)
    budget = parse_budget(args.budget)
    report = json.loads(Path(args.report).read_text())
    pooled = report["pooled"]
    harness_max_date = report["config"]["features_max_date"]

    train = features["date"] <= cutoff
    train_feats = features.loc[train].reset_index(drop=True)
    hazard_rows = build_hazard_rows(train_feats, fights)
    decision_rows = build_decision_rows(train_feats, fights)
    print(f"refit through {cutoff.date()}: n_fights={int(train.sum())} "
          f"hazard_rows={len(hazard_rows)} decision_rows={len(decision_rows)} "
          f"budget={budget} seeds={list(SEEDS)}")
    warning = stale_harness_warning(str(cutoff.date()), harness_max_date)
    if warning:
        print(warning, file=sys.stderr)

    x_hazard = feature_frame(hazard_rows.drop(columns=["hazard_label"]))
    x_decision = feature_frame(decision_rows.drop(columns=["decision_label"]))

    models = {"hazard": [], "decision": []}
    for seed in SEEDS:
        params = {"random_state": int(seed)}
        hazard = train_multiclass(x_hazard, hazard_rows["hazard_label"], None, None,
                                  HAZARD_CLASSES, params=params,
                                  fixed_rounds=budget["hazard"])
        hazard.save_model(head_path(models_dir, "hazard", seed))
        models["hazard"].append(hazard)

        decision = train_binary(x_decision, decision_rows["decision_label"], None, None,
                                params=params, fixed_rounds=budget["decision"])
        decision.save_model(head_path(models_dir, "decision", seed))
        models["decision"].append(decision)

    counts = {"fights": int(train.sum()), "hazard": len(hazard_rows),
              "decision": len(decision_rows)}
    metrics = refit_metrics(str(cutoff.date()), counts, budget, display_path(args.report),
                            pooled, harness_max_date, report["fold_years"])
    return metrics, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refit-through", default=REFIT_THROUGH, metavar="DATE",
                        help=f"train on every fight dated <= DATE ('latest' = newest fight; "
                             f"default {REFIT_THROUGH})")
    parser.add_argument("--budget", default=BUDGET,
                        help=f"rounds per member as JSON or a single int (default {json.dumps(BUDGET)})")
    parser.add_argument("--report", type=Path, default=REPORT,
                        help=f"walk-forward report whose pooled metrics fill the metrics file "
                             f"(default {REPORT.relative_to(ROOT)})")
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    parser.add_argument("--metrics-path", type=Path, default=None,
                        help=f"where to write the metrics file (default {METRICS.relative_to(ROOT)})")
    args = parser.parse_args()

    features = pd.read_parquet(PROCESSED / "features.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    models_dir = args.models_dir
    models_dir.mkdir(exist_ok=True, parents=True)

    metrics, models = run_refit(features, fights, args, models_dir)
    metrics_path = args.metrics_path or (models_dir / METRICS.name)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {len(models['hazard'])} hazard + {len(models['decision'])} decision "
          f"models to {display_path(models_dir)} and {display_path(metrics_path)}")


if __name__ == "__main__":
    main()
