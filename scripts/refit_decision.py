"""Build models/walkforward/refit_decision.json from the A/B walk-forward
reports and the cached seed noise floor.

For each candidate (xgb, torch), compares the pre-2021-split-plus-early-
stopping report ("A") against the fixed-budget-on-all-data report ("B", from
--fixed-budget-from A) on the same 2018-2025(+2026) walk-forward folds. Two
independent checks are reported:

* ``B_not_worse_than_A_by_sigma`` -- the actual deployment-recipe gate (spec
  v3 SS4 SP1): B ships as the refit recipe iff its pooled winner log-loss is
  not worse than A's by more than sigma_seed (the seed noise floor).
* ``bar_check_B_vs_A`` -- the unrelated "ships as a challenger" bar from the
  same spec section (pooled improvement > max(0.003, 2*sigma_seed) and no
  fold year regresses by more than 0.01), via `mma.walkforward.bar_check`. B
  is not a challenger here (it does not need to beat A, only not lose to
  it), so ``ships`` is expected to be False even when the deployment recipe
  is adopted; this block is kept for visibility, not as the decision rule.

Also builds ``fresh_seed_rescore``: a fresh-seed (5-9) re-run of the same B
vs A comparison for the torch candidate, confirming the deployment-recipe
gate is not a seed-lucky fluke of the original seed-0-4 comparison.

The comparison is only valid within one FEATURE TABLE: a budget derived from
reports computed on the v1 46-column table says nothing about the table the
model is actually deployed on. ``--reports`` therefore names a report set
(``REPORT_SETS`` below), each recording the four (plus two fresh-seed)
reports and the table they were computed on, and each writing its own
decision file.

Usage:
    python scripts/refit_decision.py                    # the v1 table's set
    python scripts/refit_decision.py --reports v3       # the shipped SP2 table
    python scripts/refit_decision.py --reports v3 --out /tmp/decision.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.walkforward import bar_check, paired_delta  # noqa: E402

WF = ROOT / "models" / "walkforward"
NOISE_FLOOR = WF / "noise_floor.json"

RULE = "B ships as the deployment recipe iff torch B pooled winner log-loss <= torch A + sigma_seed (spec v3 §4 SP1)"
DEPLOYMENT_RECIPE = "refit_through_latest"
SPLIT_RECIPE = "split_protocol_a"

# One entry per feature table the decision has been taken on. `table` is
# documentation only -- what makes a set coherent is that all six reports were
# produced by `run_walkforward.py` on the same `features.parquet` with the same
# `--drop-columns`, which each report records in its own `config` block.
REPORT_SETS = {
    "v1": {
        "table": "the SP1 46-column base table",
        "out": WF / "refit_decision.json",
        "reports": {
            "xgb": {"A": WF / "xgb_v1.json", "B": WF / "xgb_refit.json"},
            "torch": {"A": WF / "torch_v1.json", "B": WF / "torch_refit.json"},
        },
        "fresh": {"A": WF / "torch_v1_seeds5.json", "B": WF / "torch_refit_seeds5.json"},
        "note": (
            "follow-up experiment (not a re-decision): per-fold optimal temperature drifts from "
            "~1.3–1.65 (2018–2022) to ~0.9–1.0 (2023–2025) across all three seed "
            "sets, while the recipe applies the all-fold median 1.1; B is worse by ~+0.0024 on "
            "2023–2025 and better on 2020–2022. Test a budget/temperature taken from the "
            "most recent k folds through the same harness against torch_v1."
        ),
    },
    "v3": {
        "table": (
            "the SP2 shipped 54-column base,external table with external_missing "
            "and same_country held out of both model matrices"
        ),
        "out": WF / "refit_decision_v3.json",
        "reports": {
            "xgb": {"A": WF / "xgb_v3.json", "B": WF / "xgb_v3_refit.json"},
            "torch": {"A": WF / "torch_v3.json", "B": WF / "torch_v3_refit.json"},
        },
        "fresh": {
            "A": WF / "torch_external_diffsonly_seeds5.json",
            "B": WF / "torch_v3_refit_seeds5.json",
        },
        "note": (
            "this set re-derives the deployment budget on the SP2 feature table; the v1 "
            "budget (14 epochs / T 1.1; 82/80/76 trees) was derived from a different table "
            "and does not carry over (models/walkforward/refit_decision.json)"
        ),
    },
}


def sigma_ci_95(sigma: float, n_reports: int) -> list[float]:
    """95% CI for a sample std computed from ``n_reports`` draws, via the
    chi-square distribution on (n_reports - 1) degrees of freedom. For the
    n=3 case here, df=2, whose chi-square quantile has the closed form
    -2*ln(1-p) (chi-square with 2 dof is Exponential(scale=2))."""
    df = n_reports - 1
    chi2_upper = -2.0 * math.log(1.0 - 0.975)  # 97.5th percentile, df=2
    chi2_lower = -2.0 * math.log(1.0 - 0.025)  # 2.5th percentile, df=2
    lower = sigma * math.sqrt(df / chi2_upper)
    upper = sigma * math.sqrt(df / chi2_lower)
    return [lower, upper]


def candidate_decision(report_a: dict, report_b: dict, sigma_seed: float) -> dict:
    result = paired_delta(report_a, report_b, sigma_seed)
    result["bar_check_B_vs_A"] = bar_check(report_b, report_a, sigma_seed)
    result["budget"] = report_b["config"]["budget"]
    return result


def build_decision(report_set: dict, noise_floor: dict) -> dict:
    """The decision document for one report set. Pure apart from reading the
    reports the set names, so a caller can diff it against the committed file."""
    sigma_seed = noise_floor["sigma_seed"]
    lower, upper = sigma_ci_95(sigma_seed, noise_floor["n_reports"])

    decision = {"rule": RULE, "feature_table": report_set["table"]}
    for name, paths in report_set["reports"].items():
        report_a = json.loads(paths["A"].read_text())
        report_b = json.loads(paths["B"].read_text())
        decision[name] = candidate_decision(report_a, report_b, sigma_seed)
    # The rule is on torch, because torch is the deployed scorer; when B loses
    # to A by more than sigma_seed the recipe stays protocol A rather than
    # this file recording a recipe its own numbers do not support.
    decision["deployment_recipe"] = (
        DEPLOYMENT_RECIPE if decision["torch"]["B_not_worse_than_A_by_sigma"]
        else SPLIT_RECIPE
    )

    # Fresh-seed re-scoring (spec §4 SP1): re-run the shipped torch B vs A
    # comparison on disjoint seeds 5-9 and confirm the gate still holds --
    # does not change deployment_recipe automatically even if it fails.
    fresh_a = json.loads(report_set["fresh"]["A"].read_text())
    fresh_b = json.loads(report_set["fresh"]["B"].read_text())
    fresh_seed_rescore = paired_delta(fresh_a, fresh_b, sigma_seed)
    fresh_seed_rescore["verdict"] = (
        "confirmed" if fresh_seed_rescore["B_not_worse_than_A_by_sigma"]
        else "NOT confirmed — see notes"
    )
    decision["fresh_seed_rescore"] = fresh_seed_rescore

    decision["notes"] = [
        f"sigma_seed is an n={noise_floor['n_reports']} estimate; its 95% CI (chi-square, "
        f"{noise_floor['n_reports'] - 1} dof) is roughly [0.5σ, 6.3σ] "
        f"(for σ={sigma_seed:.6f}: lower ≈ σ·sqrt(2/7.378) = {lower:.6f}, "
        f"upper ≈ σ·sqrt(2/0.0506) = {upper:.6f})",
        "the torch rule passed by a margin smaller than the 4-dp storage precision of pooled "
        f"log-loss (Δ={decision['torch']['delta_B_minus_A']:+.4f} vs σ={sigma_seed:.5f})",
        "bar_check is informational here: ships=False is expected because B is a deployment "
        "recipe that must only be not-worse than A, not a challenger that must beat it",
        report_set["note"],
        "Fresh-seed re-scoring (spec §4 SP1): B and A both re-run with seeds 5–9; "
        "see fresh_seed_rescore.",
    ]
    decision["sigma_ci_95"] = [lower, upper]
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", choices=sorted(REPORT_SETS), default="v1",
                        help="which report set (and therefore which feature table) to decide on")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: the report set's own decision file)")
    args = parser.parse_args()

    report_set = REPORT_SETS[args.reports]
    decision = build_decision(report_set, json.loads(NOISE_FLOOR.read_text()))

    out = args.out or report_set["out"]
    out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
