"""Assemble the decision artifact for the external snapshot's decay.

    python scripts/external_decay_decision.py           # writes models/walkforward/external_decay_decision.json
    python scripts/external_decay_decision.py --print   # ... and echoes it

Reads only committed walk-forward reports, the committed feature table, the
noise floor and the pre-registration document -- paths are the constants below
-- so re-running it reproduces the artifact exactly. It writes no model and
deploys nothing.

The question, the candidates, the metric and the three-clause rule are fixed by
`docs/superpowers/plans/2026-09-09-external-decay-decision.md`, committed
before any candidate was run. Following the precedent set by
`scripts/sp3_decision.py`, this script does not retype that rule: it QUOTES the
plan's own rule block and READS THE THREE THRESHOLDS OUT OF THAT QUOTED TEXT
(`thresholds_from`), because a constant retyped next to a quote is exactly how
an artifact comes to state one rule and apply another. The sigma it reads is
then cross-checked against the measured seed noise floor.

Two things this script insists on rather than assumes:

* **The pairing.** Every candidate must differ from the incumbent K in exactly
  the columns its group names, and in nothing else -- same candidate, same
  seeds, same table, same fold years, same pooled `n`. `check_pairing` asserts
  it from each report's own `config`, so an unpaired comparison is an error
  rather than a plausible-looking number.
* **The joint metric's weights.** `pooled.joint_log_loss` is a mean over the
  SCORED rows (method known, and round known when the method is a finish), not
  over the fold's rows, so weighting the two recent folds by `n` would
  silently reweight them against each other. `scored_rows_per_fold` recomputes
  the fold masks from the committed feature table and counts the scored rows
  directly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.decay import RECENT_YEARS, coverage_by_year, removal_verdict, trailing_coverage  # noqa: E402
from mma.walkforward import make_folds  # noqa: E402

WF = ROOT / "models" / "walkforward"
OUT = WF / "external_decay_decision.json"
PLAN = ROOT / "docs" / "superpowers" / "plans" / "2026-09-09-external-decay-decision.md"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
NOISE_FLOOR = WF / "noise_floor.json"

#: The heading whose blockquote is the pre-registered rule.
RULE_HEADING = "## 5. The rule, stated mechanically"

#: The five leak-guard flags every candidate holds out, unchanged by this
#: decision. They stay in the TABLE so the external_missing slice keeps
#: working; see mma.tensors.DROPPED.
FLAGS = (
    "external_missing", "same_country",
    "notice_unknown", "home_country_a", "home_country_b",
)
#: The snapshot-dependent columns that ARE in the deployed model matrix,
#: grouped by the block that produces them (plan §2).
GROUPS = {
    "EXTERNAL": (
        "pre_ufc_wins_diff", "pre_ufc_losses_diff", "pre_ufc_finish_rate_diff",
        "pre_ufc_finish_loss_rate_diff", "pre_ufc_avg_opp_wins_diff",
        "days_since_pro_debut_diff",
    ),
    "NOTICE": (
        "notice_shortfall_days_diff", "missed_weight_over_lbs_diff",
        "short_notice_7_a", "short_notice_7_b",
        "short_notice_30_a", "short_notice_30_b",
        "missed_weight_a", "missed_weight_b",
    ),
    "CONTEXT": ("home_country_unknown",),
}
#: candidate id -> (report stem, the groups it removes)
CANDIDATES = {
    "R_ext": ("hybrid_drop_external", ("EXTERNAL",)),
    "R_notice": ("hybrid_drop_notice", ("NOTICE",)),
    "R_ctx": ("hybrid_drop_context", ("CONTEXT",)),
    "R_all": ("hybrid_drop_snapshot", ("EXTERNAL", "NOTICE", "CONTEXT")),
}
INCUMBENT = "hybrid_e2"
#: Fresh-seed confirmation (plan §6): the winning candidate and a paired K,
#: both at seeds 5-9.
FRESH_SUFFIX = "_seeds5"


def load(stem: str) -> dict:
    path = WF / f"{stem}.json"
    if not path.exists():
        raise SystemExit(f"missing walk-forward report {path}")
    return json.loads(path.read_text())


def rule_text(plan_path: Path = PLAN) -> str:
    """The plan's own rule block, quoted rather than paraphrased."""
    text = plan_path.read_text()
    start = text.index(RULE_HEADING)
    end = text.index("\n## ", start + len(RULE_HEADING))
    return text[start:end].strip()


def thresholds_from(rule: str) -> dict:
    """The three numbers the rule applies, read out of the rule's own text.

    Anchored on each clause's variable name, so a rule that changed a number
    without changing this script would change what the script applies.
    """
    patterns = {
        "joint_bar": r"`d_pooled_joint\s*<\s*([0-9.]+)`",
        "recent_bar": r"`keep_gain_recent\s*<\s*([0-9.]+)`",
        "sigma_seed": r"`d_pooled_winner\s*<=\s*([0-9.]+)`",
    }
    out = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, rule)
        if match is None:
            raise SystemExit(
                f"could not read {name} out of the plan's rule block; the rule and "
                "this script have drifted apart, which is the one thing that must "
                "not happen silently"
            )
        out[name] = float(match.group(1))
    return out


def expected_drops(groups) -> set:
    columns = set(FLAGS)
    for group in groups:
        columns |= set(GROUPS[group])
    return columns


def check_pairing(candidate: dict, incumbent: dict, groups) -> dict:
    """Assert the pair differs in exactly the named columns and nothing else."""
    cand_cfg, inc_cfg = candidate["config"], incumbent["config"]
    problems = []
    if set(inc_cfg["drop_columns"]) != set(FLAGS):
        problems.append(
            f"incumbent drops {sorted(inc_cfg['drop_columns'])}, expected the five "
            f"leak-guard flags {sorted(FLAGS)}"
        )
    want = expected_drops(groups)
    if set(cand_cfg["drop_columns"]) != want:
        problems.append(
            f"candidate drops {sorted(cand_cfg['drop_columns'])}, expected {sorted(want)}"
        )
    for key in ("candidate", "seeds", "n_feature_rows", "features_max_date",
                "train_start", "half_life", "fixed_budget_from"):
        if cand_cfg.get(key) != inc_cfg.get(key):
            problems.append(f"{key}: candidate {cand_cfg.get(key)!r} != incumbent {inc_cfg.get(key)!r}")
    if problems:
        raise SystemExit("unpaired comparison:\n  " + "\n  ".join(problems))
    return {
        "removed_columns": sorted(want - set(FLAGS)),
        "flags_held_out_by_both": sorted(FLAGS),
        "shared_config": {k: inc_cfg.get(k) for k in ("candidate", "seeds", "n_feature_rows",
                                                      "features_max_date")},
    }


def scored_rows_per_fold(features: pd.DataFrame) -> dict:
    """Rows the joint metric averages over, per fold year, and the fold `n`.

    A row is scored iff its method is known AND, when that method is a finish,
    its round is known -- exactly `mma.evaluate.joint_outcome_log_loss`'s two
    `continue` branches. Recomputed from the table rather than read off the
    report because the report records `n`, `n_method` and `n_round`, and the
    scored count is none of the three.
    """
    method, rounds = features["y_method"], features["y_finish_round"]
    scored = method.notna() & ((method == "decision") | rounds.notna())
    joint, winner = {}, {}
    for fold in make_folds(features["date"]):
        joint[str(fold.year)] = int(scored.to_numpy()[fold.eval].sum())
        winner[str(fold.year)] = int(fold.eval.sum())
    return {"joint": joint, "winner": winner}


def slice_row(report: dict, name: str) -> dict | None:
    block = (report.get("slices") or {}).get(name)
    if block is None:
        return None
    return {key: block[key] for key in ("n", "winner_log_loss", "joint_log_loss", "ece")
            if key in block}


def build(*, print_it: bool = False) -> dict:
    rule = rule_text()
    thresholds = thresholds_from(rule)
    measured_sigma = json.loads(NOISE_FLOOR.read_text())["sigma_seed"]
    if abs(thresholds["sigma_seed"] - measured_sigma) > 1e-6:
        raise SystemExit(
            f"the plan's winner tolerance ({thresholds['sigma_seed']}) is not the "
            f"measured seed noise floor ({measured_sigma}) in {NOISE_FLOOR}"
        )

    features = pd.read_parquet(FEATURES)
    weights = scored_rows_per_fold(features)
    incumbent = load(INCUMBENT)

    results = {}
    for cid, (stem, groups) in CANDIDATES.items():
        candidate = load(stem)
        pairing = check_pairing(candidate, incumbent, groups)
        verdict = removal_verdict(
            candidate, incumbent,
            joint_weights=weights["joint"], winner_weights=weights["winner"],
            recent_years=RECENT_YEARS,
            joint_bar=thresholds["joint_bar"], recent_bar=thresholds["recent_bar"],
            sigma_seed=thresholds["sigma_seed"],
        )
        results[cid] = {
            "groups": list(groups),
            "report": f"models/walkforward/{stem}.json",
            "pairing": pairing,
            **verdict,
            "external_missing_slice": {
                "candidate": slice_row(candidate, "external_missing"),
                "incumbent": slice_row(incumbent, "external_missing"),
            },
        }

    per_group = {group: results[cid]["verdict"]
                 for cid, (_, groups) in CANDIDATES.items()
                 if len(groups) == 1 for group in groups}
    to_remove = sorted(g for g, v in per_group.items() if v == "REMOVE")
    ambiguous = sorted(g for g, v in per_group.items() if v == "AMBIGUOUS")

    fresh = fresh_seed_block(thresholds, weights)

    return {
        "experiment": "external snapshot decay -- does the block still earn its place?",
        "pre_registration": str(PLAN.relative_to(ROOT)),
        "generated_by": "scripts/external_decay_decision.py",
        "rule": {
            "quoted_from_the_plan": rule,
            "thresholds_read_from_that_text": thresholds,
            "measured_sigma_seed": measured_sigma,
            "recent_years": list(RECENT_YEARS),
        },
        "column_groups": {name: list(columns) for name, columns in GROUPS.items()},
        "flags_held_out_by_every_candidate": list(FLAGS),
        "fold_weights": weights,
        "coverage": coverage_block(features),
        "incumbent": {
            "name": INCUMBENT,
            "report": f"models/walkforward/{INCUMBENT}.json",
            "pooled": incumbent["pooled"],
        },
        "candidates": results,
        "per_group_verdict": per_group,
        "groups_to_remove": to_remove,
        "ambiguous_groups": ambiguous,
        "consolidated_verdict": results["R_all"]["verdict"],
        "fresh_seed_confirmation": fresh,
        "decided": bool(fresh.get("decided")),
        "decision": decision_block(per_group, to_remove, ambiguous, results, fresh),
    }


def coverage_block(features: pd.DataFrame) -> dict:
    dates, missing = features["date"], features["external_missing"]
    return {
        "flag": "external_missing",
        "by_year": coverage_by_year(dates, missing),
        "trailing_12m": trailing_coverage(dates, missing, months=12),
        "overall_share": round(float(missing.astype(bool).mean()), 4),
        "weekly_check": "scripts/check_snapshot_coverage.py",
    }


def fresh_seed_block(thresholds: dict, weights: dict) -> dict:
    """The mandatory seeds 5-9 confirmation, when both of its reports exist.

    Absent either report this returns `decided: false` with the reason, which
    is what stops the artifact from reading as a decision before the
    confirmation the plan requires has been run.
    """
    stems = {cid: f"{stem}{FRESH_SUFFIX}" for cid, (stem, _) in CANDIDATES.items()}
    incumbent_stem = f"{INCUMBENT}{FRESH_SUFFIX}"
    available = {cid: stem for cid, stem in stems.items() if (WF / f"{stem}.json").exists()}
    if not (WF / f"{incumbent_stem}.json").exists() or not available:
        return {
            "decided": False,
            "reason": (
                "no fresh-seed pair on disk yet: the plan requires the winning "
                f"candidate and a paired incumbent both at seeds 5-9 "
                f"({incumbent_stem} plus one of {sorted(stems.values())})"
            ),
        }
    incumbent = load(incumbent_stem)
    out = {"decided": True, "incumbent": incumbent_stem, "seeds": incumbent["config"]["seeds"],
           "candidates": {}}
    for cid, stem in available.items():
        candidate = load(stem)
        groups = CANDIDATES[cid][1]
        check_pairing(candidate, incumbent, groups)
        out["candidates"][cid] = {
            "report": f"models/walkforward/{stem}.json",
            **removal_verdict(
                candidate, incumbent,
                joint_weights=weights["joint"], winner_weights=weights["winner"],
                recent_years=RECENT_YEARS,
                joint_bar=thresholds["joint_bar"], recent_bar=thresholds["recent_bar"],
                sigma_seed=thresholds["sigma_seed"],
            ),
        }
    return out


def decision_block(per_group, to_remove, ambiguous, results, fresh) -> dict:
    if ambiguous:
        outcome = "AMBIGUOUS -- stop and report"
    elif not to_remove:
        outcome = "keep every snapshot-dependent column"
    else:
        outcome = f"remove {', '.join(to_remove)}"
    confirmed = None
    if fresh.get("decided"):
        seeds05 = {g: per_group[g] for g in per_group}
        fresh_by_group = {}
        for cid, block in fresh["candidates"].items():
            for group in CANDIDATES[cid][1]:
                if len(CANDIDATES[cid][1]) == 1:
                    fresh_by_group[group] = block["verdict"]
        confirmed = {
            "seeds_0_4": seeds05,
            "seeds_5_9": fresh_by_group,
            "agree": all(fresh_by_group.get(g) == v for g, v in seeds05.items()
                         if g in fresh_by_group),
        }
    return {
        "outcome": outcome,
        "per_group": per_group,
        "consolidated": results["R_all"]["verdict"],
        "fresh_seed_agreement": confirmed,
        "taken_by": "the pre-registered rule, applied mechanically by this script",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--print", dest="echo", action="store_true")
    args = parser.parse_args(argv)

    artifact = build()
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"\nrule thresholds (read from the plan): {artifact['rule']['thresholds_read_from_that_text']}")
    print(f"coverage, trailing 12m: {artifact['coverage']['trailing_12m']['share']}")
    print("\ncandidate                 d_joint   keep_gain_recent   d_winner   verdict")
    for cid, block in artifact["candidates"].items():
        print(f"  {cid:<22} {block['pooled']['d_joint']:>+9.6f} "
              f"{block['recent']['keep_gain_recent']:>+15.6f} "
              f"{block['pooled']['d_winner']:>+11.6f}   {block['verdict']}")
    print(f"\nper-group verdict: {artifact['per_group_verdict']}")
    print(f"decision: {artifact['decision']['outcome']}")
    if not artifact["decided"]:
        print(f"NOT YET DECIDED: {artifact['fresh_seed_confirmation']['reason']}")
    if args.echo:
        print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
