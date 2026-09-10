"""Re-validate the DEPLOYED recipe on the current feature table.

    python scripts/revalidate_recipe.py --check-staleness   # cheap: is the evidence stale?
    python scripts/revalidate_recipe.py --plan              # print the runs, fit nothing
    python scripts/revalidate_recipe.py                     # the real thing (~15 min)

This is what replaced `scripts/roll_window.py`'s promotion gate. That gate
asked "should a fresh fit be promoted over the incumbent?", which is not a
question this project asks any more: the deployed recipe is refit-through-
latest and the weekly Action already retrains every member whenever new
fights arrive, so there is no candidate and no incumbent to choose between.
Its own docstring records why it could not answer the question it was built
for -- read it, it is kept as the account of a real lesson.

The question that IS open is the one this script answers. The deployed
recipe -- feature blocks, held-out columns, blend weight and temperature,
fit budgets, the hybrid composition -- was justified by walk-forward
evidence computed on a table that ends 2026-08-08. The models now train
through 2026-09-05 and the table keeps growing, so `stale_harness_warning`
in the train scripts fires and nothing re-checks that the recipe still
clears its bars on the data it is actually deployed on.

What it does:

1. Reads the deployed configuration out of the artifacts rather than
   retyping it: each run's `config` block comes from the committed report
   that justified it, the feature blocks from `data/processed/
   features_blocks.json`, and the deployed hash from
   `mma.versioning.model_version`.
2. Re-runs, on today's table, the deployed hybrid and every paired
   comparison its bars need -- the SP3 paired incumbent (the same blend
   through the simulator's cell scorer) and both members' refit A/B pairs.
   Reports land in `models/walkforward/revalidation/`; the committed
   reports that the original decisions rest on are never overwritten.
3. Applies the bars ALREADY RECORDED in the committed decision artifacts,
   reading every threshold out of those artifacts the way
   `scripts/external_decay_decision.py` reads its rule out of its plan. A
   bar this script can no longer find is a loud failure, because a bar that
   drifts from the artifact that states it is worth nothing.
4. Writes `models/walkforward/recipe_revalidation.json` and exits NON-ZERO
   if a gated bar is no longer met -- that is a finding a human must act on,
   not a line in a log.

It deploys nothing, promotes nothing and stages nothing. A bar that no
longer clears is reported, and what to do about it is a human call.

`--check-staleness` is the cheap mode the weekly Action runs: it reads the
committed metrics files and the table, reports how far the deployed models'
training data has run ahead of the harness evidence, and fits nothing. It also
reports whether `models/market_benchmark_oof.json` can still be regenerated --
that benchmark is rebuilt from a walk-forward prediction dump joined to the
pooled rows POSITIONALLY, so every refresh that grows the table retires the
previous dump, and nothing else runs that script on a schedule. Both reads are
warnings: the committed artifacts still describe the tables they were computed
on. The full re-validation is minutes of fitting and writes a decision
artifact, so it stays human-triggered.

The hybrid run in the batch below writes that dump (`--dump-predictions`),
which is what keeps the benchmark regenerable without ever refreshing
`models/walkforward/hybrid_e2.json` in place.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mma.feature_blocks import table_blocks  # noqa: E402
from mma.versioning import model_version  # noqa: E402
from mma.walkforward import paired_delta  # noqa: E402
from scripts.build_odds_benchmark import (  # noqa: E402
    OOF_BENCHMARK, oof_source_status, pooled_row_count,
)
from scripts.sp3_decision import simulator_bar_check  # noqa: E402

WF = ROOT / "models" / "walkforward"
REVAL_DIR = WF / "revalidation"
OUT = WF / "recipe_revalidation.json"
FEATURES = ROOT / "data" / "processed" / "features.parquet"

#: The committed decision artifacts whose bars this script re-applies. It
#: reads the thresholds out of them rather than retyping the numbers, so a
#: bar edited without touching this script changes what the script applies.
SP3_DECISION = WF / "sp3_decision.json"
REFIT_DECISION = WF / "refit_decision_b1.json"
SP2_2_DECISION = WF / "sp2_2_decision.json"
NOISE_FLOOR = WF / "noise_floor.json"

#: The deployed members' metrics files, which record both the date their
#: training data reaches and the date the harness evidence behind them does.
DEPLOYED_METRICS = {
    "torch": ROOT / "models" / "torch" / "metrics_val.json",
    "xgb": ROOT / "models" / "xgb_metrics_val.json",
    "hazard": ROOT / "models" / "hazard_metrics.json",
}


@dataclass(frozen=True)
class Run:
    """One walk-forward run the re-validation needs, and where its recipe
    comes from.

    `source` is the COMMITTED report whose `config` block states the run --
    the deployed recipe is what those reports measured, so reading the
    configuration back out of them is the only way a re-validation run is
    the same experiment rather than a plausible-looking neighbour.
    `budget_from` names the run within this batch whose FRESH report supplies
    a fixed budget: a refit budget derived on the 2026-08-08 table is a budget
    for that table, so re-validating against it would re-validate half the
    question.
    `dump` asks the run to write its pooled out-of-fold predictions as well as
    its report. Only the hybrid does: `scripts/build_odds_benchmark.py --mode
    oof` joins a dump to the pooled walk-forward rows POSITIONALLY, so it needs
    one written from the current table, and this batch is already re-running
    the deployed hybrid on exactly that table. A dump nobody reads is a
    megabyte of JSON per run, so the others do not write one.
    """
    key: str
    source: Path
    role: str
    budget_from: str | None = None
    dump: bool = False


RUNS = (
    Run("hybrid", WF / "hybrid_e2.json",
        "the deployed scorer: the blend's winner, the simulator's shape",
        dump=True),
    Run("blend_cells", WF / "blend_b1_cells.json",
        "SP3's paired incumbent -- the same blend read through the cell scorer"),
    Run("torch_A", WF / "torch_a1_combined.json",
        "refit rule, torch: protocol A (split, early-stopped)"),
    Run("torch_B", WF / "torch_a1_combined_refit.json",
        "refit rule, torch: protocol B (the deployed refit recipe)",
        budget_from="torch_A"),
    Run("xgb_A", WF / "xgb_ens5_s1.json",
        "refit rule, xgb: protocol A (split, early-stopped)"),
    Run("xgb_B", WF / "xgb_ens5_s1_refit.json",
        "refit rule, xgb: protocol B (the deployed refit recipe)",
        budget_from="xgb_A"),
)
RUNS_BY_KEY = {run.key: run for run in RUNS}


def load(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing artifact {path}")
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    return str(Path(path).relative_to(ROOT))


# --- the deployed configuration, read out of the artifacts -------------------


def run_args(config: dict, name: str, out_dir, budget_from: str | None = None) -> list[str]:
    """`scripts/run_walkforward.py` arguments that reproduce `config`.

    Every flag comes from the committed report's own run config, so the
    re-validation run is the same experiment on a newer table rather than a
    hand-rewritten approximation of it. The one field deliberately NOT copied
    is `fixed_budget_from`: a budget derived on the table the committed report
    ran on is a budget for that table, so it is redirected to this batch's
    fresh A report (`budget_from`) and a config that names one with no
    redirection is a usage error rather than a silently stale run.
    """
    args = ["--candidate", config["candidate"], "--name", name,
            "--out-dir", str(out_dir)]
    if config.get("seeds"):
        args += ["--seeds", config["seeds"]]
    if config.get("model_seed") is not None:
        args += ["--model-seed", str(config["model_seed"])]
    if config.get("config"):
        args += ["--config-json", json.dumps(config["config"])]
    if config.get("train_start"):
        args += ["--train-start", config["train_start"]]
    if config.get("half_life") is not None:
        args += ["--half-life", str(config["half_life"])]
    if config.get("drop_columns"):
        args += ["--drop-columns", ",".join(config["drop_columns"])]
    if config.get("blend_weight") is not None:
        args += ["--blend-weight", str(config["blend_weight"])]
    if config.get("blend_calibrator"):
        args += ["--blend-calibrator", config["blend_calibrator"]]
    if config.get("blend_calibrated") is False:
        args.append("--no-blend-calibration")
    if config.get("blend_joint_cells"):
        args.append("--blend-joint-cells")
    if config.get("hazard_calibrated"):
        args.append("--hazard-calibrate")
    if config.get("fixed_budget_from"):
        if budget_from is None:
            raise SystemExit(
                f"{name}: the committed report sets fixed_budget_from "
                f"({config['fixed_budget_from']}), which is a budget for the table "
                "that report ran on. Re-validating against it would re-validate the "
                "old table's budget; this run needs a fresh A report to derive from."
            )
        args += ["--fixed-budget-from", str(budget_from)]
    elif budget_from is not None:
        raise SystemExit(
            f"{name}: a fresh budget source was given but the committed report sets "
            "no fixed_budget_from, so this run does not take a budget at all"
        )
    return args


def deployed_configuration() -> dict:
    """The deployed recipe as the artifacts state it.

    Nothing here is retyped. The held-out columns and each member's run
    configuration come from the committed reports; the feature blocks come
    from the sidecar `scripts/build_features.py` writes next to the table it
    built (which is also what the serving path reads, so it is the deployment
    contract rather than a description of one); the hash comes from
    `mma.versioning.model_version` over the artifacts that actually score.
    """
    configs = {run.key: load(run.source)["config"] for run in RUNS}
    drops = {run.key: tuple(configs[run.key]["drop_columns"]) for run in RUNS}
    if len(set(drops.values())) != 1:
        raise SystemExit(
            "the committed reports disagree about which columns are held out of the "
            f"model matrix, so there is no single deployed recipe to re-validate: {drops}"
        )
    return {
        "feature_blocks": list(table_blocks()),
        "drop_columns": list(next(iter(drops.values()))),
        "deployed_hash": model_version(ROOT),
        "runs": {
            run.key: {
                "role": run.role,
                "recipe_read_from": _rel(run.source),
                "recipe_was_measured_on": configs[run.key].get("features_max_date"),
                "budget_from": run.budget_from,
            }
            for run in RUNS
        },
        "blend": {k: v for k, v in load(ROOT / "models" / "blend.json").items()
                  if k in ("weight", "temperature")},
        "simulator": {k: v for k, v in load(ROOT / "models" / "simulator.json").items()
                      if k in ("n_runs", "alpha", "sim_seed", "default_rounds")},
    }


# --- the bars, read out of the committed decision artifacts ------------------


def sp3_thresholds(artifact: dict) -> dict:
    """SP3's joint bar and winner tolerance, out of `sp3_decision.json`.

    That artifact already read both out of the pre-registration's own quoted
    text (`scripts/sp3_decision.py`'s `joint_bar_from` / `sigma_from`), so
    reading them back out of the artifact keeps one chain of custody from the
    plan to the number this script applies. A missing key is a loud failure:
    a re-validation that silently fell back to a retyped 0.01 would be
    applying a bar nothing states.
    """
    bar = artifact.get("bar") or {}
    for key in ("joint_bar", "applied_at"):
        if bar.get(key) is None:
            raise SystemExit(
                f"{_rel(SP3_DECISION)} has no bar.{key}; the recorded joint_bar and "
                "sigma_seed are what this script re-applies and it will not invent them"
            )
    return {
        "joint_bar": float(bar["joint_bar"]),
        "sigma_seed": float(bar["applied_at"]),
        "quoted_from_the_plan": bar.get("quoted_from_the_plan"),
        "form": bar.get("form"),
        "read_from": _rel(SP3_DECISION),
    }


def refit_rule(artifact: dict) -> dict:
    """The deployment-recipe rule, out of `refit_decision_b1.json`.

    The rule text is the artifact's own `rule` field and the tolerance is the
    sigma the artifact applied. `gated_member` is torch because that is what
    the recorded rule gates on ("the rule is on torch, because torch is the
    deployed scorer", `scripts/refit_decision.py`); the XGB member's A/B is
    re-run and reported alongside, but gating on it here would be a stricter
    rule than the one recorded.
    """
    rule = artifact.get("rule")
    if not rule:
        raise SystemExit(
            f"{_rel(REFIT_DECISION)} has no rule field; the deployment-recipe rule is "
            "what this script re-applies and it will not retype it"
        )
    sigma = (artifact.get("torch") or {}).get("sigma_seed")
    if sigma is None:
        raise SystemExit(f"{_rel(REFIT_DECISION)} has no torch.sigma_seed")
    return {
        "text": rule,
        "sigma_seed": float(sigma),
        "gated_member": "torch",
        "read_from": _rel(REFIT_DECISION),
        "recorded_deployment_recipe": artifact.get("deployment_recipe"),
    }


def ece_gate(artifact: dict) -> dict:
    """The blend's calibration gate, out of `sp2_2_decision.json`.

    Rule 4 as amended compares means over three disjoint seed sets against a
    2 sigma tolerance on the pooled spread. Three seed sets per arm is six
    walk-forward runs this script does not make, so what it re-applies is the
    gate's own THRESHOLD -- the incumbent mean plus that tolerance, the
    largest pooled ECE the gate passes -- which is exactly the form the
    artifact's own sub-note used to place the shipped scorer against it.
    A single seed set is a weaker read of the same gate and is recorded as
    one; it is not a different bar.
    """
    amended = (artifact.get("ece_gate") or {}).get("amended") or {}
    for key in ("incumbent_mean", "tolerance_2sigma", "form"):
        if amended.get(key) is None:
            raise SystemExit(
                f"{_rel(SP2_2_DECISION)} has no ece_gate.amended.{key}; the recorded "
                "calibration gate is what this script re-applies"
            )
    mean = float(amended["incumbent_mean"])
    tolerance = float(amended["tolerance_2sigma"])
    return {
        "form": amended["form"],
        "incumbent_mean": mean,
        "tolerance_2sigma": tolerance,
        "threshold": mean + tolerance,
        "read_from": _rel(SP2_2_DECISION),
        "weaker_read_than_the_gate": (
            "the gate averages three disjoint seed sets per arm; this re-validation "
            "measures one seed set (0-4) against the gate's own largest passing "
            "pooled ECE (incumbent_mean + tolerance_2sigma), which is the form the "
            "artifact's own sub-note used to place the shipped scorer"
        ),
        "evaluated_on": (
            "the HARNESS form -- one temperature fitted per fold on that fold's "
            "inner-validation year -- which is the form every pooled ECE the gate "
            "compared was measured in. The deployed scorer applies one fixed "
            "post-average temperature; see sp2_2_decision.json's deployed_form."
        ),
    }


def check_sigma_agreement(sp3_sigma: float, refit_sigma: float, measured: float) -> None:
    """Both recorded bars must quote the seed noise floor that was measured.

    They are stored to different precisions -- SP3's is the plan's 6-dp quote,
    the refit artifact's is the floor at full precision -- so agreement is
    asserted at 6 dp. A disagreement means at least one bar is applying a
    tolerance nobody measured, which is the one thing that must not happen
    quietly.
    """
    for label, value in (("sp3_decision", sp3_sigma), ("refit_decision", refit_sigma)):
        if round(float(value), 6) != round(float(measured), 6):
            raise SystemExit(
                f"{label} applies sigma_seed {value} but {_rel(NOISE_FLOOR)} measured "
                f"{measured}; a bar and the floor it came from disagree"
            )


# --- the bars, applied to the fresh reports ---------------------------------


def check_sp3_bar(hybrid: dict, incumbent: dict, thresholds: dict) -> dict:
    """SP3's two-clause bar, re-applied on the current table.

    The arithmetic is `scripts.sp3_decision.simulator_bar_check` itself, not a
    second implementation of it: a re-validation that applied its own version
    of the bar could differ from the artifact it re-validates in exactly the
    way it exists to detect.
    """
    out = simulator_bar_check(hybrid, incumbent,
                              thresholds["sigma_seed"], thresholds["joint_bar"])
    return {
        "bar": "SP3's simulator bar: joint beats the paired incumbent by more than "
               "joint_bar, and the winner marginal is not worse by more than sigma_seed",
        "gated": True,
        "thresholds": {k: thresholds[k] for k in ("joint_bar", "sigma_seed")},
        "thresholds_read_from": thresholds.get("read_from"),
        **out,
        "still_clears": bool(out["ships"]),
    }


def check_refit_rule(report_a: dict, report_b: dict, rule: dict,
                     member: str = "torch") -> dict:
    """The deployment-recipe rule, re-applied on the current table.

    `mma.walkforward.paired_delta` is the same arithmetic
    `scripts/refit_decision.py` writes the committed artifact with, so the
    number here is comparable to the one recorded there by construction.
    """
    out = paired_delta(report_a, report_b, rule["sigma_seed"])
    return {
        "bar": rule["text"],
        "member": member,
        "gated": member == rule["gated_member"],
        "gated_note": (
            "the recorded rule gates on torch; the xgb member's A/B is re-run and "
            "reported, but gating on it would be a stricter rule than the one recorded"
        ),
        "thresholds": {"sigma_seed": rule["sigma_seed"]},
        "thresholds_read_from": rule.get("read_from"),
        "delta_B_minus_A": out["delta_B_minus_A"],
        "per_fold_B_minus_A": out["per_fold_B_minus_A"],
        "A_pooled": out["A_pooled"],
        "B_pooled": out["B_pooled"],
        "still_clears": bool(out["B_not_worse_than_A_by_sigma"]),
    }


def check_ece_gate(blend_report: dict, gate: dict) -> dict:
    """The blend's calibration gate, re-applied on the current table."""
    ece = float(blend_report["pooled"]["ece"])
    return {
        "bar": gate["form"],
        "gated": True,
        "thresholds": {"threshold": gate["threshold"],
                       "incumbent_mean": gate["incumbent_mean"],
                       "tolerance_2sigma": gate["tolerance_2sigma"]},
        "thresholds_read_from": gate.get("read_from"),
        "weaker_read_than_the_gate": gate.get("weaker_read_than_the_gate"),
        "evaluated_on": gate.get("evaluated_on"),
        "pooled_ece": ece,
        "headroom": round(gate["threshold"] - ece, 6),
        "still_clears": bool(ece <= gate["threshold"]),
    }


#: Each margin the re-validation re-measures, and whether the bar it reports
#: to is cleared by going DOWN (a log-loss delta, an ECE) or by going UP.
#: `margin_movement` needs the direction to say which way a change moved the
#: recipe relative to its bar; getting it backwards would report a
#: deterioration as headroom.
LOWER_IS_BETTER = {
    "sp3_joint_delta": True,
    "sp3_winner_delta": True,
    "refit_torch_delta": True,
    "refit_xgb_delta": True,
    "blend_pooled_ece": True,
}


def recorded_margins() -> dict:
    """The margins as the committed decision artifacts recorded them.

    The "then" side of the comparison, read out of the artifacts rather than
    off a re-run, so it is the number each decision was actually taken on.
    Every one is the seeds 0-4 measurement against the same paired incumbent
    this re-validation re-runs, which is what makes the two sides comparable
    at all.
    """
    sp3 = load(SP3_DECISION)["experiments"]["D2"]["bar_check"]
    refit = load(REFIT_DECISION)
    ece = load(SP2_2_DECISION)["ece_gate"]
    return {
        "sp3_joint_delta": float(sp3["joint_delta"]),
        "sp3_winner_delta": float(sp3["winner_delta"]),
        "refit_torch_delta": float(refit["torch"]["delta_B_minus_A"]),
        "refit_xgb_delta": float(refit["xgb"]["delta_B_minus_A"]),
        "blend_pooled_ece": float(ece["B1_pooled_ece"]),
        "read_from": {
            "sp3_joint_delta": f"{_rel(SP3_DECISION)} experiments.D2.bar_check",
            "sp3_winner_delta": f"{_rel(SP3_DECISION)} experiments.D2.bar_check",
            "refit_torch_delta": f"{_rel(REFIT_DECISION)} torch",
            "refit_xgb_delta": f"{_rel(REFIT_DECISION)} xgb",
            "blend_pooled_ece": f"{_rel(SP2_2_DECISION)} ece_gate.B1_pooled_ece",
        },
    }


def margin_movement(then: dict, bars: dict) -> dict:
    """How far each margin moved between the recorded decision and today.

    A bar that still clears can still be worth reading about: a margin that
    halved on one table's worth of new fights is the early warning the next
    re-validation needs, and a verdict of "still justified" with no numbers
    beside it hides exactly that.
    """
    now = {
        "sp3_joint_delta": float(bars["sp3_joint_bar"]["joint_delta"]),
        "sp3_winner_delta": float(bars["sp3_joint_bar"]["winner_delta"]),
        "refit_torch_delta": float(bars["refit_rule_torch"]["delta_B_minus_A"]),
        "refit_xgb_delta": float(bars["refit_rule_xgb"]["delta_B_minus_A"]),
        "blend_pooled_ece": float(bars["blend_ece_gate"]["pooled_ece"]),
    }
    out = {}
    for key, now_value in now.items():
        change = round(now_value - then[key], 6)
        worse = change > 0 if LOWER_IS_BETTER[key] else change < 0
        out[key] = {
            "then": then[key],
            "now": now_value,
            "change": change,
            "closer_to_the_bar": bool(worse and change != 0),
            "recorded_in": (then.get("read_from") or {}).get(key),
        }
    return out


def verdict(bars: dict) -> dict:
    """The plain verdict. Only GATED bars decide it; ungated reads are surfaced.

    A gated bar that no longer clears is a finding, not a failure of this
    script: it says the evidence the deployed recipe rests on no longer
    supports it on the data the recipe is deployed on. What to do about that
    is a human call, which is why nothing here changes a model.
    """
    failed = sorted(name for name, bar in bars.items()
                    if bar.get("gated") and not bar.get("still_clears"))
    ungated_failed = sorted(name for name, bar in bars.items()
                            if not bar.get("gated") and not bar.get("still_clears"))
    if failed:
        outcome = (
            "The deployed recipe is NO LONGER justified by its own recorded bars on "
            f"the current table: {', '.join(failed)} no longer met. Nothing has been "
            "changed; this is a finding for a human to act on."
        )
    else:
        outcome = (
            "The deployed recipe still clears every bar it was justified by, "
            "re-measured on the current table."
        )
    return {
        "still_justified": not failed,
        "outcome": outcome,
        "bars_no_longer_met": failed,
        "ungated_bars_not_met": ungated_failed,
        "taken_by": "the recorded bars, re-applied mechanically by this script",
    }


def exit_code(verdict_block: dict) -> int:
    return 0 if verdict_block["still_justified"] else 1


# --- staleness: the cheap weekly read ---------------------------------------


def staleness(table_max_date: str, n_table_rows: int, members: dict) -> dict:
    """How far the deployed models have run ahead of their harness evidence.

    Reads dates, fits nothing. Each member's metrics file records both the
    date its training data reaches (`train_through`) and the date the harness
    report behind it ran on (`harness_features_max_date`); the gap between
    them is exactly what `stale_harness_warning` fires on in the train
    scripts, and it is the trigger for running the full re-validation.
    """
    out = {}
    for name, metrics in members.items():
        train_through = metrics.get("train_through")
        harness = metrics.get("harness_features_max_date")
        if train_through is None or harness is None:
            out[name] = {"train_through": train_through,
                         "harness_features_max_date": harness,
                         "days_behind": None, "stale": None}
            continue
        days = int((pd.Timestamp(train_through) - pd.Timestamp(harness)).days)
        out[name] = {
            "train_through": train_through,
            "harness_features_max_date": harness,
            "days_behind": max(days, 0),
            "stale": days > 0,
        }
    stale = any(m.get("stale") for m in out.values())
    return {
        "table_max_date": table_max_date,
        "n_table_rows": n_table_rows,
        "members": out,
        "stale": bool(stale),
        "what_it_means": (
            "the deployed models train on fights the walk-forward evidence behind "
            "their recipe never saw, so the recipe's justification is older than the "
            "recipe's training data"
        ) if stale else (
            "the harness evidence covers every fight the deployed models trained on"
        ),
        "what_to_do": (
            "run `python scripts/revalidate_recipe.py` (a few minutes of fitting) to "
            "re-measure every recorded bar on the current table; it writes "
            "models/walkforward/recipe_revalidation.json and exits non-zero only if a "
            "bar is no longer met"
        ),
    }


def benchmark_pairing(pooled_rows: int, sources: list[dict] | None = None) -> dict:
    """Whether `models/market_benchmark_oof.json` can still be regenerated.

    That benchmark is the project's honest "does the model beat the market?"
    answer, and the only thing that can rebuild it is a walk-forward
    prediction dump joined POSITIONALLY to the pooled rows of the table it was
    written from. Every weekly refresh grows the table and retires the
    previous dump, and nothing runs the benchmark script on a schedule -- so
    without this read the artifact quietly becomes unregenerable and the
    failure surfaces only when someone next tries.

    It is a WARNING, not a failure, for the same reason the harness-staleness
    read above is: the committed benchmark still describes the table it was
    computed on, and re-running it is a human-triggered few minutes of fitting.
    """
    rows = oof_source_status(pooled_rows) if sources is None else list(sources)
    usable = next((row for row in rows if row.get("pairs")), None)
    return {
        "artifact": _rel(OOF_BENCHMARK),
        "n_pooled_walkforward_rows": int(pooled_rows),
        "sources": rows,
        "usable_source": usable["report"] if usable else None,
        "regenerable": usable is not None,
        "what_it_means": (
            f"{_rel(OOF_BENCHMARK)} can be rebuilt on the current table from "
            f"{usable['report']}" if usable else
            "no committed prediction dump pairs with the current table, so "
            f"{_rel(OOF_BENCHMARK)} cannot be regenerated at all"
        ),
        "what_to_do": (
            "run `python scripts/revalidate_recipe.py` -- its hybrid run dumps "
            "the predictions this benchmark scores -- and then `python "
            "scripts/build_odds_benchmark.py --mode oof`"
        ),
    }


def staleness_warning(report: dict) -> str | None:
    """Warning text for anything the harness evidence no longer covers, else
    None. Written to stderr by the weekly Action's step, the way
    `scripts/check_snapshot_coverage.py` warns about the other measurement
    that ages rather than breaks.

    Two things age here and they age separately: the recipe's justification
    (measured on a table the models have since trained past) and the market
    benchmark's regenerability (a prediction dump the current table has
    outgrown). Either alone is worth saying."""
    warnings = []
    if report["stale"]:
        behind = max(m["days_behind"] or 0 for m in report["members"].values())
        warnings.append(
            f"WARNING: the deployed recipe's walk-forward evidence is {behind} day(s) "
            f"older than the data the deployed models train on (table now "
            f"{report['table_max_date']}, {report['n_table_rows']} rows). "
            "The recipe's justification has not been re-measured on that data; "
            "run `python scripts/revalidate_recipe.py`."
        )
    benchmark = report.get("benchmark")
    if benchmark is not None and not benchmark["regenerable"]:
        dumps = ", ".join(
            f"{row['report']} ({row['n_dump']} rows)" if row["exists"] else
            f"{row['report']} (absent)"
            for row in benchmark["sources"]
        )
        warnings.append(
            f"WARNING: {benchmark['artifact']} can no longer be regenerated. The "
            f"current table rebuilds {benchmark['n_pooled_walkforward_rows']} pooled "
            f"walk-forward rows and no committed dump matches it: {dumps}. "
            f"To rebuild it, {benchmark['what_to_do']}."
        )
    return "\n\n".join(warnings) or None


def print_staleness(report: dict) -> None:
    print(f"deployed recipe evidence, against a table ending "
          f"{report['table_max_date']} ({report['n_table_rows']} rows):")
    for name, member in sorted(report["members"].items()):
        print(f"  {name:<8} trains through {member['train_through']}, "
              f"harness evidence from {member['harness_features_max_date']} "
              f"({member['days_behind']} day(s) behind)")
    print(f"  {report['what_it_means']}")
    if report["stale"]:
        print(f"  {report['what_to_do']}")
    benchmark = report.get("benchmark")
    if benchmark is None:
        return
    print(f"\nmarket benchmark, against {benchmark['n_pooled_walkforward_rows']} "
          f"pooled walk-forward rows:")
    width = max(len(row["report"]) for row in benchmark["sources"])
    for row in benchmark["sources"]:
        state = ("no dump on disk" if not row["exists"]
                 else f"{row['n_dump']} pooled rows"
                      f"{' -- pairs' if row['pairs'] else ' -- does not pair'}")
        print(f"  {row['report']:<{width}} {state}")
    print(f"  {benchmark['what_it_means']}")
    if not benchmark["regenerable"]:
        print(f"  {benchmark['what_to_do']}")


def staleness_from_disk() -> dict:
    """`staleness` over the committed metrics files and the current table,
    plus the market benchmark's pairing against that same table."""
    features = pd.read_parquet(FEATURES, columns=["date"])
    report = staleness(
        table_max_date=str(pd.Timestamp(features["date"].max()).date()),
        n_table_rows=int(len(features)),
        members={name: load(path) for name, path in DEPLOYED_METRICS.items()},
    )
    return {**report, "benchmark": benchmark_pairing(pooled_row_count(features["date"]))}


# --- running the harness -----------------------------------------------------


def planned_runs(out_dir: Path) -> list[dict]:
    """The batch, in dependency order, as (key, name, argv) records."""
    plan = []
    for run in RUNS:
        config = load(run.source)["config"]
        name = f"revalidation_{run.key}"
        budget_from = (out_dir / f"revalidation_{run.budget_from}.json"
                       if run.budget_from else None)
        args = run_args(config, name, out_dir,
                        None if budget_from is None else str(budget_from))
        dump = Path(out_dir) / "preds" / f"{name}.json" if run.dump else None
        if dump is not None:
            args += ["--dump-predictions", str(dump)]
        plan.append({
            "key": run.key, "name": name, "role": run.role,
            "recipe_read_from": _rel(run.source),
            "args": args,
            "report": out_dir / f"{name}.json",
            "dump": dump,
        })
    return plan


def execute(plan: list[dict]) -> dict:
    """Run every planned harness run and return the fresh reports by key."""
    reports = {}
    for step in plan:
        print(f"\n=== {step['key']}: {step['role']}")
        print("    " + " ".join(step["args"]))
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_walkforward.py"), *step["args"]],
            cwd=ROOT, check=True,
        )
        reports[step["key"]] = load(step["report"])
    return reports


def reuse_reports(out_dir: Path, table_max_date: str, n_feature_rows: int) -> dict:
    """The batch's reports already on disk, checked against the current table.

    Re-deriving the artifact without re-fitting is what every other decision
    script in this project does (they are pure over committed reports), and it
    is what makes a fix to the artifact's wording cost seconds rather than a
    re-run. What it must not do is assemble a verdict out of runs made on a
    table that is no longer the current one, or out of two runs made on
    different tables -- so each report's own `config` is checked against the
    table this invocation is re-validating.
    """
    reports = {}
    for run in RUNS:
        path = Path(out_dir) / f"revalidation_{run.key}.json"
        if not path.exists():
            raise SystemExit(
                f"missing {path.name} in {out_dir}; --from-reports reuses a batch that "
                "has already been run, and this one has not"
            )
        report = json.loads(path.read_text())
        config = report["config"]
        if (config.get("features_max_date") != table_max_date
                or config.get("n_feature_rows") != n_feature_rows):
            raise SystemExit(
                f"{path.name} was computed on a table ending "
                f"{config.get('features_max_date')} with {config.get('n_feature_rows')} "
                f"rows, but the current table ends {table_max_date} with "
                f"{n_feature_rows} rows; re-run the batch rather than re-deriving a "
                "verdict from stale reports"
            )
        reports[run.key] = report
    return reports


# --- the artifact ------------------------------------------------------------


def build(reports: dict, configuration: dict, *, table: dict,
          run_date: str | None = None) -> dict:
    """The re-validation artifact, given the fresh reports. Pure."""
    sp3 = sp3_thresholds(load(SP3_DECISION))
    refit = refit_rule(load(REFIT_DECISION))
    gate = ece_gate(load(SP2_2_DECISION))
    measured_sigma = float(load(NOISE_FLOOR)["sigma_seed"])
    check_sigma_agreement(sp3["sigma_seed"], refit["sigma_seed"], measured_sigma)

    bars = {
        "sp3_joint_bar": check_sp3_bar(reports["hybrid"], reports["blend_cells"], sp3),
        "refit_rule_torch": check_refit_rule(reports["torch_A"], reports["torch_B"],
                                             refit, "torch"),
        "refit_rule_xgb": check_refit_rule(reports["xgb_A"], reports["xgb_B"],
                                           refit, "xgb"),
        "blend_ece_gate": check_ece_gate(reports["blend_cells"], gate),
    }
    decision = verdict(bars)
    moved = margin_movement(recorded_margins(), bars)
    return {
        "experiment": "periodic re-validation of the deployed recipe on the current table",
        "generated_by": "scripts/revalidate_recipe.py",
        "date": run_date or date.today().isoformat(),
        "replaces": (
            "scripts/roll_window.py's promotion gate, which asked a question this "
            "project no longer has: the deployed recipe is refit-through-latest and "
            "the weekly Action already retrains every member, so there is no "
            "candidate-versus-incumbent choice to make"
        ),
        "table": table,
        "deployed_configuration": configuration,
        "reports": {key: _rel(REVAL_DIR / f"revalidation_{key}.json") for key in reports},
        "measured_sigma_seed": measured_sigma,
        "bars": bars,
        "margin_movement": moved,
        "verdict": decision,
        "what_this_does_not_do": (
            "It deploys nothing, promotes nothing and stages nothing. It does not "
            "re-derive the blend's fixed post-average temperature (that is "
            "scripts/derive_blend_temperature.py against a fresh dump), and it does "
            "not re-run the fresh-seed confirmations the original decisions required "
            "before shipping -- a re-validation asks whether what already ships still "
            "clears, not whether something new may ship."
        ),
    }


def print_summary(artifact: dict) -> None:
    table = artifact["table"]
    print(f"\nre-validated on {table['features_max_date']} "
          f"({table['n_feature_rows']} rows; the recipe was justified on "
          f"{table['recipe_was_justified_on']})")
    print(f"deployed hash: {artifact['deployed_configuration']['deployed_hash']}")
    print("\nbar                     gated   still clears   numbers")
    for name, bar in artifact["bars"].items():
        if name == "sp3_joint_bar":
            numbers = (f"joint {bar['joint_delta']:+.6f} vs bar {-bar['joint_bar']:.3f}; "
                       f"winner {bar['winner_delta']:+.6f} vs {bar['sigma_seed']:.6f}")
        elif name == "blend_ece_gate":
            numbers = (f"ECE {bar['pooled_ece']:.4f} vs threshold "
                       f"{bar['thresholds']['threshold']:.6f}")
        else:
            numbers = (f"delta B-A {bar['delta_B_minus_A']:+.4f} vs sigma "
                       f"{bar['thresholds']['sigma_seed']:.6f}")
        print(f"  {name:<22} {str(bool(bar['gated'])):<7} "
              f"{str(bool(bar['still_clears'])):<14} {numbers}")
    print("\nmargin                  then         now       change  closer to the bar")
    for name, row in artifact["margin_movement"].items():
        print(f"  {name:<22} {row['then']:>+9.6f} {row['now']:>+9.6f} "
              f"{row['change']:>+12.6f}  {row['closer_to_the_bar']}")
    print(f"\n{artifact['verdict']['outcome']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-staleness", action="store_true",
                        help="report how far the deployed models have run ahead of "
                             "their harness evidence, and whether the market "
                             "benchmark can still be regenerated; exits 0, fits nothing")
    parser.add_argument("--plan", action="store_true",
                        help="print the harness runs this would make and exit; fits nothing")
    parser.add_argument("--from-reports", action="store_true",
                        help="re-derive the artifact from a batch already in "
                             "--reports-dir instead of re-running it; refuses reports "
                             "computed on any table but the current one")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--reports-dir", type=Path, default=REVAL_DIR)
    parser.add_argument("--print", dest="echo", action="store_true",
                        help="echo the JSON as well as the human summary")
    args = parser.parse_args(argv)

    if args.check_staleness:
        report = staleness_from_disk()
        print_staleness(report)
        if args.echo:
            print(json.dumps(report, indent=2))
        warning = staleness_warning(report)
        if warning:
            print(f"\n{warning}", file=sys.stderr)
        return 0

    plan = planned_runs(args.reports_dir)
    if args.plan:
        for step in plan:
            print(f"{step['key']}: {step['role']}")
            print("  recipe from " + step["recipe_read_from"])
            print("  run_walkforward.py " + " ".join(step["args"]))
        return 0

    configuration = deployed_configuration()
    if args.from_reports:
        current = pd.read_parquet(FEATURES, columns=["date"])
        reports = reuse_reports(
            args.reports_dir,
            table_max_date=str(pd.Timestamp(current["date"].max()).date()),
            n_feature_rows=int(len(current)),
        )
    else:
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        reports = execute(plan)

    hybrid_config = reports["hybrid"]["config"]
    table = {
        "features_max_date": hybrid_config["features_max_date"],
        "n_feature_rows": hybrid_config["n_feature_rows"],
        "recipe_was_justified_on": load(RUNS_BY_KEY["hybrid"].source)["config"]["features_max_date"],
        "n_feature_rows_when_justified": load(RUNS_BY_KEY["hybrid"].source)["config"]["n_feature_rows"],
        "blocks": configuration["feature_blocks"],
    }
    artifact = build(reports, configuration, table=table)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print_summary(artifact)
    return exit_code(artifact["verdict"])


if __name__ == "__main__":
    raise SystemExit(main())
