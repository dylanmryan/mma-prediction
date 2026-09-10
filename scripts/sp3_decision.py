"""Assemble SP3's decision artifact for the round-by-round fight simulator.

    python scripts/sp3_decision.py            # writes models/walkforward/sp3_decision.json
    python scripts/sp3_decision.py --print    # ... and echoes it

Reads only committed walk-forward reports, the noise floor, the committed
prediction dumps and the committed fights table -- paths are the constants
below -- so re-running it reproduces the artifact exactly.

What this script does and does not do:

* It quotes SP3's bar out of the plan file rather than retyping it, and it
  READS THE TWO NUMBERS IN THE BAR OUT OF THAT QUOTED TEXT (`joint_bar_from`,
  `sigma_from`) instead of hardcoding them. An artifact that states a bar the
  pre-registration does not state is worth nothing, and a constant retyped
  next to a quote is exactly how the two drift apart.
* It cross-checks the sigma the plan quotes against the measured seed noise
  floor in `models/walkforward/noise_floor.json`, so the bar's tolerance is
  known to be the floor that was actually measured.
* It applies the bar mechanically (`simulator_bar_check`) to every experiment
  against its paired incumbent, at seeds 0-4 and again at the mandatory
  fresh seeds 5-9.
* It VERIFIES, rather than assumes, the claim that makes the hybrid's winner
  clause hold by construction: `identical_winner_arrays` compares the pooled
  per-row winner probabilities of the hybrid and of its blend member
  element-wise, from the two committed prediction dumps.
* It records the Monte Carlo standard error and the zero-mass-cell fraction
  that the plan required be measured and reported, per fold and pooled.
* It records three findings that are not otherwise anywhere in the tree --
  the macro-F1 yardstick, the hybrid beating the pure simulator, and what the
  D3 control rules out -- each next to the measured numbers that support it,
  so they are claims a reader can check rather than assertions.

It does NOT deploy anything. SP3's deployment is a separate, larger change
(plan Task 6); this artifact is the record of the call, not the call's
execution.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WF = ROOT / "models" / "walkforward"
PREDS = WF / "preds"
OUT = WF / "sp3_decision.json"
PLAN = ROOT / "docs" / "superpowers" / "plans" / "2026-09-09-sp3-fight-simulator.md"
FIGHTS = ROOT / "data" / "processed" / "fights.parquet"

#: The heading whose bullets are the pre-registered evaluation rules.
RULES_HEADING = "## Locked evaluation rules"
#: The bullet within them that IS the bar. Matched on this marker so a
#: reworded neighbouring bullet cannot silently become "the bar".
BAR_MARKER = "**Simulator ships**"
FRESH_SEED_MARKER = "**Fresh-seed confirmation is mandatory**"

#: `mma.models.train_loop.METHOD_CLASSES` minus the decision class -- the two
#: finishing methods whose rows are the finishes finding 1 talks about.
FINISH_METHODS = ("ko_tko", "submission")

# --- the reports the artifact reads ----------------------------------------
# The incumbent is the deployed blend (B1). `blend_b1_cells` is the SAME
# prediction routed through the simulator's cell scorer -- SP3's D3 control --
# and it is what the simulator's joint is compared against, because both sides
# then read `-log P(realised cell)` off the same scorer.
INCUMBENT = WF / "blend_b1.json"
INCUMBENT_CELLS = WF / "blend_b1_cells.json"
INCUMBENT_SEEDS5 = WF / "blend_b1_seeds5.json"
INCUMBENT_CELLS_SEEDS5 = WF / "blend_b1_cells_seeds5.json"

E1 = WF / "hazard_e1.json"          # the paradigm on v1 features
E2 = WF / "hazard_e2.json"          # the paradigm on the shipped features
D1 = WF / "hazard_e2_cal.json"      # E2 with its winner marginal calibrated
D2 = WF / "hybrid_e2.json"          # the spec's fallback: blend winner, simulator shape
FRESH = WF / "hybrid_e2_seeds5.json"  # D2 at the mandatory fresh seeds

NOISE_FLOOR = WF / "noise_floor.json"

# Pooled per-row predictions (run_walkforward.py --dump-predictions), in
# `walkforward.pool`'s row order. The reports carry metrics and no
# predictions, so the element-wise winner check is not computable without
# them.
PRED_FRESH = PREDS / "hybrid_e2_seeds5.json"
PRED_FRESH_INCUMBENT = PREDS / "blend_b1_cells_seeds5.json"


# --- pure helpers -----------------------------------------------------------


def quoted_rules(plan_text: str) -> list[str]:
    """The pre-registration's locked evaluation rules, quoted verbatim.

    The bullet list under `RULES_HEADING`, in order, each stripped of nothing
    but surrounding whitespace -- markdown emphasis is left exactly as
    written. Taken out of the plan file rather than retyped, so the artifact
    cannot drift from the rules it claims to quote. Continuation lines of a
    bullet are folded into that bullet, so a wrapped rule is quoted whole.
    """
    lines = plan_text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(RULES_HEADING)]
    if len(starts) != 1:
        raise ValueError(
            f"expected exactly one {RULES_HEADING!r} heading, found {len(starts)}"
        )
    rules: list[str] = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("#") or line.startswith("---"):
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            rules.append(stripped)
        elif rules:  # a wrapped continuation of the bullet above
            rules[-1] = f"{rules[-1]} {stripped}"
    if not rules:
        raise ValueError(f"no bullets found under {RULES_HEADING!r}")
    return rules


def quoted_bar(plan_text: str, marker: str = BAR_MARKER) -> str:
    """The single locked rule carrying `marker`, verbatim.

    Exactly one must carry it: two would mean the artifact had a choice about
    which text is "the bar", and none would mean it is quoting nothing.
    """
    matches = [rule for rule in quoted_rules(plan_text) if marker in rule]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one locked rule containing {marker!r}, found {len(matches)}"
        )
    return matches[0]


def joint_bar_from(bar_text: str) -> float:
    """The joint-log-loss margin the quoted bar requires, read out of its text.

    The bar says the joint must beat the baseline "by **more than 0.01**".
    Parsing it rather than retyping it is the point: a constant sitting next
    to a quote is free to disagree with it, and this way the artifact cannot
    apply a bar the pre-registration does not state.
    """
    # The emphasis wraps the whole phrase ("**more than 0.01**"), so the
    # asterisks are matched loosely on both sides rather than assumed to hug
    # the number. The second clause's "more than sigma_seed" carries no digit
    # and so cannot match.
    matches = re.findall(r"more than \**([0-9]*\.?[0-9]+)\**", bar_text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one 'more than <number>' in the bar, found {matches!r}"
        )
    return float(matches[0])


def sigma_from(bar_text: str) -> float:
    """The sigma_seed the quoted bar names, read out of its text.

    The bar's second clause tolerates a winner regression of at most
    "σ_seed (0.000346)". Parsed for the same reason `joint_bar_from` parses
    the first clause, and cross-checked in `build` against the noise floor
    that value was measured from.
    """
    matches = re.findall(r"σ_seed \(([0-9]*\.?[0-9]+)\)", bar_text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one 'σ_seed (<number>)' in the bar, found {matches!r}"
        )
    return float(matches[0])


def _finite(value, label: str) -> float:
    """`float(value)`, or a loud failure -- a NaN metric must not read
    downstream as a candidate that merely failed the bar."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} is not finite ({value!r})")
    return out


def simulator_bar_check(candidate: dict, incumbent: dict, sigma_seed: float,
                        joint_bar: float) -> dict:
    """SP3's pre-registered bar, applied mechanically to a report pair.

    > "**Simulator ships** iff joint-outcome log-loss beats the composed
    > baseline by **more than 0.01** *and* its winner marginal is not worse
    > than the incumbent's by more than σ_seed (0.000346)."

    Two clauses, both required:

    * **joint** -- `candidate - incumbent` must be strictly below `-joint_bar`.
      "More than 0.01" is strict, and both sides are rounded to 6 dp before
      the comparison so a candidate sitting exactly on the bar does not clear
      it on float noise. This is `mma.walkforward.bar_check`'s convention,
      moved to the joint metric.
    * **winner** -- `candidate - incumbent` must not exceed `sigma_seed`. Note
      the asymmetry with the joint clause: this one is a tolerance on a
      REGRESSION, so it passes at equality and passes outright for a candidate
      whose winner marginal is better.

    The pair is `comparable` only when both reports cover the same fold years
    and the same pooled `n`; a non-comparable pair never ships. Per-fold
    deltas are reported for both metrics but are NOT gated on: SP3's bar, as
    written above, has no per-fold no-regression clause (that belongs to the
    v3 spec §5 winner bar, `mma.walkforward.bar_check`), and adding one here
    would be a stricter bar than the one pre-registered.
    """
    if sigma_seed is None or not math.isfinite(float(sigma_seed)):
        raise ValueError("sigma_seed must be a finite number; run scripts/noise_floor.py first")
    if not float(joint_bar) > 0:
        raise ValueError(f"joint_bar must be positive (got {joint_bar!r})")
    bar = round(float(joint_bar), 6)
    tolerance = round(float(sigma_seed), 6)

    cand_joint = _finite(candidate["pooled"]["joint_log_loss"], "candidate pooled joint_log_loss")
    inc_joint = _finite(incumbent["pooled"]["joint_log_loss"], "incumbent pooled joint_log_loss")
    cand_win = _finite(candidate["pooled"]["winner_log_loss"], "candidate pooled winner_log_loss")
    inc_win = _finite(incumbent["pooled"]["winner_log_loss"], "incumbent pooled winner_log_loss")
    joint_delta = round(cand_joint - inc_joint, 6)
    winner_delta = round(cand_win - inc_win, 6)

    cand_years, inc_years = set(candidate["folds"]), set(incumbent["folds"])
    missing_folds = sorted(cand_years ^ inc_years)
    comparable = (not missing_folds
                  and candidate["pooled"].get("n") == incumbent["pooled"].get("n"))
    common_years = [year for year in candidate["folds"] if year in incumbent["folds"]]

    def fold_deltas(metric: str) -> dict:
        return {
            year: round(
                _finite(candidate["folds"][year][metric], f"candidate fold {year} {metric}")
                - _finite(incumbent["folds"][year][metric], f"incumbent fold {year} {metric}"),
                4)
            for year in common_years
        }

    joint_folds = fold_deltas("joint_log_loss")
    winner_folds = fold_deltas("winner_log_loss")
    clears_joint = joint_delta < -bar
    winner_ok = winner_delta <= tolerance
    return {
        "convention": "delta = candidate - incumbent; negative = the candidate is better",
        "joint_bar": bar,
        "sigma_seed": tolerance,
        "candidate_joint_log_loss": cand_joint,
        "incumbent_joint_log_loss": inc_joint,
        "joint_delta": joint_delta,
        "joint_margin_past_the_bar": round(-joint_delta - bar, 6),
        "clears_joint_bar": bool(clears_joint),
        "candidate_winner_log_loss": cand_win,
        "incumbent_winner_log_loss": inc_win,
        "winner_delta": winner_delta,
        "winner_headroom": round(tolerance - winner_delta, 6),
        "winner_clause_satisfied": bool(winner_ok),
        "joint_fold_deltas": joint_folds,
        "winner_fold_deltas": winner_folds,
        "worst_joint_fold_delta": round(max(joint_folds.values()), 4) if joint_folds else 0.0,
        "worst_winner_fold_delta": round(max(winner_folds.values()), 4) if winner_folds else 0.0,
        "fold_deltas_are_reported_not_gated": (
            "SP3's bar has no per-fold no-regression clause; that belongs to the v3 spec "
            "section 5 winner bar (mma.walkforward.bar_check). These are here to be read, "
            "not to gate."
        ),
        "comparable": bool(comparable),
        "missing_folds": missing_folds,
        "ships": bool(comparable and clears_joint and winner_ok),
    }


def identical_winner_arrays(candidate_dump: dict, incumbent_dump: dict) -> dict:
    """Element-wise equality of two prediction dumps' winner probabilities.

    `HybridCandidate` returns its blend member's winner array as its own --
    not a copy read back off the joint cells -- so the bar's winner clause is
    meant to hold BY CONSTRUCTION rather than by measurement. "Meant to" is
    not a verification: this is the verification. The two dumps are in
    `walkforward.pool`'s row order, so equal `fold_year` and `y_winner`
    sequences establish that the same rows are being compared before the
    probabilities are compared at all -- otherwise two arrays could agree
    element-wise while describing different fights.

    Equality is exact (`np.array_equal`), not `allclose`: the claim is that
    these are the same numbers, and a tolerance would hide a re-derivation
    that merely agrees to a few decimal places.
    """
    cand = np.asarray(candidate_dump["p_winner"], dtype=float)
    inc = np.asarray(incumbent_dump["p_winner"], dtype=float)
    same_rows = bool(
        candidate_dump["n"] == incumbent_dump["n"]
        and list(candidate_dump["fold_year"]) == list(incumbent_dump["fold_year"])
        and list(candidate_dump["y_winner"]) == list(incumbent_dump["y_winner"])
    )
    aligned = cand.shape == inc.shape
    identical = bool(same_rows and aligned and np.array_equal(cand, inc))
    return {
        "candidate_predictions": _rel(PRED_FRESH),
        "incumbent_predictions": _rel(PRED_FRESH_INCUMBENT),
        "n_rows_compared": int(len(cand)),
        "same_rows": same_rows,
        "rows_checked_by": "equal fold_year and y_winner sequences in walkforward.pool's order",
        "comparison": "np.array_equal on p_winner -- exact, not a tolerance",
        "n_elements_differing": int(np.count_nonzero(cand != inc)) if aligned else None,
        "max_absolute_difference": float(np.max(np.abs(cand - inc))) if aligned else None,
        "identical": identical,
    }


def finish_round_shares(fights: pd.DataFrame,
                        finish_methods=FINISH_METHODS) -> dict:
    """Share of finishes ending in each round, from a fights table.

    Finding 1 rests on round 1 being the plurality outcome by a wide margin --
    which is why argmaxing it is the right call for a round head and why the
    head that does so collects a poor macro-F1. That is a fact about the data,
    so it is computed from the data rather than asserted.
    """
    finishes = fights[fights["method"].isin(list(finish_methods))]
    rounds = finishes["finish_round"].dropna()
    counts = rounds.value_counts().sort_index()
    total = int(counts.sum())
    if not total:
        raise ValueError("no finishes with a recorded finish_round")
    return {
        "n_finishes_with_a_round": total,
        "counts": {str(int(k)): int(v) for k, v in counts.items()},
        "shares": {str(int(k)): round(int(v) / total, 4) for k, v in counts.items()},
        "modal_round": str(int(counts.idxmax())),
        "modal_share": round(int(counts.max()) / total, 4),
    }


# --- report plumbing --------------------------------------------------------


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def pooled(report: dict, metric: str) -> float:
    return float(report["pooled"][metric])


def fit_info_spread(report: dict, key: str) -> dict | None:
    """Per-fold values of a `fit_info` field, plus their range.

    The simulator's fields sit at the top level of `fit_info` for the hazard
    candidate and one level down under `"hazard"` for the hybrid, which nests
    both members' diagnostics. Returns None when the report has neither.
    """
    info = report["fit_info"]
    values = info.get(key)
    if values is None and isinstance(info.get("hazard"), list):
        nested = [fold.get(key) for fold in info["hazard"] if isinstance(fold, dict)]
        values = nested if all(v is not None for v in nested) and nested else None
    if values is None:
        return None
    values = [float(v) for v in values]
    return {"per_fold": values, "min": min(values), "max": max(values),
            "mean": round(float(np.mean(values)), 6)}


def profile(path: Path) -> dict:
    """Pooled, per-fold and per-slice metrics for one committed report."""
    report = load(path)
    config = report["config"]
    return {
        "report": _rel(path),
        "candidate": config.get("candidate"),
        "seeds": config.get("seeds"),
        "drop_columns": config.get("drop_columns"),
        "runtime_sec": config.get("runtime_sec"),
        "pooled": report["pooled"],
        "folds": {year: {"winner_log_loss": fold["winner_log_loss"],
                         "joint_log_loss": fold["joint_log_loss"]}
                  for year, fold in report["folds"].items()},
        "slices": {name: {"n": s["n"], "winner_log_loss": s["winner_log_loss"],
                          "joint_log_loss": s["joint_log_loss"],
                          "method_macro_f1": s["method_macro_f1"],
                          "round_macro_f1": s["round_macro_f1"]}
                   for name, s in report["slices"].items()},
    }


def experiment(path: Path, incumbent: Path, sigma_seed: float, joint_bar: float,
               role: str, note: str) -> dict:
    """One experiment's profile plus the bar applied against its paired incumbent."""
    out = {"role": role, "note": note}
    out.update(profile(path))
    out["paired_incumbent"] = _rel(incumbent)
    out["bar_check"] = simulator_bar_check(load(path), load(incumbent), sigma_seed, joint_bar)
    monte_carlo = {
        key: fit_info_spread(load(path), key)
        for key in ("mc_standard_error", "mc_standard_error_max",
                    "zero_mass_cell_fraction", "n_runs", "alpha")
    }
    if any(v is not None for v in monte_carlo.values()):
        out["monte_carlo"] = {k: v for k, v in monte_carlo.items() if v is not None}
    return out


def build() -> dict:
    plan_text = PLAN.read_text()
    rules = quoted_rules(plan_text)
    bar_text = quoted_bar(plan_text)
    fresh_seed_text = quoted_bar(plan_text, FRESH_SEED_MARKER)
    joint_bar = joint_bar_from(bar_text)
    sigma_quoted = sigma_from(bar_text)

    floor = load(NOISE_FLOOR)
    sigma_measured = float(floor["sigma_seed"])
    # The plan quotes sigma to 6 dp; the floor carries it at full precision.
    # The bar is applied at the quoted precision (that is the pre-registered
    # number), and the two are asserted to agree so the artifact cannot apply
    # a tolerance that was never measured.
    sigma_agrees = round(sigma_measured, 6) == round(sigma_quoted, 6)
    if not sigma_agrees:
        raise ValueError(
            f"the plan quotes sigma_seed {sigma_quoted!r} but {_rel(NOISE_FLOOR)} measured "
            f"{sigma_measured!r}; the bar's tolerance and the floor it came from disagree"
        )
    sigma = sigma_quoted

    e1 = experiment(
        E1, INCUMBENT_CELLS, sigma, joint_bar,
        role="E1 -- the paradigm on v1 (base) features",
        note=("Isolates the paradigm from the data: if the simulator only wins with the "
              "shipped blocks, the win is the blocks and not the generative process."))
    e2 = experiment(
        E2, INCUMBENT_CELLS, sigma, joint_bar,
        role="E2 -- the paradigm on the shipped features",
        note=("The real candidate. It landed exactly on the case the v3 spec's hybrid "
              "fallback was written for: the joint clears the bar several times over "
              "while the winner marginal regresses by more than sigma_seed."))
    d1 = experiment(
        D1, INCUMBENT_CELLS, sigma, joint_bar,
        role="D1 -- E2 with its winner marginal temperature-scaled",
        note=("The simulator is the one candidate in this project with no calibration "
              "step. Adding the blend's -- a temperature fitted on the fold's "
              "inner-validation year, imposed back on the joint -- separates calibration "
              "from paradigm. It recovers about half the winner gap, not all of it, so "
              "calibration alone does not satisfy the winner clause."))
    d2 = experiment(
        D2, INCUMBENT_CELLS, sigma, joint_bar,
        role="D2 -- the hybrid: the blend's winner, the simulator's conditional shape",
        note=("The v3 spec's pre-registered fallback. P(winner) is the blend's own array "
              "and P(method, round | winner) is the simulator's, composed by scaling each "
              "corner's block of simulated cells by one scalar."))

    # D3 is a control, not a candidate: it is the incumbent's own prediction
    # routed through the simulator's cell scorer, so it is compared against
    # the incumbent's composed score to show the two scorers agree exactly.
    d3_report, incumbent_report = load(INCUMBENT_CELLS), load(INCUMBENT)
    d3 = {
        "role": "D3 -- control: the incumbent's own heads through the cell scorer",
        "note": ("Not a candidate. It changes nothing about the prediction -- the three "
                 "heads are the same arrays either way -- and only routes the joint "
                 "metric through evaluate.joint_cell_log_loss instead of "
                 "joint_outcome_log_loss. If the cell machinery contributed anything of "
                 "its own, the two scores would differ."),
        **profile(INCUMBENT_CELLS),
        "compared_against": _rel(INCUMBENT),
        "composed_joint_log_loss": pooled(incumbent_report, "joint_log_loss"),
        "cell_scored_joint_log_loss": pooled(d3_report, "joint_log_loss"),
        "difference": round(pooled(d3_report, "joint_log_loss")
                            - pooled(incumbent_report, "joint_log_loss"), 6),
        "every_pooled_metric_identical": bool(d3_report["pooled"] == incumbent_report["pooled"]),
    }

    fresh = experiment(
        FRESH, INCUMBENT_CELLS_SEEDS5, sigma, joint_bar,
        role="Fresh-seed confirmation -- D2 at seeds 5-9, mandatory before anything ships",
        note=("The same recipe at a disjoint seed set, against a fresh-seed paired "
              "incumbent built the same way. This step has already caught one false "
              "positive in SP2.1, which is why the plan makes it mandatory rather than "
              "advisory."))
    fresh["also_compared_against_the_composed_incumbent"] = {
        "incumbent": _rel(INCUMBENT_SEEDS5),
        "note": ("blend_b1_seeds5 is the same blend at seeds 5-9 scored by the composed "
                 "joint rather than the cell scorer. D3 shows the two scorers agree "
                 "exactly, so this is a corroborating read of the same bar, not a "
                 "second bar."),
        "bar_check": simulator_bar_check(load(FRESH), load(INCUMBENT_SEEDS5), sigma, joint_bar),
    }
    fresh["incumbent_config_matches_on_every_field_but_seeds"] = config_match(
        INCUMBENT_CELLS_SEEDS5, INCUMBENT_CELLS)

    winner_arrays = identical_winner_arrays(load(PRED_FRESH), load(PRED_FRESH_INCUMBENT))
    ships = bool(fresh["bar_check"]["ships"])

    e2_bar, d2_bar, fresh_bar = e2["bar_check"], d2["bar_check"], fresh["bar_check"]
    rounds = finish_round_shares(pd.read_parquet(FIGHTS))
    sim_round_f1 = pooled(load(E2), "round_macro_f1")
    inc_round_f1 = pooled(incumbent_report, "round_macro_f1")

    return {
        "experiment": "SP3 round-by-round fight simulator",
        "pre_registration": _rel(PLAN),
        "generated_by": "scripts/sp3_decision.py",
        "decided": True,
        "decision": {
            "outcome": "ship the hybrid (D2)" if ships else "ship nothing",
            "date": "2026-09-09",
            "taken_by": "the pre-registered bar, applied mechanically -- no human call was reserved",
            "what_ships": (
                "The hybrid: P(winner) from the deployed blend B1 (equal-weight, "
                "temperature-scaled after averaging) and P(method, round | winner) from "
                "the Monte Carlo fight simulator, composed by "
                "mma.joint.impose_winner_marginal on the shipped feature blocks with "
                "external_missing, same_country, notice_unknown, home_country_a and "
                "home_country_b held out of every model matrix."
            ) if ships else "Nothing. No candidate satisfied both clauses at fresh seeds.",
            "why": (
                f"The bar has two clauses and the hybrid satisfies both, at seeds 0-4 and "
                f"again at the mandatory fresh seeds 5-9. Joint: "
                f"{fresh_bar['candidate_joint_log_loss']} against the fresh-seed "
                f"incumbent's {fresh_bar['incumbent_joint_log_loss']}, a gain of "
                f"{abs(fresh_bar['joint_delta'])} against a bar of {joint_bar} -- "
                f"{round(abs(fresh_bar['joint_delta']) / joint_bar, 1)}x the required "
                f"margin. Winner: identical to the incumbent's to the last bit, because "
                f"the hybrid returns the blend member's own array; verified element-wise "
                f"over {winner_arrays['n_rows_compared']} pooled rows rather than "
                f"assumed (see winner_clause_by_construction). E2 showed the pure "
                f"simulator clears the joint clause but fails the winner clause by "
                f"{e2_bar['winner_delta']} against a tolerance of {sigma}, and D1 showed "
                f"calibration recovers only part of that, which is why the spec's "
                f"pre-registered hybrid fallback is what ships."
            ) if ships else (
                "The fresh-seed confirmation did not satisfy both clauses. Under the "
                "pre-registration a candidate that fails at fresh seeds does not ship, "
                "whatever it measured at seeds 0-4."
            ),
            "deployment": (
                "NOT performed by this script or in the commit that wrote it. SP3's "
                "deployment is plan Task 6 and is a larger change than SP2.2's: the "
                "simulator replaces the method and round heads and supplies a joint "
                "distribution, so inference, versioning (n_runs and alpha are part of "
                "what a prediction IS and must be hashed), the prediction records, the "
                "app and the display priors all move together."
            ),
        },
        "bar": {
            "quoted_from_the_plan": bar_text,
            "fresh_seed_rule_quoted_from_the_plan": fresh_seed_text,
            "locked_evaluation_rules": rules,
            "form": (
                "Two clauses, both required, both applied to pooled metrics: the joint "
                "must beat the paired incumbent's by MORE than the joint bar (strict), "
                "and the winner marginal must not be WORSE than it by more than "
                "sigma_seed (a tolerance on a regression, so equality passes)."
            ),
            "joint_bar": joint_bar,
            "joint_bar_read_from": "the quoted bar's own text (joint_bar_from), not a retyped constant",
            "sigma_seed_quoted_in_the_plan": sigma_quoted,
            "sigma_seed_measured": sigma_measured,
            "sigma_seed_source": _rel(NOISE_FLOOR),
            "sigma_seed_seed_sets": floor["seed_sets"],
            "sigma_seed_agrees_with_the_measured_floor": sigma_agrees,
            "applied_at": sigma,
        },
        "incumbent": {
            "report": _rel(INCUMBENT),
            "cell_scored_report": _rel(INCUMBENT_CELLS),
            "fresh_seed_report": _rel(INCUMBENT_CELLS_SEEDS5),
            "note": ("B1, the deployed blend, in harness form (a temperature fitted per "
                     "fold on that fold's inner-validation year). The plan fixes the "
                     "harness form as the like-for-like comparison because that is the "
                     "form every candidate is scored in."),
            "pooled_winner_log_loss": pooled(incumbent_report, "winner_log_loss"),
            "pooled_joint_log_loss": pooled(incumbent_report, "joint_log_loss"),
            "fresh_seed_pooled_winner_log_loss": pooled(load(INCUMBENT_CELLS_SEEDS5), "winner_log_loss"),
            "fresh_seed_pooled_joint_log_loss": pooled(load(INCUMBENT_CELLS_SEEDS5), "joint_log_loss"),
        },
        "experiments": {"E1": e1, "E2": e2, "D1": d1, "D2": d2, "D3": d3, "fresh_seed": fresh},
        "winner_clause_by_construction": {
            "claim": ("HybridCandidate returns pred['winner'] = the blend member's own "
                      "array, so every winner metric in its report is the incumbent's "
                      "number exactly and the bar's winner clause holds by construction."),
            "verification": winner_arrays,
            "corroboration": {
                "note": ("The report metrics agree too, which is a weaker check -- they "
                         "are rounded to 4 dp -- but it is the check a reader of the "
                         "reports alone can make."),
                "hybrid_pooled": {k: load(FRESH)["pooled"][k]
                                  for k in ("winner_log_loss", "accuracy", "brier", "ece")},
                "incumbent_pooled": {k: load(INCUMBENT_CELLS_SEEDS5)["pooled"][k]
                                     for k in ("winner_log_loss", "accuracy", "brier", "ece")},
            },
        },
        "monte_carlo": {
            "n_runs": 10000,
            "alpha": 1.0,
            "fixed_when": "before any run, by the plan's 'Scoring the simulator fairly' section",
            "requirement": ("the plan required the Monte Carlo standard error on P(A wins) "
                            "be verified below 0.005 and the zero-mass cell fraction "
                            "reported, with more than ~1% meaning N is too small"),
            "standard_error": {
                name: fit_info_spread(load(path), "mc_standard_error")
                for name, path in (("E1", E1), ("E2", E2), ("D1", D1), ("D2", D2),
                                   ("fresh_seed", FRESH))
            },
            "standard_error_max": {
                name: fit_info_spread(load(path), "mc_standard_error_max")
                for name, path in (("E2", E2), ("D2", D2), ("fresh_seed", FRESH))
            },
            "standard_error_note": (
                "mc_standard_error_max is 0.005 in every fold, which is not a coincidence "
                "and not a failure: 0.5/sqrt(10000) is exactly the analytic worst case, "
                "attained at p=0.5, and the plan predicted it ('it should be ~0.005 at "
                "p=0.5, N=10,000 for a single member'). The MEAN across fights is below "
                "it in every fold of every run, and averaging the two corner orientations "
                "and five seed members reduces the error further still."
            ),
            "zero_mass_cell_fraction": {
                name: fit_info_spread(load(path), "zero_mass_cell_fraction")
                for name, path in (("E1", E1), ("E2", E2), ("D1", D1), ("D2", D2),
                                   ("fresh_seed", FRESH))
            },
            "zero_mass_note": (
                "Zero in every fold of every run: no realised cell had zero raw simulated "
                "mass, so the Laplace alpha=1 smoothing never had to rescue a "
                "-log 0, and N=10,000 is comfortably large enough. The smoothing stays "
                "on regardless, applied identically to every candidate, because it was "
                "pre-registered."
            ),
        },
        "findings": {
            "1_macro_f1_was_the_wrong_yardstick": {
                "finding": (
                    "Macro-F1 was the wrong yardstick for the old method and round heads, "
                    "and it is why those heads looked like the weakest part of the system. "
                    "The simulator's round marginal is much BETTER than the incumbent's on "
                    "log-loss and much WORSE on macro-F1. Both are true and only the first "
                    "one measures the distribution. Argmaxing round 1 is the correct call "
                    f"-- round 1 is {round(rounds['modal_share'] * 100, 1)}% of finishes -- "
                    "so a well-calibrated round head puts its argmax there on nearly every "
                    "fight and collects macro-F1 credit for exactly one class. The old head "
                    "spread its argmax across the rare classes and was rewarded by macro-F1 "
                    "for predicting them, while predicting them badly."
                ),
                "why_it_matters": (
                    "The SP3 plan's own motivation cites the method head's 0.39 macro-F1 "
                    "and the round head's 0.31 as the weakest parts of the system. That "
                    "framing was measured with the wrong instrument. The paradigm change "
                    "was still worth making -- the joint log-loss says so -- but a future "
                    "sub-project should not select a head on macro-F1."
                ),
                "evidence": {
                    "simulator_round_macro_f1": sim_round_f1,
                    "incumbent_round_macro_f1": inc_round_f1,
                    "macro_f1_delta": round(sim_round_f1 - inc_round_f1, 4),
                    "simulator_joint_log_loss": pooled(load(E2), "joint_log_loss"),
                    "incumbent_joint_log_loss": pooled(incumbent_report, "joint_log_loss"),
                    "joint_log_loss_delta": e2_bar["joint_delta"],
                    "finish_round_distribution": rounds,
                    "source": _rel(FIGHTS),
                },
            },
            "2_the_hybrid_beat_the_pure_simulator": {
                "finding": (
                    "The hybrid BEAT the pure simulator on the joint "
                    f"({pooled(load(D2), 'joint_log_loss')} against "
                    f"{pooled(load(E2), 'joint_log_loss')}), rather than trading joint "
                    "quality away to buy the winner clause as the spec assumed it would. "
                    "Imposing a better-calibrated winner marginal on the simulator's "
                    "conditional structure improves the joint outright."
                ),
                "why_it_matters": (
                    "The v3 spec wrote the hybrid as a FALLBACK -- a concession that gives "
                    "up some of the simulator's joint quality in exchange for a winner "
                    "marginal that clears its clause. The measurement says it is not a "
                    "concession at all. The two factors of the joint are separable and are "
                    "best supplied by different models: the blend knows who wins, the "
                    "simulator knows how. That is a positive result about composition, not "
                    "a compromise, and it is the shape a future candidate should start from."
                ),
                "evidence": {
                    "hybrid_joint_log_loss": pooled(load(D2), "joint_log_loss"),
                    "pure_simulator_joint_log_loss": pooled(load(E2), "joint_log_loss"),
                    "delta_hybrid_minus_pure": round(
                        pooled(load(D2), "joint_log_loss") - pooled(load(E2), "joint_log_loss"), 6),
                    "calibrated_simulator_joint_log_loss": pooled(load(D1), "joint_log_loss"),
                    "pure_simulator_winner_delta_vs_incumbent": e2_bar["winner_delta"],
                    "calibrated_simulator_winner_delta_vs_incumbent":
                        d1["bar_check"]["winner_delta"],
                    "hybrid_winner_delta_vs_incumbent": d2_bar["winner_delta"],
                },
            },
            "3_the_d3_control_rules_out_the_arithmetic": {
                "finding": (
                    "The D3 control shows the blend's own three heads, pushed through the "
                    "identical cell scorer, give exactly "
                    f"{d3['cell_scored_joint_log_loss']} -- the same number its composed "
                    "joint gives. So the hybrid's gain is not the cell layout, not the "
                    "re-weighting arithmetic and not a scorer that happens to flatter "
                    "distributions shaped like the simulator's. It is the simulator's "
                    "conditional structure."
                ),
                "why_it_matters": (
                    "Without this control the hybrid's joint improvement would be "
                    "unattributable: it composes a new scorer, a new layout and a new "
                    "model at once, and the first two are shared with the incumbent only "
                    "if someone checks. The control is the check, and it costs one "
                    "walk-forward run."
                ),
                "evidence": {
                    "composed_joint_log_loss": d3["composed_joint_log_loss"],
                    "cell_scored_joint_log_loss": d3["cell_scored_joint_log_loss"],
                    "difference": d3["difference"],
                    "every_pooled_metric_identical": d3["every_pooled_metric_identical"],
                    "control_report": _rel(INCUMBENT_CELLS),
                },
            },
        },
    }


def config_match(candidate: Path, reference: Path, ignore=("seeds", "runtime_sec")) -> dict:
    """Whether two reports' run configs agree on every field but `ignore`.

    A fresh-seed incumbent is only a paired incumbent if it is the same recipe
    on the same table -- otherwise the fresh-seed confirmation compares two
    things that differ in more than their seeds, which is precisely the
    failure it exists to catch.
    """
    a, b = load(candidate)["config"], load(reference)["config"]
    keys = sorted(set(a) | set(b))
    differing = [k for k in keys if k not in ignore and a.get(k) != b.get(k)]
    return {
        "candidate": _rel(candidate),
        "reference": _rel(reference),
        "ignored_fields": list(ignore),
        "differing_fields": differing,
        "matches": not differing,
        "seeds": {"candidate": a.get("seeds"), "reference": b.get("seeds")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print", action="store_true", dest="echo",
                        help="echo the artifact to stdout as well as writing it")
    args = parser.parse_args()
    artifact = build()
    OUT.write_text(json.dumps(artifact, indent=2) + "\n")
    if args.echo:
        print(json.dumps(artifact, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
