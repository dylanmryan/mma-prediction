"""Assemble SP2.2's decision artifact, including the human call rule 4 reserved.

    python scripts/sp2_2_decision.py            # writes models/walkforward/sp2_2_decision.json
    python scripts/sp2_2_decision.py --print    # ... and echoes it

Reads only committed walk-forward reports, noise floors and prediction dumps
(paths are the constants below), so re-running it reproduces the artifact
exactly. The pre-registration's five decision rules -- and the block that
amends rule 4 -- are quoted out of the plan file itself rather than retyped
here, so the artifact cannot drift from them.

What this script does and does not do:

* It applies rule 2 mechanically -- `mma.walkforward.bar_check` at seeds 0-4
  and again at seeds 5-9 against the fresh-seed paired incumbent -- and reports
  which branches of rules 1, 2, 3 and 5 the measured numbers satisfy.
* It reports the facts rule 4's ECE gate needs to be applied honestly rather
  than mechanically against noise: the gate compares two single numbers whose
  precision had never been measured, so the artifact carries an ECE noise floor
  for both recipes, the same comparison at four bin counts, and the reliability
  curves the two ECEs summarise.
* It does not INVENT a resolution of rule 4. Rule 4's text reserved the call
  for a human ("record it and stop for a human call"); the call was made on
  2026-09-08 -- B1 ships, with the ECE gate re-specified -- and this script
  records it. The gate as written stays in the artifact beside the amended one
  (`ece_gate.rule_as_written` and `ece_gate.amended`), because amending a
  pre-registered rule after seeing results is only defensible if the original
  stays readable next to the replacement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.evaluate import expected_calibration_error, reliability_curve  # noqa: E402
from mma.walkforward import bar_check, slice_comparison  # noqa: E402

WF = ROOT / "models" / "walkforward"
PREDS = WF / "preds"
OUT = WF / "sp2_2_decision.json"
PLAN = ROOT / "docs" / "superpowers" / "plans" / "2026-09-08-sp2-2-blend-experiment.md"

MIN_BAR = 0.003
BIN_COUNTS = (5, 10, 15, 20)  # 10 is `expected_calibration_error`'s default

# The plan file's amendment block. It sits BELOW the numbered decision rules so
# `quoted_rules` still parses the list unchanged and rule 4's original wording
# stays exactly where it was written.
AMENDMENT_START = "<!-- AMENDMENT: rule 4 -->"
AMENDMENT_END = "<!-- END AMENDMENT -->"

# Which arm of the ECE noise floor is which, for the amended gate.
ECE_ARM_INCUMBENT = "incumbent_torch_S0"
ECE_ARM_B1 = "B1_blend_S1"

# --- the reports the artifact reads ----------------------------------------
# S0 = the shipped `base,external` table; S1 = the combined table with SP2.1's
# four blocks restored. `I` is the deployed scorer and the incumbent rule 4
# names. Every *_seeds5 / *_seeds10 report is the same recipe on the same table
# at a disjoint seed set.
INCUMBENT = WF / "torch_external_diffsonly_extslice.json"          # I: S0, seeds 0-4
INCUMBENT_SEEDS5 = WF / "torch_external_diffsonly_seeds5.json"     # I recipe, S0, seeds 5-9
INCUMBENT_SEEDS10 = WF / "torch_external_diffsonly_seeds10.json"   # I recipe, S0, seeds 10-14
PAIRED_S1 = WF / "torch_a1_combined.json"                          # torch arm, S1, seeds 0-4
PAIRED_S1_SEEDS5 = WF / "torch_a1_combined_seeds5.json"            # torch arm, S1, seeds 5-9
B0 = WF / "blend_b0_seeds0.json"
B0_SEEDS5 = WF / "blend_b0_seeds5.json"
B0_SEEDS10 = WF / "blend_b0_seeds10.json"
B1 = WF / "blend_b1.json"
B1_SEEDS5 = WF / "blend_b1_seeds5.json"
B1_SEEDS10 = WF / "blend_b1_seeds10.json"
B1_ISOTONIC = WF / "blend_b1_isotonic.json"
DIAGNOSTICS = {
    "blend_b0_uncalibrated": WF / "blend_b0_uncal.json",
    "blend_b0_weight_0.3": WF / "blend_b0_w03.json",
    "blend_b0_weight_0.7": WF / "blend_b0_w07.json",
    "xgb_member_5seed_S0": WF / "xgb_ens5_s0.json",
    "xgb_member_5seed_S1": WF / "xgb_ens5_s1.json",
    "torch_member_5seed_S1": PAIRED_S1,
}
NOISE_FLOOR_BLEND = WF / "noise_floor_blend.json"
NOISE_FLOOR_ECE = WF / "noise_floor_ece.json"

# Pooled per-row predictions, written by run_walkforward.py --dump-predictions.
# The reports carry metrics and no predictions, so the reliability curves and
# the bin-count sweep would otherwise not be computable at all.
PRED_INCUMBENT = PREDS / "torch_external_diffsonly_extslice.json"
PRED_B1 = PREDS / "blend_b1.json"
PRED_B1_ISOTONIC = PREDS / "blend_b1_isotonic.json"


# --- pure helpers -----------------------------------------------------------


def quoted_rules(plan_text: str) -> dict[str, str]:
    """The pre-registration's numbered decision rules, quoted verbatim.

    Taken out of the plan file rather than retyped, so the artifact cannot
    drift from the rules it claims to quote. The block is the numbered list
    under the "### Decision rule" heading, and it ends at the next heading or
    blank-line-then-non-list; markdown emphasis is left exactly as written.
    """
    lines = plan_text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("### Decision rule")]
    if len(starts) != 1:
        raise ValueError(f"expected exactly one '### Decision rule' heading, found {len(starts)}")
    rules: dict[str, str] = {}
    for line in lines[starts[0] + 1:]:
        if line.startswith("#"):
            break
        stripped = line.strip()
        if not stripped:
            continue
        head, _, rest = stripped.partition(". ")
        if head.isdigit() and rest:
            rules[head] = stripped
        elif rules:  # a non-list, non-blank line after the list has begun ends it
            break
    if sorted(rules) != ["1", "2", "3", "4", "5"]:
        raise ValueError(f"expected rules 1-5, parsed {sorted(rules)}")
    return rules


def quoted_amendment(plan_text: str) -> str:
    """The plan's rule-4 amendment block, quoted verbatim between its markers.

    Taken out of the plan file for the same reason `quoted_rules` is: an
    artifact that states a gate the pre-registration does not state is worth
    nothing. Exactly one delimited block must exist.
    """
    before, sep, rest = plan_text.partition(AMENDMENT_START)
    if not sep or AMENDMENT_START in rest:
        raise ValueError(
            f"expected exactly one {AMENDMENT_START!r} block in the plan file"
        )
    body, sep, after = rest.partition(AMENDMENT_END)
    if not sep or AMENDMENT_END in after:
        raise ValueError(
            f"expected exactly one {AMENDMENT_END!r} closing the amendment block"
        )
    return body.strip()


def pooled_sd(sds, ns) -> float:
    """The pooled sample standard deviation of several equally-trusted arms.

    sqrt(sum((n_i - 1) * s_i^2) / sum(n_i - 1)) -- the ordinary pooled-variance
    estimator. For two arms of three seed sets each it reduces to the RMS of
    the two sds, which is what the amendment's "2 sigma of the pooled spread"
    means: neither arm's own sd sets the tolerance on its own.
    """
    numerator = sum((n - 1) * float(s) ** 2 for s, n in zip(sds, ns))
    denominator = sum(n - 1 for n in ns)
    if denominator <= 0:
        raise ValueError("pooled_sd needs at least two measurements in total")
    return math.sqrt(numerator / denominator)


def amended_ece_gate(incumbent_arm: dict, candidate_arm: dict) -> dict:
    """Rule 4 as amended: mean pooled ECE over three disjoint seed sets each,
    and the candidate may not exceed the incumbent by more than 2 sigma of the
    pooled spread.

    The same shape as every other bar in this project (`mma.walkforward.
    bar_check`, `scripts/noise_floor.py`): a measured difference against a
    measured noise floor, rather than one number against another number of
    unknown precision. `passes` is True when the candidate is not worse by more
    than the tolerance -- a better-calibrated candidate passes outright.
    """
    sd = pooled_sd(
        [incumbent_arm["sigma_seed"], candidate_arm["sigma_seed"]],
        [incumbent_arm["n_reports"], candidate_arm["n_reports"]],
    )
    tolerance = 2.0 * sd
    difference = float(candidate_arm["mean"]) - float(incumbent_arm["mean"])
    return {
        "form": (
            "a candidate's mean pooled ECE across three disjoint seed sets must not "
            "exceed the incumbent's mean across three disjoint seed sets by more than "
            "2 sigma of the pooled spread"
        ),
        "incumbent_mean": round(float(incumbent_arm["mean"]), 6),
        "incumbent_sd": round(float(incumbent_arm["sigma_seed"]), 6),
        "candidate_mean": round(float(candidate_arm["mean"]), 6),
        "candidate_sd": round(float(candidate_arm["sigma_seed"]), 6),
        "pooled_sd": sd,
        "tolerance_2sigma": tolerance,
        "difference_candidate_minus_incumbent": difference,
        "passes": bool(difference <= tolerance),
    }


def spread(values) -> dict:
    """Mean / sample sd / min / max of a set of measurements.

    The sd is the sample (ddof=1) standard deviation, matching
    scripts/noise_floor.py, and is nan for a single value rather than 0 -- one
    run measures no spread at all.
    """
    values = [float(v) for v in values]
    if not values:
        raise ValueError("spread needs at least one value")
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {"n": len(values), "values": values, "mean": round(float(np.mean(values)), 6),
            "sd": sd, "min": min(values), "max": max(values)}


def binning_sensitivity(incumbent: dict, candidate: dict, bin_counts=BIN_COUNTS) -> dict:
    """The ECE gate's comparison recomputed at several equal-width bin counts.

    `mma.evaluate.expected_calibration_error` defaults to 10 equal-width bins
    and rule 4 compares two numbers computed that way. Nothing in the
    pre-registration says 10; if the SIGN of candidate-minus-incumbent depends
    on the bin count then the gate is being decided by a binning artifact, and
    the sweep is here to make that visible either way.

    `gap` is candidate - incumbent, so positive means the candidate is worse
    calibrated (which is what rule 4 gates on). `sign_flips` is true when the
    sign of `gap` is not the same at every bin count.
    """
    rows = []
    for n_bins in bin_counts:
        inc = expected_calibration_error(incumbent["y_winner"], incumbent["p_winner"], n_bins=n_bins)
        cand = expected_calibration_error(candidate["y_winner"], candidate["p_winner"], n_bins=n_bins)
        # Rounded first, then subtracted, so a reader can subtract the two
        # printed numbers and get the printed gap.
        inc, cand = round(inc, 6), round(cand, 6)
        rows.append({
            "n_bins": n_bins,
            "incumbent_ece": inc,
            "candidate_ece": cand,
            "gap": round(cand - inc, 6),
            "candidate_worse": bool(cand > inc),
        })
    signs = {int(np.sign(row["gap"])) for row in rows}
    return {
        "convention": "gap = candidate - incumbent; positive = candidate worse calibrated",
        "default_n_bins": 10,
        "rows": rows,
        "sign_flips": len(signs) > 1,
        "candidate_worse_at_every_bin_count": all(row["candidate_worse"] for row in rows),
    }


def worst_bins(curve: list[dict], k: int = 3) -> list[dict]:
    """The `k` bins contributing most to ECE, largest contribution first.

    Contribution is `weight * |gap|` -- the bin's own term in the ECE sum -- so
    a large miscalibration in a bin holding 1% of fights ranks below a small
    one in a bin holding 30%, which is the distinction the gate cannot see.
    """
    filled = [row for row in curve if row["n"]]
    ranked = sorted(filled, key=lambda row: -(row["weight"] * abs(row["gap"])))
    return [{**row, "ece_contribution": round(row["weight"] * abs(row["gap"]), 6)}
            for row in ranked[:k]]


def fresh_seed_branch(seeds04: dict, seeds59: dict) -> dict:
    """Which half of rule 2 a candidate satisfies. Both are required to ship."""
    return {
        "clears_at_seeds_0_4": bool(seeds04["ships"]),
        "clears_at_seeds_5_9": bool(seeds59["ships"]),
        "rule_2_satisfied": bool(seeds04["ships"] and seeds59["ships"]),
    }


# --- report plumbing --------------------------------------------------------


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def pooled(report: dict, metric: str = "winner_log_loss") -> float:
    return float(report["pooled"][metric])


def profile(path: Path) -> dict:
    """Pooled, per-fold and per-slice metrics for one committed report."""
    report = load(path)
    config = report["config"]
    return {
        "report": str(path.relative_to(ROOT)),
        "candidate": config.get("candidate"),
        "seeds": config.get("seeds"),
        "drop_columns": config.get("drop_columns"),
        "blend_weight": config.get("blend_weight"),
        "blend_calibrated": config.get("blend_calibrated"),
        # Committed blend reports predate the flag and are all temperature-scaled;
        # only the isotonic remediation writes the key.
        "blend_calibrator": (config.get("blend_calibrator", "temperature")
                             if config.get("candidate") == "blend" else None),
        "pooled": report["pooled"],
        "folds": {year: fold["winner_log_loss"] for year, fold in report["folds"].items()},
        "fold_ece": {year: fold["ece"] for year, fold in report["folds"].items()},
        "slices": {name: {"n": s["n"], "winner_log_loss": s["winner_log_loss"], "ece": s["ece"]}
                   for name, s in report["slices"].items()},
    }


def arm(path: Path, incumbent: Path, sigma: float) -> dict:
    out = profile(path)
    out["bar_check"] = bar_check(load(path), load(incumbent), sigma)
    out["slice_deltas"] = slice_comparison(load(path), load(incumbent))
    out["paired_incumbent"] = str(incumbent.relative_to(ROOT))
    return out


def _round_row(row: dict) -> dict:
    """6 dp everywhere, so `lo`/`hi` read 0.6 rather than 0.6000000000000001."""
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}


def curve_block(path: Path, n_bins: int = 10) -> dict:
    dump = load(path)
    curve = reliability_curve(dump["y_winner"], dump["p_winner"], n_bins=n_bins)
    ece = expected_calibration_error(dump["y_winner"], dump["p_winner"], n_bins=n_bins)
    return {
        "predictions": str(path.relative_to(ROOT)),
        "n": dump["n"], "n_bins": n_bins,
        "pooled_ece": round(ece, 6),
        "curve": [_round_row(row) for row in curve],
        "top_ece_contributors": [_round_row(row) for row in worst_bins(curve)],
    }


def build() -> dict:
    plan_text = PLAN.read_text()
    rules = quoted_rules(plan_text)
    amendment = quoted_amendment(plan_text)
    sigma_blend = float(load(NOISE_FLOOR_BLEND)["sigma_seed"])
    bar = round(max(MIN_BAR, 2.0 * sigma_blend), 6)
    ece_floor = load(NOISE_FLOOR_ECE)

    incumbent_ece = pooled(load(INCUMBENT), "ece")

    b0_04 = arm(B0, INCUMBENT, sigma_blend)
    b0_59 = arm(B0_SEEDS5, INCUMBENT_SEEDS5, sigma_blend)
    b1_04 = arm(B1, INCUMBENT, sigma_blend)
    b1_04_paired = bar_check(load(B1), load(PAIRED_S1), sigma_blend)
    b1_59 = arm(B1_SEEDS5, PAIRED_S1_SEEDS5, sigma_blend)

    b0_branch = fresh_seed_branch(b0_04["bar_check"], b0_59["bar_check"])
    b1_branch = fresh_seed_branch(b1_04["bar_check"], b1_59["bar_check"])

    inc_pred, b1_pred = load(PRED_INCUMBENT), load(PRED_B1)
    sensitivity = binning_sensitivity(inc_pred, b1_pred)

    iso = profile(B1_ISOTONIC)
    iso_bar = bar_check(load(B1_ISOTONIC), load(PAIRED_S1), sigma_blend)

    b1_minus_b0 = round(pooled(load(B1)) - pooled(load(B0)), 6)

    amended_gate = amended_ece_gate(ece_floor["arms"][ECE_ARM_INCUMBENT],
                                    ece_floor["arms"][ECE_ARM_B1])

    return {
        "experiment": "SP2.2 calibrated two-model blend",
        "pre_registration": str(PLAN.relative_to(ROOT)),
        "generated_by": "scripts/sp2_2_decision.py",
        "decided": True,
        "decision": {
            "outcome": "ship B1",
            "date": "2026-09-08",
            "taken_by": "human call reserved by rule 4",
            "what_ships": (
                "B1: the equal-weight (0.5) blend of a 5-seed XGBoost ensemble and the "
                "5-seed torch ensemble, temperature-scaled after averaging, on S1 "
                "(base,external,trajectory,notice,context,opponent_adjusted) with "
                "external_missing, same_country, notice_unknown, home_country_a and "
                "home_country_b held out of both model matrices."
            ),
            "ships": {
                "B1": True,
                "B0": False,
                "isotonic_remediation": False,
            },
            "why": (
                "Rule 2 is satisfied by B1 and only by B1: it clears the bar against I at "
                "seeds 0-4 and clears it again at seeds 5-9 against its own fresh-seed "
                "paired incumbent. Rule 3 does not apply (B0 does not clear, so there is "
                "no choice between candidates) and rule 5 does not apply (something "
                "clears). Rule 4 is the only rule left, and its gate AS WRITTEN fails: "
                "B1's pooled ECE 0.0124 against the single incumbent value 0.0088 the "
                "rule names. That rule reserved the case for a human, and the human "
                "amended the gate -- see `ece_gate.amended` and the amendment quoted "
                "from the plan file -- because the incumbent's own ECE ranges "
                "0.0088-0.0160 across three disjoint seed sets, the 0.0036 gap is "
                "smaller than one sd of the metric it tests, and the sign of the gap "
                "reverses at 5, 15 and 20 bins. Under the amended gate B1 passes."
            ),
            "recorded_weakness": (
                "The ECE gate was re-specified AFTER the numbers were seen. That is the "
                "weakest part of this experiment and is recorded rather than smoothed "
                "over: the original gate is preserved verbatim in the plan and in "
                "`ece_gate.rule_as_written`, the replacement is the project's standard "
                "measured-difference-against-a-measured-noise-floor form rather than a "
                "bespoke threshold, and the log-loss bar B1 actually cleared was never "
                "touched."
            ),
        },
        "incumbent": {
            "report": str(INCUMBENT.relative_to(ROOT)),
            "note": "I: the deployed scorer, a 5-seed torch ensemble on S0",
            "pooled_winner_log_loss": pooled(load(INCUMBENT)),
            "pooled_ece": incumbent_ece,
        },
        "noise_floors": {
            "blend_winner_log_loss": {
                "source": str(NOISE_FLOOR_BLEND.relative_to(ROOT)),
                "sigma_blend": sigma_blend,
                "bar": bar,
                "seed_sets": load(NOISE_FLOOR_BLEND)["seed_sets"],
                "pooled_winner_log_loss": load(NOISE_FLOOR_BLEND)["pooled_winner_log_loss"],
            },
            "ece": {
                "source": str(NOISE_FLOOR_ECE.relative_to(ROOT)),
                "seed_sets_measured": {
                    "incumbent_recipe_S0": [str(q.relative_to(ROOT)) for q in
                                            (INCUMBENT, INCUMBENT_SEEDS5, INCUMBENT_SEEDS10)],
                    "B1_S1": [str(q.relative_to(ROOT)) for q in (B1, B1_SEEDS5, B1_SEEDS10)],
                    "B0_S0": [str(q.relative_to(ROOT)) for q in (B0, B0_SEEDS5, B0_SEEDS10)],
                },
                "B0_pooled_ece_by_seed_set": [pooled(load(q), "ece") for q in
                                              (B0, B0_SEEDS5, B0_SEEDS10)],
                "why": (
                    "Rule 4 compares two single ECE numbers with no known precision. "
                    "This is that precision: the same recipe run on three disjoint seed "
                    "sets, exactly as sigma_blend was measured for log-loss. It is a "
                    "measurement, not a threshold -- no bar is derived from it, because "
                    "the pre-registration states none."
                ),
                **ece_floor,
            },
        },
        "candidates": {
            "B0": {
                "role": "primary candidate: equal-weight blend on S0",
                "seeds_0_4": b0_04,
                "seeds_5_9": b0_59,
                "rule_2": b0_branch,
                "pooled_ece": pooled(load(B0), "ece"),
                "ece_vs_incumbent": round(pooled(load(B0), "ece") - incumbent_ece, 6),
            },
            "B1": {
                "role": "secondary candidate: the identical construction on S1",
                "seeds_0_4": b1_04,
                "seeds_0_4_against_the_paired_S1_torch_arm": b1_04_paired,
                "seeds_5_9": b1_59,
                "rule_2": b1_branch,
                "pooled_ece": pooled(load(B1), "ece"),
                "ece_vs_incumbent": round(pooled(load(B1), "ece") - incumbent_ece, 6),
            },
        },
        "diagnostics": {name: profile(path) for name, path in DIAGNOSTICS.items()},
        "ece_gate": {
            "rule_as_written": rules["4"],
            "incumbent_pooled_ece": incumbent_ece,
            "B0_pooled_ece": pooled(load(B0), "ece"),
            "B1_pooled_ece": pooled(load(B1), "ece"),
            "B1_fails_the_gate_as_written": bool(pooled(load(B1), "ece") > incumbent_ece),
            "gap_as_written": round(pooled(load(B1), "ece") - incumbent_ece, 6),
            "incumbent_pooled_ece_by_seed_set": {
                "0-4 (the number rule 4 names)": pooled(load(INCUMBENT), "ece"),
                "5-9": pooled(load(INCUMBENT_SEEDS5), "ece"),
                "10-14": pooled(load(INCUMBENT_SEEDS10), "ece"),
            },
            "B1_pooled_ece_by_seed_set": {
                "0-4": pooled(load(B1), "ece"),
                "5-9": pooled(load(B1_SEEDS5), "ece"),
                "10-14": pooled(load(B1_SEEDS10), "ece"),
            },
            "binning_sensitivity": sensitivity,
            "reliability_curves": {
                "incumbent": curve_block(PRED_INCUMBENT),
                "B1": curve_block(PRED_B1),
                "B1_isotonic_remediation": curve_block(PRED_B1_ISOTONIC),
            },
            "amended": {
                "label": "RULE 4 AS AMENDED 2026-09-08, AFTER SEEING THE NUMBERS",
                "text": amendment,
                "quoted_verbatim_from": str(PLAN.relative_to(ROOT)),
                "source": str(NOISE_FLOOR_ECE.relative_to(ROOT)),
                "incumbent_arm": ECE_ARM_INCUMBENT,
                "candidate_arm": ECE_ARM_B1,
                "incumbent_pooled_ece": ece_floor["arms"][ECE_ARM_INCUMBENT]["pooled_ece"],
                "candidate_pooled_ece": ece_floor["arms"][ECE_ARM_B1]["pooled_ece"],
                **amended_gate,
                "weakness": (
                    "Amending a pre-registered gate after seeing results is a real "
                    "weakness. The mitigations, and they are mitigations rather than a "
                    "defence: the amendment is written down and dated in the plan file, "
                    "the original wording is preserved above and in `rule_as_written`, "
                    "the replacement is the same form every other bar in this project "
                    "uses rather than a threshold chosen to fit, and the log-loss bar "
                    "the candidate actually cleared was never touched."
                ),
            },
            "resolved_here": (
                "Rule 4 sent this to a human; the human took it on 2026-09-08 and it is "
                "recorded in `decision`. The facts above are what that call rested on."
            ),
        },
        "remediation_isotonic": {
            "label": "POST-HOC VARIANT -- NOT A CANDIDATE AND NOT A SHIPPING FORM",
            "what_it_changes": (
                "Only the calibration step the pre-registration already specifies. The "
                "candidate is defined as 'temperature-scaled after averaging'; this "
                "fits an isotonic regression on the same fold inner-validation year "
                "instead, on the reasoning that one scalar cannot fix a SHAPE mismatch "
                "between two differently-calibrated probability streams."
            ),
            "provenance": (
                "Measured after seeing B1 fail the ECE gate. It has no fresh-seed "
                "confirmation of its own; rule 2 would require one before it could ship "
                "in any form, and rule 3's choice between candidates does not contemplate "
                "it at all."
            ),
            "report": iso,
            "bar_check_against_the_paired_S1_torch_arm": iso_bar,
            "slice_deltas_against_the_paired_S1_torch_arm": slice_comparison(
                load(B1_ISOTONIC), load(PAIRED_S1)),
            "vs_B1_temperature": {
                "winner_log_loss": round(pooled(load(B1_ISOTONIC)) - pooled(load(B1)), 6),
                "ece": round(pooled(load(B1_ISOTONIC), "ece") - pooled(load(B1), "ece"), 6),
            },
            "vs_incumbent_I": {
                "winner_log_loss": round(pooled(load(B1_ISOTONIC)) - pooled(load(INCUMBENT)), 6),
                "ece": round(pooled(load(B1_ISOTONIC), "ece") - incumbent_ece, 6),
            },
        },
        "rules": {
            "quoted_verbatim_from": str(PLAN.relative_to(ROOT)),
            "text": rules,
            "amendment": amendment,
            "status": {
                "1": {
                    "satisfied": True,
                    "evidence": (
                        "Paired incumbents exist on both tables at both seed sets: "
                        f"{INCUMBENT.name} / {INCUMBENT_SEEDS5.name} on S0 and "
                        f"{PAIRED_S1.name} / {PAIRED_S1_SEEDS5.name} on S1, each the "
                        "deployed recipe with the candidate's own drop-columns. The S0 "
                        "incumbent was re-run under this branch's code and reproduces "
                        "its committed report in every pooled metric, every fold metric, "
                        "every slice and every per-fold fit budget; the only difference "
                        "anywhere in the JSON is a `config.model_seed: null` key, which "
                        "SP2.1 added to the run-config schema after that report was "
                        "written and which no learner reads."
                    ),
                },
                "2": {
                    "B0": b0_branch,
                    "B1": b1_branch,
                    "satisfied_by": [name for name, branch in
                                     (("B0", b0_branch), ("B1", b1_branch))
                                     if branch["rule_2_satisfied"]],
                },
                "3": {
                    "applies": bool(b0_branch["rule_2_satisfied"] and b1_branch["rule_2_satisfied"]),
                    "B1_minus_B0": b1_minus_b0,
                    "margin_required": bar,
                    "note": (
                        "Rule 3 chooses between two candidates that BOTH clear rule 2. "
                        f"Clearing it: {[n for n, br in (('B0', b0_branch), ('B1', b1_branch)) if br['rule_2_satisfied']] or 'neither'}."
                    ),
                },
                "4": {
                    "applies": bool(b1_branch["rule_2_satisfied"]
                                    and pooled(load(B1), "ece") > incumbent_ece),
                    "resolved": True,
                    "gate_applied": "amended",
                    "fails_as_written": bool(pooled(load(B1), "ece") > incumbent_ece),
                    "passes_as_amended": bool(amended_gate["passes"]),
                    "reason": (
                        "The rule reserved this for a human call; the call was taken on "
                        "2026-09-08 and amended the gate (see `ece_gate.amended`). B1 "
                        "passes the amended gate, so rule 4 no longer blocks it."
                    ),
                },
                "5": {
                    "applies": not (b0_branch["rule_2_satisfied"] or b1_branch["rule_2_satisfied"]),
                    "note": "rule 5 is the nothing-clears branch; B1 clears on log-loss.",
                },
            },
        },
        "human_call": {
            "reserved_by": "rule 4",
            "rule_as_written": rules["4"],
            "taken": "2026-09-08",
            "outcome": "B1 ships, with the ECE gate re-specified (see rules.amendment)",
            "what_the_call_rested_on": [
                "the ECE noise floors, which say how much of the ECE gap is measurable: "
                "the incumbent's own ECE ranges 0.0088-0.0160 (sd 0.0038) and B1's "
                "0.0124-0.0144 (sd 0.0010), so the 0.0036 gap the gate turned on is "
                "smaller than one sd of the incumbent's own metric",
                "the bin-count sweep, which says the gap does not survive a different "
                "binning of the same predictions: B1 is worse at 10 bins and better at "
                "5, 15 and 20, and 10 is only `expected_calibration_error`'s default",
                "the reliability curves, which say the miscalibration is a couple of "
                "bins on each side rather than a systematic slope on either",
                "the isotonic remediation, which fails clearly on both axes and is a "
                "post-hoc variant that would need a fresh-seed confirmation of its own "
                "before it could ship in any form -- it does not ship",
            ],
            "not_decided_by_this_script": (
                "This script applies rules 1, 2, 3 and 5 mechanically and reports the "
                "facts rule 4 needs. The amendment to rule 4 and the decision to ship "
                "B1 are a human's, recorded here and in the plan file; nothing in this "
                "script chose them."
            ),
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--print", dest="echo", action="store_true")
    args = parser.parse_args(argv)
    decision = build()
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    if args.echo:
        print(json.dumps(decision, indent=2))
    status = decision["rules"]["status"]
    gate = decision["ece_gate"]["amended"]
    print(f"rule 2 satisfied by: {status['2']['satisfied_by'] or 'nothing'}")
    print(f"rule 4 applies: {status['4']['applies']}, resolved: {status['4']['resolved']} "
          f"({status['4']['gate_applied']} gate)")
    print(f"amended ECE gate: {gate['difference_candidate_minus_incumbent']:+.5f} against a "
          f"2-sigma tolerance of {gate['tolerance_2sigma']:.5f} -> passes={gate['passes']}")
    print(f"decision: {decision['decision']['outcome']} ({decision['decision']['taken_by']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
