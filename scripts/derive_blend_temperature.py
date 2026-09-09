"""Derive -- and honestly score -- the fixed post-average temperature the blend SERVES.

    python scripts/derive_blend_temperature.py           # writes models/walkforward/blend_temperature.json
    python scripts/derive_blend_temperature.py --print   # ... and echoes the comparison

The defect this exists to fix
-----------------------------
`mma.candidates.BlendCandidate` fits ONE TEMPERATURE PER FOLD, on that fold's
inner-validation year, and `models/walkforward/blend_b1.json` reports the
pooled log-loss and ECE of that form. `mma.inference.BlendedPredictor` cannot
do this: deployment has no held-out year, so it applies a single fixed
temperature from `models/blend.json` to every prediction it ever makes. Those
are two different scorers, and only the second one ships.

The fixed value shipped before this script was the MEDIAN of the eight
per-fold fits (0.80), a rule borrowed from `scripts/run_walkforward.
fixed_budget_from`, where it selects a *training budget* (epochs, trees). A
median epoch count is a defensible central tendency of a budget; a median
temperature has no calibration justification, and no harness run ever measured
what it costs. Re-scoring the committed dump under it gives pooled ECE 0.0177
against the 0.0124 the project published -- so the published calibration
number described a form that does not ship.

The rule this script applies, stated before the numbers
-------------------------------------------------------
Deployment knows every prior fight and no future one. The honest analogue on
the harness is a WALK-FORWARD TEMPERATURE: for fold Y, fit one temperature on
the pooled out-of-fold predictions of every fold strictly before Y, then apply
it to Y. That is exactly the information a deployed model has, so pooling
those folds gives an honest out-of-sample score for the FIXED-temperature
form. The first fold has no earlier fold and keeps the harness's own inner-val
fit; the artifact and this docstring both say so, because it is the one row
whose temperature is not walk-forward.

Four forms are scored, and the deployed one is chosen by this rule, fixed
before any of them was computed:

* **R2, walk-forward** -- PREFERRED, and the one deployed. It is the only
  candidate that is both fixed at serving time and validated without using the
  evaluation rows to choose it.
* **R1, median-of-per-fold** -- the incumbent 0.80. Used only if R2 is
  unavailable for a structural reason (it is not).
* **R3, one temperature fitted on all out-of-fold rows** -- reported for
  context and NOT a candidate: it is fitted on the very rows it is then scored
  on, so its pooled metrics are in-sample and optimistic.
* **per-fold fitted** -- the form the harness measured. Not deployable; it is
  here so the published harness numbers stay visible beside the deployed ones.

The DEPLOYED temperature is R2's rule run one fold past the last: deployment
is the fold after 2025, every fold is "before" it, so the rule fits on all of
them. That value coincides arithmetically with R3's fitted value -- the same
rows, the same grid -- and the coincidence is not a change of rule. The
difference between R2 and R3 is never which rows are FITTED on at deployment,
it is which rows are SCORED: R2 never scores a row whose temperature saw it,
R3 scores every row with a temperature that did. So R2's pooled numbers are
the honest estimate of the deployed procedure, and R3's are not.

Everything is recomputed from two committed artifacts -- the report and its
per-row prediction dump -- so re-running reproduces the JSON exactly. The
inversion that recovers the pre-temperature probabilities
(`sigmoid(logit(p)*T_fold)`) is asserted against the report's own published
pooled log-loss and ECE before anything else runs: if the dump and the report
disagree, this script raises rather than deriving a number from them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mma.blend import apply_temperature, logit  # noqa: E402
from mma.evaluate import (  # noqa: E402
    accuracy, brier_score, expected_calibration_error, log_loss,
)
from mma.models.train_loop import fit_temperature  # noqa: E402
from scripts.run_walkforward import fixed_budget_from  # noqa: E402

WF = ROOT / "models" / "walkforward"
REPORT = WF / "blend_b1.json"
DUMP = WF / "preds" / "blend_b1.json"
ARTIFACT = WF / "blend_temperature.json"

BIN_COUNTS = (5, 10, 15, 20)  # 10 is `expected_calibration_error`'s default
# The reports round pooled metrics to 4 dp; matching them means every number
# this artifact publishes is comparable to the ones already in the record
# without a reader having to re-round anything.
DP = 4


# --- the arithmetic ---------------------------------------------------------


def fold_order(years: np.ndarray) -> list[int]:
    return sorted({int(v) for v in years})


def invert_per_fold(p_scored, years, fold_temperatures: dict) -> np.ndarray:
    """Recover the pre-temperature probability of every dumped row.

    `BlendCandidate` writes `sigmoid(logit(p_pre) / T_fold)` and nothing else,
    so `sigmoid(logit(p) * T_fold)` returns `p_pre` exactly (up to the clip in
    `mma.blend.logit`, which never binds on a blended probability).
    """
    p_scored = np.asarray(p_scored, dtype=float)
    years = np.asarray(years)
    out = np.empty_like(p_scored)
    for year, temperature in fold_temperatures.items():
        mask = years == year
        out[mask] = 1.0 / (1.0 + np.exp(-logit(p_scored[mask]) * float(temperature)))
    return out


def apply_per_fold(pre, years, temperatures: dict) -> np.ndarray:
    """Temperature-scale each fold's rows by that fold's temperature.

    Uses `mma.blend.apply_temperature`, the same function the deployed scorer
    calls, so a "re-scored" probability here is the served arithmetic.
    """
    pre = np.asarray(pre, dtype=float)
    years = np.asarray(years)
    out = np.empty_like(pre)
    for year, temperature in temperatures.items():
        mask = years == year
        out[mask] = apply_temperature(pre[mask], temperature)
    return out


def fit_temperature_on(pre, y) -> float:
    """The harness's own temperature fit (`mma.models.train_loop.fit_temperature`,
    a 0.5-3.0 grid at 0.01), rounded to the grid's own 2 dp so the value a
    reader sees is the value that is applied."""
    return round(float(fit_temperature(logit(pre), np.asarray(y, dtype=float))), 2)


def walkforward_temperatures(pre, y, years, first_fold_temperature: float) -> dict:
    """Fold Y's temperature, fitted on the pooled rows of the folds BEFORE Y.

    The first fold has no predecessor and takes `first_fold_temperature` (the
    harness's own inner-validation fit for that fold) unchanged.
    """
    years = np.asarray(years)
    order = fold_order(years)
    temperatures = {order[0]: round(float(first_fold_temperature), 2)}
    for i, year in enumerate(order[1:], start=1):
        prior = np.isin(years, order[:i])
        temperatures[year] = fit_temperature_on(np.asarray(pre)[prior],
                                                np.asarray(y)[prior])
    return temperatures


def deployed_temperature(pre, y) -> float:
    """R2's rule at serving time: every fold is before the fold being served."""
    return fit_temperature_on(pre, y)


# --- scoring ----------------------------------------------------------------


def pooled_metrics(y, p) -> dict:
    """The report's own pooled shape, on this rule's re-scored probabilities.

    `accuracy` is here because it is invariant to temperature -- scaling is
    monotone and fixes 0.5 -- so a reader can see at a glance that only the
    calibration-sensitive metrics moved. `brier` does move, and the model card
    quotes it, so it cannot be taken from the harness form.
    """
    return {
        "n": int(len(y)),
        "winner_log_loss": round(log_loss(y, p), DP),
        "accuracy": round(accuracy(y, p), DP),
        "brier": round(brier_score(y, p), DP),
        "ece": {str(b): round(expected_calibration_error(y, p, n_bins=b), DP)
                for b in BIN_COUNTS},
    }


def per_fold_metrics(y, p, years, temperatures: dict) -> dict:
    y, p, years = np.asarray(y), np.asarray(p), np.asarray(years)
    rows = {}
    for year in fold_order(years):
        mask = years == year
        rows[str(year)] = {
            "temperature": temperatures[year],
            **pooled_metrics(y[mask], p[mask]),
        }
    return rows


def score_rule(y, p_scored, years, temperatures: dict) -> dict:
    return {
        "pooled": pooled_metrics(y, p_scored),
        "folds": per_fold_metrics(y, p_scored, years, temperatures),
        "temperature_spread": {
            "min": min(temperatures.values()),
            "max": max(temperatures.values()),
            "distinct": sorted({float(v) for v in temperatures.values()}),
        },
    }


# --- inputs and the round-trip guard ----------------------------------------


def load_inputs(report_path: Path = REPORT, dump_path: Path = DUMP) -> dict:
    report = json.loads(Path(report_path).read_text())
    dump = json.loads(Path(dump_path).read_text())
    years = np.asarray(dump["fold_year"], dtype=int)
    fold_temperatures = {int(year): float(t) for year, t
                         in zip(report["fold_years"], report["fit_info"]["temperature"])}
    if sorted(fold_temperatures) != fold_order(years):
        raise ValueError(
            f"report folds {sorted(fold_temperatures)} do not match the dump's "
            f"{fold_order(years)}"
        )
    return {
        "report": report,
        "report_path": Path(report_path),
        "dump_path": Path(dump_path),
        "years": years,
        "y": np.asarray(dump["y_winner"], dtype=float),
        "p_scored": np.asarray(dump["p_winner"], dtype=float),
        "fold_temperatures": fold_temperatures,
    }


def verify_round_trip(inputs: dict) -> dict:
    """Invert the per-fold temperatures, re-apply them, and require the result
    to reproduce the SOURCE REPORT's own published pooled log-loss and ECE.

    Raises rather than returning a flag. Every rule below is scored on the
    recovered pre-temperature probabilities, so if this does not hold the dump
    and the report describe different runs and nothing downstream means
    anything.
    """
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    back = apply_per_fold(pre, inputs["years"], inputs["fold_temperatures"])
    max_error = float(np.abs(back - inputs["p_scored"]).max())
    pooled = inputs["report"]["pooled"]
    got_ll = round(log_loss(inputs["y"], back), DP)
    got_ece = round(expected_calibration_error(inputs["y"], back), DP)
    if max_error > 1e-12 or got_ll != pooled["winner_log_loss"] or got_ece != pooled["ece"]:
        raise ValueError(
            "round-trip failed: the committed dump does not reproduce "
            f"{inputs['report_path'].name}. max |p - reapplied| = {max_error:.3e}; "
            f"log-loss {got_ll} vs {pooled['winner_log_loss']}; "
            f"ECE {got_ece} vs {pooled['ece']}. STOP -- do not derive a "
            "temperature from these inputs."
        )
    return {
        "checked": "invert each fold's fitted temperature, re-apply it, and "
                   "re-score: the result must equal the source report's own "
                   "pooled winner log-loss and ECE at the report's 4 dp",
        "max_abs_probability_error": max_error,
        "winner_log_loss": got_ll,
        "report_winner_log_loss": pooled["winner_log_loss"],
        "ece_10_bins": got_ece,
        "report_ece": pooled["ece"],
        "passed": True,
    }


# --- the artifact -----------------------------------------------------------


def build_record(report_path: Path = REPORT, dump_path: Path = DUMP) -> dict:
    inputs = load_inputs(report_path, dump_path)
    round_trip = verify_round_trip(inputs)

    y, years = inputs["y"], inputs["years"]
    fold_temps = inputs["fold_temperatures"]
    pre = invert_per_fold(inputs["p_scored"], years, fold_temps)
    order = fold_order(years)

    # per-fold fitted: the harness's own form, re-scored from the dump.
    harness_temps = {year: round(float(t), 2) for year, t in fold_temps.items()}
    harness = score_rule(y, apply_per_fold(pre, years, fold_temps), years, harness_temps)

    # R1: the median rule that shipped before this derivation.
    r1_t = float(fixed_budget_from(inputs["report"])["temperature"])
    r1_temps = {year: r1_t for year in order}
    r1 = score_rule(y, apply_per_fold(pre, years, r1_temps), years, r1_temps)

    # R2: the walk-forward rule -- fold Y's temperature fitted on folds < Y.
    r2_temps = walkforward_temperatures(pre, y, years,
                                        first_fold_temperature=fold_temps[order[0]])
    r2 = score_rule(y, apply_per_fold(pre, years, r2_temps), years, r2_temps)
    deployed_t = deployed_temperature(pre, y)

    # R3: one temperature fitted on every out-of-fold row, then scored on them.
    r3_t = fit_temperature_on(pre, y)
    r3_temps = {year: r3_t for year in order}
    r3 = score_rule(y, apply_per_fold(pre, years, r3_temps), years, r3_temps)

    return {
        "what": (
            "the fixed post-average temperature models/blend.json serves, and the "
            "honest pooled score of the fixed-temperature form -- derived by "
            "re-scoring the committed per-row dump under each candidate rule"
        ),
        "generated_by": "scripts/derive_blend_temperature.py",
        "source_report": str(Path(report_path).relative_to(ROOT)),
        "source_predictions": str(Path(dump_path).relative_to(ROOT)),
        "why": (
            "The harness fits one temperature per fold on that fold's inner-validation "
            "year; deployment has no held-out year and applies one fixed value to every "
            "prediction. They are different scorers and only the second ships, so the "
            "harness's pooled ECE does not describe the deployed model. Before this "
            "artifact existed the deployed value was the median of the per-fold fits "
            "(0.80) -- a rule borrowed from training-BUDGET derivation "
            "(scripts.run_walkforward.fixed_budget_from), where a median epoch count is "
            "a central tendency of a budget. It has no calibration justification and no "
            "harness run validated it."
        ),
        "selection_rule": (
            "Stated before the numbers were computed: prefer R2, the walk-forward "
            "temperature, because it is the only rule that is both fixed at serving "
            "time and validated without using the evaluation rows to choose it. R1 only "
            "if R2 were unavailable for a structural reason. R3 is reported for context "
            "and is not a candidate."
        ),
        "round_trip_verification": round_trip,
        "fold_years": order,
        "harness_per_fold_temperatures": {str(k): round(float(v), 2)
                                          for k, v in fold_temps.items()},
        "rules": {
            "per_fold_fitted": {
                "label": "the form the harness measured (NOT deployable)",
                "candidate": False,
                "deployment_temperature": None,
                "note": (
                    "One temperature per fold, fitted on that fold's inner-validation "
                    "year. Reproduces models/walkforward/blend_b1.json's pooled numbers "
                    "exactly; it is here so the published harness figures stay visible "
                    "beside the deployed ones. Deployment cannot do this: there is no "
                    "held-out year to fit on."
                ),
                **harness,
            },
            "R1": {
                "label": "median of the per-fold fits (the rule that shipped before)",
                "candidate": True,
                "deployment_temperature": r1_t,
                "note": (
                    "scripts.run_walkforward.fixed_budget_from's median rule, applied to "
                    "fit_info.temperature. Fixed at serving time, but the rule is "
                    "borrowed from training-budget derivation and nothing validated it "
                    "as a calibration rule."
                ),
                **r1,
            },
            "R2": {
                "label": "walk-forward: fold Y uses a temperature fitted on folds < Y",
                "candidate": True,
                "deployment_temperature": deployed_t,
                "note": (
                    f"The first fold ({order[0]}) has no earlier fold and keeps the "
                    "harness's own inner-validation fit; every later fold's temperature "
                    "is fitted on the pooled out-of-fold rows of the strictly earlier "
                    "folds only, which is exactly the information a deployed model has. "
                    "The pooled numbers here are therefore an honest out-of-sample score "
                    "for the fixed-temperature form. `deployment_temperature` is the same "
                    "rule run one fold past the last: at serving time every fold is "
                    "'before', so it is fitted on all of them."
                ),
                "per_fold_temperature": {str(k): v for k, v in r2_temps.items()},
                **r2,
            },
            "R3": {
                "label": "one temperature fitted on every out-of-fold row",
                "candidate": False,
                "deployment_temperature": r3_t,
                "note": (
                    "Reported for context only: it is fitted on the very rows it is then "
                    "scored on, so its pooled metrics are in-sample on the evaluation "
                    "rows and optimistic. Its FITTED VALUE coincides with R2's "
                    "deployment temperature -- same rows, same grid -- which is "
                    "arithmetic, not a change of rule: R2 and R3 differ in which rows are "
                    "SCORED, and R2 never scores a row whose temperature saw it."
                ),
                **r3,
            },
        },
        "deployed": {
            "rule": "R2",
            "temperature": deployed_t,
            "derivation": (
                "walk-forward temperature: fitted by mma.models.train_loop.fit_temperature "
                "on the pooled out-of-fold pre-temperature predictions of every fold "
                "before the one being served (at deployment, all of them), via "
                "scripts/derive_blend_temperature.py"
            ),
            "honest_pooled": r2["pooled"],
            "honest_pooled_is": (
                "R2's walk-forward pooled score -- the out-of-sample estimate of what "
                "this fixed-temperature procedure achieves. It is the number the project "
                "publishes for the deployed model."
            ),
            "in_sample_at_the_deployed_value": r3["pooled"],
            "in_sample_note": (
                "The same rows scored at the deployed temperature itself. Optimistic, "
                "because that temperature was fitted on them; recorded so the gap between "
                "the honest and the flattering figure is visible."
            ),
            "supersedes": {
                "rule": "R1",
                "temperature": r1_t,
                "pooled": r1["pooled"],
                "note": (
                    "What the deployed artifact carried before this derivation, re-scored "
                    "here for the first time."
                ),
            },
            "vs_harness_form": {
                "winner_log_loss": round(r2["pooled"]["winner_log_loss"]
                                         - harness["pooled"]["winner_log_loss"], DP),
                "ece_10_bins": round(r2["pooled"]["ece"]["10"]
                                     - harness["pooled"]["ece"]["10"], DP),
                "convention": "deployed minus harness; negative = the deployed form is better",
            },
            "seed_set": (
                "seeds 0-4 only. models/walkforward/preds/ carries a dump for blend_b1 "
                "alone, so the deployed form cannot be re-scored on the seeds 5-9 and "
                "10-14 reports; its ECE here is one measurement, not a three-seed-set "
                "mean like the amended rule-4 gate's arms."
            ),
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--predictions", type=Path, default=DUMP)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    parser.add_argument("--print", dest="echo", action="store_true")
    args = parser.parse_args(argv)

    record = build_record(args.report, args.predictions)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    if args.echo:
        print(json.dumps(record, indent=2))

    print(f"round-trip: {record['round_trip_verification']['winner_log_loss']} / "
          f"{record['round_trip_verification']['ece_10_bins']} reproduces "
          f"{record['source_report']} exactly")
    header = f"{'rule':<18}{'T':>8}{'log-loss':>10}" + "".join(
        f"{'ECE@' + str(b):>10}" for b in BIN_COUNTS)
    print(header)
    for name, rule in record["rules"].items():
        t = rule["deployment_temperature"]
        t_text = f"{t:.2f}" if isinstance(t, float) else "per-fold"
        print(f"{name:<18}{t_text:>8}{rule['pooled']['winner_log_loss']:>10.4f}"
              + "".join(f"{rule['pooled']['ece'][str(b)]:>10.4f}" for b in BIN_COUNTS))
    print(f"deployed: {record['deployed']['rule']} at T="
          f"{record['deployed']['temperature']:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
