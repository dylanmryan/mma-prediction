"""Apply the SP2.1 pre-registered decision rule and write the decision artifact.

    python scripts/sp2_1_decision.py            # writes models/walkforward/sp2_1_decision.json
    python scripts/sp2_1_decision.py --print    # ... and echoes it

Reads only committed walk-forward reports (paths are the constants below), so
re-running it reproduces the artifact exactly. It decides nothing on its own:
the rule is quoted verbatim from the pre-registration
(docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md, "Attribution,
decided before seeing the numbers") and this script only evaluates which of
its four branches the measured numbers fall into.

Two facts sit OUTSIDE that rule -- A1-XGB's own bar, and the scorer ladder
between XGB and torch. The pre-registration's instruction for that case is
explicit: "if the numbers are ambiguous in a way the rule does not cover, say
so explicitly and stop for a human call rather than inventing a tie-break".
They are therefore quantified in `open_questions` and NOT decided here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.walkforward import bar_check, slice_comparison  # noqa: E402

WF = ROOT / "models" / "walkforward"
OUT = WF / "sp2_1_decision.json"

MIN_BAR = 0.003

# --- the reports the decision reads ----------------------------------------
INCUMBENT = WF / "torch_external_diffsonly_extslice.json"   # A0 / I: the deployed scorer
XGB_INCUMBENT = WF / "xgb_external_diffsonly_extslice.json"  # same table, XGB rung
TORCH_A1 = WF / "torch_a1_combined.json"
XGB_A1 = WF / "xgb_a1_combined.json"
TORCH_C = WF / "torch_C_best.json"
TORCH_T = WF / "torch_T_best.json"
SEARCH_C = WF / "search" / "C" / "summary.json"
SEARCH_T = WF / "search" / "T" / "summary.json"
NOISE_FLOOR_TORCH = WF / "noise_floor.json"
NOISE_FLOOR_XGB = WF / "noise_floor_xgb.json"
BLEND_S0 = WF / "blend_a0_xgb_torch.json"
BLEND_S1 = WF / "blend_a1_xgb_torch.json"

# Fresh-seed re-scores. XGBoost has no seed ensemble, so its analogue of
# torch's fresh seed set is a fresh `random_state` over the subsample /
# colsample draws (scripts/run_walkforward.py --model-seed). Seed 0 is the
# originally committed report in each family; 1-3 are the fresh seeds.
XGB_A1_SEEDS = {0: XGB_A1, 1: WF / "xgb_a1_combined_seed1.json",
                2: WF / "xgb_a1_combined_seed2.json", 3: WF / "xgb_a1_combined_seed3.json"}
XGB_INCUMBENT_SEEDS = {0: XGB_INCUMBENT,
                       1: WF / "xgb_external_diffsonly_extslice_seed1.json",
                       2: WF / "xgb_external_diffsonly_extslice_seed2.json",
                       3: WF / "xgb_external_diffsonly_extslice_seed3.json"}
FRESH_SEEDS = (1, 2, 3)

RULE = (
    "3. Attribution, decided before seeing the numbers:\n"
    "   - If T clears and C does not -> H1 supported; ship S1 with the searched config.\n"
    "   - If both clear and T - C > sigma_seed -> both matter; ship S1 with the searched "
    "config, and report the split.\n"
    "   - If both clear and T - C <= sigma_seed -> H0: the gain is architecture, not "
    "features. Ship the searched config on S0 (the simpler table), and record that the "
    "extra blocks contributed nothing even at higher capacity.\n"
    "   - If neither clears -> the blocks are dead and the architecture is adequate. "
    "Record and revert."
)

CONFIRMATION_RULE = (
    "2. Ships only if the 5-seed result clears the bar vs I (bar_check ... ships: true) "
    "and a fresh-seed re-score with seeds 5-9 also clears it, measured against the "
    "fresh-seed paired incumbent. Both are required; the search is best-of-25 and "
    "therefore carries selection optimism, which the fresh-seed arm is there to strip out."
)


# --- pure helpers -----------------------------------------------------------


def spread(values) -> dict:
    """Mean / sample sd / min / max of a set of pooled log-losses.

    The sd is the sample (ddof=1) standard deviation, matching
    scripts/noise_floor.py, and is nan for a single value rather than 0 --
    one run measures no spread at all.
    """
    values = [float(v) for v in values]
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {"n": len(values), "values": values, "mean": round(float(np.mean(values)), 6),
            "sd": sd, "min": min(values), "max": max(values)}


def fresh_seed_confirmation(candidate_values, incumbent_values, sigma_seed: float) -> dict:
    """Does the mean candidate-minus-incumbent delta across fresh seeds clear the bar?

    The bar keeps the pre-registered ``max(0.003, 2 * sigma_seed)`` form and
    the strict, 6-dp-rounded comparison of ``mma.walkforward.bar_check``, so a
    delta sitting exactly on the bar does not clear it on float noise. Unlike
    bar_check this is a *paired mean over seeds*, not a single pairing: the
    point of the confirmation is that a lucky seed cannot carry the result.
    Both seed sets must be non-empty and the same size -- an unpaired
    comparison would mix a seed effect into the delta.
    """
    if not candidate_values or not incumbent_values:
        raise ValueError("fresh-seed confirmation needs at least one run on each side")
    if len(candidate_values) != len(incumbent_values):
        raise ValueError(
            f"fresh-seed sets must be paired, got {len(candidate_values)} candidate "
            f"run(s) and {len(incumbent_values)} incumbent run(s)"
        )
    if sigma_seed is None or not np.isfinite(sigma_seed):
        raise ValueError("sigma_seed must be a finite number")
    bar = round(max(MIN_BAR, 2.0 * float(sigma_seed)), 6)
    cand, inc = spread(candidate_values), spread(incumbent_values)
    delta = round(cand["mean"] - inc["mean"], 6)
    return {
        "bar": bar, "sigma_seed": float(sigma_seed),
        "candidate": cand, "incumbent": inc,
        "mean_delta": delta, "passes": bool(delta < -bar),
    }


def attribution_branch(c_ships: bool, t_ships: bool, t_minus_c: float, sigma_seed: float) -> dict:
    """Which of the pre-registered rule-3 branches the C/T numbers fall into.

    Mechanical and total: the four branches partition the (c_ships, t_ships)
    square, with the both-clear case split on T - C against sigma_seed. No
    tie-break is invented anywhere -- if the inputs are the measured numbers
    the output is forced.

    ``t_minus_c`` is the log-loss difference T - C, so it is NEGATIVE when the
    treatment is better. The rule's phrase "T - C > sigma_seed" for the
    both-matter branch reads as "T beats C by more than seed noise", which in
    log-loss is ``t_minus_c < -sigma_seed``; the literal sign would make the
    branch fire when the treatment is *worse*, which is not a reading the rule
    can bear. One branch the rule does not name at all is C clearing while T
    does not -- it is reported as uncovered rather than folded into a
    neighbour, per the pre-registration's instruction to stop for a human call
    rather than invent a tie-break.
    """
    if t_ships and not c_ships:
        return {"branch": "T clears and C does not",
                "verdict": "H1 supported", "action": "ship S1 with the searched config"}
    if c_ships and not t_ships:
        # Not a branch the rule names: C clearing alone is the pure-architecture
        # result, which rule 3's third branch reaches only via "both clear".
        return {"branch": "C clears and T does not",
                "verdict": "not covered by the pre-registered rule",
                "action": "stop for a human call"}
    if c_ships and t_ships:
        if t_minus_c < -abs(float(sigma_seed)):
            return {"branch": "both clear and T - C > sigma_seed",
                    "verdict": "features and architecture both matter",
                    "action": "ship S1 with the searched config, and report the split"}
        return {"branch": "both clear and T - C <= sigma_seed",
                "verdict": "H0: the gain is architecture, not features",
                "action": "ship the searched config on S0 (the simpler table)"}
    return {"branch": "neither clears",
            "verdict": "the blocks are dead and the architecture is adequate",
            "action": "record and revert"}


# --- report plumbing --------------------------------------------------------


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def pooled_ll(report: dict) -> float:
    return float(report["pooled"]["winner_log_loss"])


def profile(path: Path) -> dict:
    """Pooled, per-fold and per-slice winner log-loss for one committed report."""
    report = load(path)
    return {
        "report": str(path.relative_to(ROOT)),
        "candidate": report["config"].get("candidate"),
        "drop_columns": report["config"].get("drop_columns"),
        "pooled": report["pooled"],
        "folds": {year: fold["winner_log_loss"] for year, fold in report["folds"].items()},
        "slices": {name: {"n": s["n"], "winner_log_loss": s["winner_log_loss"], "ece": s["ece"]}
                   for name, s in report["slices"].items()},
    }


def arm(path: Path, incumbent: Path, sigma_seed: float) -> dict:
    out = profile(path)
    out["bar_check"] = bar_check(load(path), load(incumbent), sigma_seed)
    out["slice_deltas"] = slice_comparison(load(path), load(incumbent))
    return out


def search_distribution(path: Path) -> dict:
    summary = load(path)
    return {
        "summary": str(path.relative_to(ROOT)),
        "blocks": summary["blocks"], "n_configs": summary["n_configs"],
        "seeds": summary["seeds"], "distribution": summary["distribution"],
        "best_index": summary["best"]["index"], "best_config": summary["best"]["config"],
        "all_pooled": [entry["pooled_winner_log_loss"] for entry in summary["ranked"]],
    }


def build() -> dict:
    sigma_torch = load(NOISE_FLOOR_TORCH)["sigma_seed"]
    xgb_floor = load(NOISE_FLOOR_XGB)

    a1_seed_lls = {seed: pooled_ll(load(p)) for seed, p in XGB_A1_SEEDS.items()}
    inc_seed_lls = {seed: pooled_ll(load(p)) for seed, p in XGB_INCUMBENT_SEEDS.items()}
    # Conservative and stated in advance: the confirmation uses the LARGER of the
    # two recipes' seed sd, so a noisier arm cannot borrow a quieter one's bar.
    sigma_xgb = max(spread(list(a1_seed_lls.values()))["sd"],
                    spread(list(inc_seed_lls.values()))["sd"])

    c, t = arm(TORCH_C, INCUMBENT, sigma_torch), arm(TORCH_T, INCUMBENT, sigma_torch)
    t_minus_c = round(pooled_ll(load(TORCH_T)) - pooled_ll(load(TORCH_C)), 6)
    branch = attribution_branch(c["bar_check"]["ships"], t["bar_check"]["ships"],
                                t_minus_c, sigma_torch)

    fresh_all = fresh_seed_confirmation(list(a1_seed_lls.values()),
                                        list(inc_seed_lls.values()), sigma_xgb)
    fresh_only = fresh_seed_confirmation([a1_seed_lls[s] for s in FRESH_SEEDS],
                                         [inc_seed_lls[s] for s in FRESH_SEEDS], sigma_xgb)

    torch_s0, torch_s1 = profile(INCUMBENT), profile(TORCH_A1)
    xgb_s0, xgb_s1 = profile(XGB_INCUMBENT), profile(XGB_A1)
    blend_s0, blend_s1 = profile(BLEND_S0), profile(BLEND_S1)
    deployed_ll = pooled_ll(load(INCUMBENT))

    return {
        "experiment": "SP2.1 combined blocks x model capacity",
        "pre_registration": "docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md",
        "incumbent": {"report": str(INCUMBENT.relative_to(ROOT)),
                      "pooled_winner_log_loss": deployed_ll,
                      "note": "A0; the deployed scorer (torch 5-seed ensemble on S0)"},
        "noise_floors": {
            "torch": {"sigma_seed": sigma_torch, "bar": max(MIN_BAR, 2 * sigma_torch),
                      "source": str(NOISE_FLOOR_TORCH.relative_to(ROOT))},
            "xgb": {
                "sigma_seed_incumbent_recipe": xgb_floor["sigma_seed"],
                "sigma_seed_a1_recipe": spread(list(a1_seed_lls.values()))["sd"],
                "sigma_seed_used": sigma_xgb,
                "bar": max(MIN_BAR, 2 * sigma_xgb),
                "source": str(NOISE_FLOOR_XGB.relative_to(ROOT)),
                "note": ("first measurement of XGBoost's own seed noise in this project. "
                         "XGB has no seed ensemble; its stochasticity is subsample/"
                         "colsample under one random_state, varied with --model-seed. "
                         "Note it is ~2.5-3.6x the torch ensemble's sigma_seed of "
                         f"{round(sigma_torch, 6)} -- a 5-seed average is a quieter "
                         "measurement than a single fit."),
            },
        },
        "arms": {
            "A0": {**torch_s0, "role": "incumbent, no run needed"},
            "A1_torch": {**arm(TORCH_A1, INCUMBENT, sigma_torch),
                         "role": "S1, deployed config, torch"},
            "A1_xgb": {**arm(XGB_A1, XGB_INCUMBENT, sigma_xgb),
                       "role": "S1, deployed config, XGB (paired against the XGB rung of S0)"},
            "C": {**c, "role": "control: S0, searched"},
            "T": {**t, "role": "treatment: S1, searched"},
            "A2": {"role": "recency on the winner of {C, T}",
                   "run": False,
                   "reason": ("gated in the pre-registration on the winner having cleared "
                              "the bar; neither C nor T cleared, so A2 was not run")},
        },
        "search_distributions": {"C": search_distribution(SEARCH_C),
                                 "T": search_distribution(SEARCH_T)},
        "fresh_seed_confirmation": {
            "rule": CONFIRMATION_RULE,
            "applies_to": ("A1-XGB only: it is the one result that cleared a bar. C and T "
                           "both failed at 5 seeds, and rule 2's confirmation is a "
                           "requirement for shipping, not for recording a failure, so the "
                           "torch fresh-seed re-scores of C and T were deliberately not run."),
            "method": ("XGB has no seed ensemble, so the analogue of torch's fresh seed set "
                       "is a fresh random_state (--model-seed). Seed 0 is the originally "
                       "committed run; 1-3 are fresh. --model-seed 0 was verified to "
                       "reproduce both committed seed-0 reports exactly before use."),
            "a1_xgb_by_seed": a1_seed_lls,
            "incumbent_xgb_by_seed": inc_seed_lls,
            "fresh_seeds_only": {"seeds": list(FRESH_SEEDS), **fresh_only},
            "all_seeds": {"seeds": sorted(a1_seed_lls), **fresh_all},
            "passes": bool(fresh_only["passes"] and fresh_all["passes"]),
            "verdict": (
                "FAILS. The seed-0 pairing that cleared the bar (-0.0039) does not "
                "survive re-seeding: the mean delta is "
                f"{fresh_only['mean_delta']} over the fresh seeds and "
                f"{fresh_all['mean_delta']} over all four, both short of the "
                f"{fresh_only['bar']} bar. A1-XGB's clearance was seed luck."
            ),
        },
        "attribution": {
            "rule": RULE,
            "sigma_seed": sigma_torch,
            "C_pooled": pooled_ll(load(TORCH_C)), "C_ships": c["bar_check"]["ships"],
            "T_pooled": pooled_ll(load(TORCH_T)), "T_ships": t["bar_check"]["ships"],
            "T_minus_C": t_minus_c,
            "T_minus_C_convention": ("log-loss difference T - C; negative means the "
                                     "treatment is better, so the rule's 'T - C > "
                                     "sigma_seed' branch is T - C < -sigma_seed here"),
            "T_minus_C_within_sigma": bool(abs(t_minus_c) <= sigma_torch),
            **branch,
            "decision": (
                "Neither C nor T clears the bar against the deployed incumbent "
                f"(C {c['bar_check']['delta']}, T {t['bar_check']['delta']}, bar "
                f"{c['bar_check']['bar']}), and T - C = {t_minus_c} is inside "
                f"sigma_seed = {round(sigma_torch, 6)}. Under the fourth branch the "
                "restored blocks are dead and the deployed architecture is adequate: "
                "record and revert. Nothing from this experiment ships."
            ),
        },
        "open_questions": [
            {
                "id": "a1_xgb_cleared_its_own_bar",
                "status": "resolved by the fresh-seed confirmation; recorded for the record",
                "question": ("A1-XGB improved -0.0039 against the XGB rung of S0 with all "
                             "four slices better -- does the combined feature set carry "
                             "signal that gradient boosting extracts and the MLP cannot?"),
                "covered_by_rule": False,
                "quantification": {
                    "seed_0_delta": arm(XGB_A1, XGB_INCUMBENT, sigma_xgb)["bar_check"]["delta"],
                    "seed_0_ships": arm(XGB_A1, XGB_INCUMBENT, sigma_xgb)["bar_check"]["ships"],
                    "mean_delta_over_4_seeds": fresh_all["mean_delta"],
                    "bar": fresh_all["bar"],
                    "slice_deltas_seed_0": slice_comparison(load(XGB_A1), load(XGB_INCUMBENT)),
                },
                "answer": ("No, not at the pre-registered bar. Re-seeding halves the "
                           "effect: the mean delta is -0.00275, inside the 0.003 bar. "
                           "The combined table does look worth roughly 0.002-0.003 to "
                           "XGBoost and roughly nothing to torch, which is directionally "
                           "the capacity hypothesis, but it is under the bar and cannot "
                           "ship under this pre-registration."),
            },
            {
                "id": "scorer_ladder_may_have_flipped",
                "status": "OPEN -- outside the pre-registered rule; needs a human call",
                "question": ("SP1 ranked torch above XGB (0.6510 vs 0.6537) and the torch "
                             "ensemble is what is deployed. On S1 the single-seed numbers "
                             "read XGB 0.6467 vs torch 0.6475. Should the deployed scorer "
                             "change, and to what?"),
                "covered_by_rule": False,
                "comparison_table": {
                    "torch_S0": torch_s0, "torch_S1": torch_s1,
                    "xgb_S0": xgb_s0, "xgb_S1": xgb_s1,
                    "blend_S0": blend_s0, "blend_S1": blend_s1,
                },
                "vs_deployed_torch_S0": {
                    "deployed_pooled": deployed_ll,
                    "xgb_S1_seed_0": round(pooled_ll(load(XGB_A1)) - deployed_ll, 6),
                    "xgb_S1_mean_over_4_seeds": round(
                        spread(list(a1_seed_lls.values()))["mean"] - deployed_ll, 6),
                    "torch_S1": round(pooled_ll(load(TORCH_A1)) - deployed_ll, 6),
                    "blend_S0": round(pooled_ll(load(BLEND_S0)) - deployed_ll, 6),
                    "blend_S1": round(pooled_ll(load(BLEND_S1)) - deployed_ll, 6),
                },
                "seed_noise_context": (
                    f"XGB's own sigma_seed is {round(sigma_xgb, 6)} against torch's "
                    f"{round(sigma_torch, 6)}. A single XGB fit is a noisier measurement "
                    "than a 5-seed torch ensemble, so the two 4-dp numbers are not "
                    "comparable at face value: averaged over four seeds XGB+S1 is 0.6480, "
                    "which is 0.0004 WORSE than the deployed torch+S0's 0.6476, not better."
                ),
                "blend_note": (
                    "The equal-weight average of the two scorers is the best number "
                    "measured anywhere in SP2.1: 0.6459 on S0 and 0.6436 on S1, i.e. "
                    "-0.0017 and -0.0040 against the deployed scorer. Both use the "
                    "single seed-0 XGB member, so each carries roughly half of XGB's "
                    "0.00125 seed sd. A blend is NOT one of the five pre-registered "
                    "arms and cannot ship under SP2.1; it is measured here so the human "
                    "call is made with the third option on the table, and it is the "
                    "obvious candidate for a fresh pre-registration."
                ),
                "what_switching_the_scorer_would_require": [
                    "src/mma/inference.py: Ensemble loads models/torch/seed_*.pt through "
                    "MultiTaskNet + Preprocessor, and predict_symmetrized averages the "
                    "corner-swapped pass over that ensemble. Both are torch-specific and "
                    "would need an XGB (or blend) implementation behind the same interface.",
                    "scripts/train_torch.py / scripts/train_xgb.py: the deployed artifact set "
                    "would change; the XGB path currently trains one model per head with no "
                    "seed ensemble, so a switch also decides whether to add one.",
                    "src/mma/versioning.py: MODEL_ARTIFACT_GLOBS hashes the torch weights and "
                    "preprocessor only, so the model hash would stop covering the artifacts "
                    "actually serving predictions until the globs are updated.",
                    "app.py: builds and caches the torch Ensemble and renders its model card "
                    "from the torch walk-forward report.",
                    "scripts/predict_upcoming.py and src/mma/prospective.py: both call "
                    "predict_symmetrized on the torch ensemble.",
                    "scripts/roll_window.py: rolls the deployed torch artifacts forward.",
                    "src/mma/explain.py: already XGB-based, so a switch would finally align "
                    "the explanation path with the scoring path instead of splitting them.",
                    "scripts/refit_decision.py: the refit-through-latest budget was derived "
                    "for the torch recipe and would have to be re-derived.",
                    "tests/test_inference.py, test_serving_parity.py, test_prospective.py, "
                    "test_predict_upcoming.py, test_versioning.py, test_app.py and the "
                    "processed-artifact guards test_processed_torch.py / test_processed_xgb.py "
                    "all assert the torch serving path.",
                ],
                "not_decided_here": (
                    "The pre-registration says: 'if the numbers are ambiguous in a way the "
                    "rule does not cover, say so explicitly and stop for a human call rather "
                    "than inventing a tie-break'. SP2.1's arms were all framed around torch, "
                    "so nothing in it authorises changing the deployed scorer. No artifact "
                    "was retrained and nothing was redeployed."
                ),
            },
        ],
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
    print(f"attribution branch: {decision['attribution']['branch']}")
    print(f"fresh-seed confirmation (A1-XGB): "
          f"{'PASS' if decision['fresh_seed_confirmation']['passes'] else 'FAIL'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
