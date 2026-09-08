"""Measure the seed noise floor of a walk-forward recipe.

sigma_seed = sample standard deviation of a pooled walk-forward metric across
runs that differ only in their seed set. For the default metric, winner
log-loss, the shipping bar is max(0.003, 2 * sigma_seed):
    python scripts/noise_floor.py models/walkforward/torch_v1.json models/walkforward/torch_v1_seeds5.json models/walkforward/torch_v1_seeds10.json

`--metric ece` measures the same spread for pooled expected calibration error
instead, and `--arm` measures several recipes into one file. SP2.2 needs this
because its pre-registration gates on ECE ("a shipping candidate must not have
a worse pooled ECE than the incumbent") and that gate compares two single
numbers whose precision had never been measured -- so it could not be applied
honestly rather than mechanically against noise:
    python scripts/noise_floor.py --metric ece \
        --arm incumbent a.json b.json c.json --arm B1 d.json e.json f.json \
        --out models/walkforward/noise_floor_ece.json

There is deliberately NO `bar` in the ece output. The bar is a log-loss
shipping rule from the spec; an ECE spread is a precision measurement and
dressing it up as a threshold would invent a rule no pre-registration states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "walkforward" / "noise_floor.json"
MIN_BAR = 0.003


def check_disjoint_seed_sets(seed_sets: list[str | None]) -> None:
    """Raise SystemExit if any two comma-separated seed strings share a seed
    (e.g. "0,1,2,3,4" and "3,4,5" overlap on 3 and 4). ``None`` entries
    (reports with no seeds config, e.g. non-torch candidates) are skipped
    with a warning rather than failing -- there is nothing to compare."""
    parsed = []
    for seeds in seed_sets:
        if seeds is None:
            print(f"noise_floor: warning: report has no seeds config (config.seeds is null), skipping overlap check")
            continue
        parsed.append((seeds, {int(s) for s in seeds.split(",")}))
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            seeds_i, set_i = parsed[i]
            seeds_j, set_j = parsed[j]
            overlap = set_i & set_j
            if overlap:
                raise SystemExit(
                    "noise_floor: reports must use disjoint seed sets, "
                    f"but {seeds_i!r} and {seeds_j!r} share {sorted(overlap)}"
                )


def seed_label(report: dict) -> str | None:
    """The seed set a report was produced under, as a comparable string.

    The torch candidate records its ensemble in ``config.seeds``. The xgb
    candidate has no ensemble -- its stochasticity is one ``random_state``
    over subsample/colsample -- so its seed is ``config.model_seed`` (the
    ``--model-seed`` flag), or, when a run set it through ``--config-json``,
    ``config.config.random_state``. An xgb report with neither ran at the
    ``mma.models.xgb.BASE_PARAMS`` default of 0, which is a real seed and
    must take part in the overlap check rather than being waved through.
    Anything else (the elo floor, an unrecognised candidate) has no seed.
    """
    config = report.get("config", {})
    seeds = config.get("seeds")
    if seeds is not None:
        return str(seeds)
    if config.get("candidate") != "xgb":
        return None
    explicit = (config.get("config") or {}).get("random_state")
    if explicit is not None:
        return str(int(explicit))
    model_seed = config.get("model_seed")
    return str(int(model_seed)) if model_seed is not None else "0"


def sigma_from_reports(reports: list[dict], metric: str = "winner_log_loss") -> dict:
    """Spread of one pooled metric across seed sets.

    The winner_log_loss result keeps its exact historical shape -- key
    ``pooled_winner_log_loss`` and a ``bar`` -- because three committed floors
    have it. Any other metric gets ``pooled_<metric>``, a ``mean`` and
    ``two_sigma``, and no ``bar``: the max(0.003, 2 sigma) rule is the spec's
    log-loss shipping bar and nothing authorises applying it to another metric.
    """
    seed_sets = [seed_label(r) for r in reports]
    check_disjoint_seed_sets(seed_sets)
    values = [r["pooled"][metric] for r in reports]
    sigma = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    out = {"n_reports": len(values), f"pooled_{metric}": values, "sigma_seed": sigma}
    if metric == "winner_log_loss":
        out["bar"] = max(MIN_BAR, 2.0 * sigma)
    else:
        out["mean"] = float(np.mean(values))
        out["two_sigma"] = 2.0 * sigma
    return out


def arm_result(paths: list[Path], metric: str) -> dict:
    reports = [json.loads(p.read_text()) for p in paths]
    result = sigma_from_reports(reports, metric)
    result["reports"] = [str(p) for p in paths]
    result["seed_sets"] = [seed_label(r) for r in reports]
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="*", type=Path,
                        help="reports for a single unnamed recipe (mutually exclusive with --arm)")
    parser.add_argument("--metric", default="winner_log_loss", choices=["winner_log_loss", "ece"],
                        help="pooled metric whose seed spread is measured (default winner_log_loss)")
    parser.add_argument("--arm", nargs="+", action="append", metavar=("LABEL", "REPORT"),
                        default=None,
                        help="a named recipe: LABEL followed by its reports; repeatable")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if bool(args.reports) == bool(args.arm):
        raise SystemExit("pass either positional reports or one or more --arm groups, not both")
    if args.arm:
        arms = {}
        for entry in args.arm:
            label, *paths = entry
            if not paths:
                raise SystemExit(f"--arm {label} names no reports")
            if label in arms:
                raise SystemExit(f"--arm label {label!r} is used twice")
            arms[label] = arm_result([Path(p) for p in paths], args.metric)
        result = {"metric": args.metric, "arms": arms}
    else:
        # Historical single-recipe shape, unchanged: three committed floors have
        # it and re-running this script must still reproduce them byte for byte.
        result = arm_result(args.reports, args.metric)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
