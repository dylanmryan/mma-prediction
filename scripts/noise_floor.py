"""Measure the seed noise floor of the incumbent torch configuration.

sigma_seed = sample standard deviation of pooled walk-forward winner
log-loss across runs that differ only in their seed set. The shipping bar
is max(0.003, 2 * sigma_seed). Usage:
    python scripts/noise_floor.py models/walkforward/torch_v1.json models/walkforward/torch_v1_seeds5.json models/walkforward/torch_v1_seeds10.json
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


def sigma_from_reports(reports: list[dict]) -> dict:
    seed_sets = [r.get("config", {}).get("seeds") for r in reports]
    check_disjoint_seed_sets(seed_sets)
    values = [r["pooled"]["winner_log_loss"] for r in reports]
    sigma = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {"n_reports": len(values), "pooled_winner_log_loss": values,
            "sigma_seed": sigma, "bar": max(MIN_BAR, 2.0 * sigma)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    result = sigma_from_reports(reports)
    result["reports"] = [str(p) for p in args.reports]
    result["seed_sets"] = [r.get("config", {}).get("seeds") for r in reports]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
