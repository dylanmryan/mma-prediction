"""What should the shipping bar actually be?

Every decision in this project is taken against `max(0.003, 2 sigma_seed)`,
and for the deployed blend sigma_seed is **0.0001** -- so the operative bar is
thirty times the only noise anyone ever measured. That is not obviously wrong,
but it has never been checked, and it is expensive in both directions: a bar
set too high rejects real gains (`recency` at -0.0011, `opponent_adjusted` on
torch, the capacity-search winner at -0.0024), and one set too low ships noise.

**The gap in the evidence.** `noise_floor.py` measures how far a refit moves
on the SAME fights. A shipping decision asks something else: will this delta
hold on the NEXT four thousand fights? Those are different questions with
different answers, and only the first has ever been measured here.

This instrument measures the second, and combines the two:

* **Sampling error** -- a paired, per-fight log-loss difference, bootstrapped.
  Paired because the two candidates are scored on identical fights, so the
  enormous fight-to-fight variance in log-loss cancels and only the difference
  is resampled.
* **Clustered by fold year**, because the harness fits one model per fold and
  the fights inside a fold share it; treating 4,804 fights as 4,804
  independent draws would understate the error. Whole fold-years are resampled
  rather than fights-within-years: with eight clusters the between-fold
  variance is what dominates, and the one-stage form is the conservative one.
* **Plus seed noise, not instead of it.** The bootstrap holds the two fits
  fixed and varies the fights; `noise_floor.py` holds the fights fixed and
  varies the fits. They are orthogonal, so the honest standard error is
  `sqrt(se_sampling^2 + sigma_seed^2)` and a bar built on either alone is
  understated.

**Alignment is checked, never assumed.** These dumps predate `fight_id` and
carry only `fold_year` and `y_winner`, so they can be paired only positionally
-- which is the exact bug class that silently broke the market benchmark in
September. `paired_delta` refuses any pair whose outcome or fold-year columns
disagree element-wise.

**This script decides nothing.** It reports a standard error and the bar that
follows from it. Changing a bar is a pre-registered act, and the new bar has
to be committed before the next experiment is scored, not after.

  OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/bar_audit.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PREDS = ROOT / "models" / "walkforward" / "preds"
OUT_PATH = ROOT / "models" / "bar_audit.json"

#: The seed noise this project already measured, read from the artifact that
#: measured it rather than retyped -- if `noise_floor.py` is ever re-run on a
#: refreshed table, this moves with it.
NOISE_FLOOR = ROOT / "models" / "walkforward" / "noise_floor_blend.json"
SIGMA_SEED = float(json.loads(NOISE_FLOOR.read_text())["sigma_seed"])

#: The bar every committed decision in this repo was taken against.
PROJECT_BAR = 0.003

N_BOOT = 10_000
SEED = 0
_EPS = 1e-15


@dataclass(frozen=True)
class Dump:
    """One candidate's out-of-fold predictions, as `run_walkforward --dump-predictions`
    writes them."""

    name: str
    y: np.ndarray
    p: np.ndarray
    fold_year: np.ndarray


def load_dump(path: Path) -> Dump:
    raw = json.loads(Path(path).read_text())
    return Dump(
        name=str(raw.get("name", Path(path).stem)),
        y=np.asarray(raw["y_winner"], dtype=float),
        p=np.asarray(raw["p_winner"], dtype=float),
        fold_year=np.asarray(raw["fold_year"], dtype=int),
    )


def per_fight_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The pooled log-loss, left undivided. Its mean is what the reports quote."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def paired_delta(candidate: Dump, incumbent: Dump) -> np.ndarray:
    """Per-fight `candidate - incumbent` log-loss, so NEGATIVE is an improvement.

    That is the project's own sign convention: SP2.2's blend is recorded at
    -0.0039, `external` at -0.0034.
    """
    if len(candidate.y) != len(incumbent.y):
        raise ValueError(
            f"cannot pair dumps of different length: {candidate.name} has "
            f"{len(candidate.y)} rows, {incumbent.name} has {len(incumbent.y)}. "
            "They were scored on different tables."
        )
    if not np.array_equal(candidate.y, incumbent.y):
        raise ValueError(
            f"{candidate.name} and {incumbent.name} disagree on the outcome "
            "column, so they are not row-aligned and cannot be paired "
            "positionally"
        )
    if not np.array_equal(candidate.fold_year, incumbent.fold_year):
        raise ValueError(
            f"{candidate.name} and {incumbent.name} disagree on fold_year, so "
            "they are not row-aligned and cannot be paired positionally"
        )
    return per_fight_log_loss(candidate.y, candidate.p) - per_fight_log_loss(
        incumbent.y, incumbent.p
    )


def bootstrap(
    delta: np.ndarray,
    fold_year: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    cluster: bool = True,
) -> dict:
    """Resample the paired delta; return its mean, standard error and 95% CI.

    `cluster=True` resamples whole fold years (the reported form); `False`
    resamples fights independently and is carried only to show how much the
    clustering costs.
    """
    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    if cluster:
        years = np.unique(fold_year)
        sums = np.array([delta[fold_year == y].sum() for y in years])
        counts = np.array([int((fold_year == y).sum()) for y in years])
        drawn = rng.integers(0, len(years), size=(n_boot, len(years)))
        means = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    else:
        idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
        means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    observed = float(delta.mean())
    # How often the resampled effect lands on the other side of zero. For an
    # effect near the boundary this is the reading that carries the most, and
    # it is not recoverable from the standard error alone.
    wrong_sign = float(np.mean(means >= 0.0) if observed < 0 else np.mean(means <= 0.0))
    return {
        "mean": observed,
        "se": float(means.std(ddof=1)),
        "p_wrong_sign": wrong_sign,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(len(delta)),
        "n_clusters": int(len(np.unique(fold_year))),
        "clustered": bool(cluster),
    }


def combined_se(se_sampling: float, sigma_seed: float = SIGMA_SEED) -> float:
    """Sampling error and seed error are orthogonal, so they add in quadrature."""
    return float(math.sqrt(float(se_sampling) ** 2 + float(sigma_seed) ** 2))


def implied_bars(se: float, n_arms: int = 1) -> dict:
    """The bar that follows from a standard error.

    `single_arm` is the project's own `2 sigma` form applied to the honest
    sigma. `after_search` adds the `sigma sqrt(2 ln N)` that a best-of-N
    produces from nothing -- the correction this project already applies by
    hand when judging a search, kept here so a bar and its search are quoted
    together. At `n_arms=1` the correction is zero and the two agree.
    """
    se = float(se)
    inflation = se * math.sqrt(2.0 * math.log(n_arms)) if n_arms > 1 else 0.0
    return {
        "single_arm": 2.0 * se,
        "after_search": 2.0 * se + inflation,
        "n_arms": int(n_arms),
    }


@dataclass(frozen=True)
class Pair:
    """One comparison to audit, and what it is evidence about."""

    pair: str
    candidate: str
    incumbent: str
    kind: str  # "decision" | "null" | "sanity"
    note: str


#: The reference comparison. Its standard error is the one quoted as the
#: harness's detection floor, because it is a real winner-head decision on the
#: real table -- not a seed swap (too small) and not the isotonic diagnostic
#: (far too large). A floor quoted without naming its pair would be meaningless.
REFERENCE_PAIR = Pair(
    pair="blend_b1 vs torch_external_diffsonly_extslice",
    candidate="blend_b1",
    incumbent="torch_external_diffsonly_extslice",
    kind="decision",
    note="SP2.2's shipped two-family blend against the torch-alone incumbent "
         "it replaced; recorded in the experiment log at -0.0039",
)

PAIRS = (
    REFERENCE_PAIR,
    Pair(
        pair="blend_b1 vs blend_b1_cells_seeds5",
        candidate="blend_b1",
        incumbent="blend_b1_cells_seeds5",
        kind="null",
        note="one recipe at two seed sets: a true effect of zero, so the bar "
             "derived here must cover it",
    ),
    Pair(
        pair="hybrid_e2 vs hybrid_e2_seeds5",
        candidate="hybrid_e2",
        incumbent="hybrid_e2_seeds5",
        kind="null",
        note="the same null on the deployed hybrid's winner marginal",
    ),
    Pair(
        pair="blend_b1_isotonic vs blend_b1",
        candidate="blend_b1_isotonic",
        incumbent="blend_b1",
        kind="sanity",
        note="isotonic recalibration, 0.7049 against 0.6437: an effect nobody "
             "disputes, so the instrument has to resolve it",
    ),
)

ARTIFACT = OUT_PATH


def audit_pair(spec: Pair, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    candidate = load_dump(PREDS / f"{spec.candidate}.json")
    incumbent = load_dump(PREDS / f"{spec.incumbent}.json")
    delta = paired_delta(candidate, incumbent)
    clustered = bootstrap(delta, candidate.fold_year, n_boot, seed, cluster=True)
    iid = bootstrap(delta, candidate.fold_year, n_boot, seed, cluster=False)
    se = combined_se(clustered["se"])
    bars = implied_bars(se)
    years = np.unique(candidate.fold_year)
    # String keys, because this dict is written to JSON and read back by the
    # reproducibility test: int keys would not survive the round trip.
    per_fold = {
        str(int(y)): round(float(delta[candidate.fold_year == y].mean()), 5)
        for y in years
    }
    return {
        "pair": spec.pair,
        "kind": spec.kind,
        "note": spec.note,
        "n": clustered["n"],
        "n_folds": clustered["n_clusters"],
        "delta_log_loss": round(clustered["mean"], 5),
        "se_sampling_clustered": round(clustered["se"], 5),
        "se_sampling_iid": round(iid["se"], 5),
        "sigma_seed": round(SIGMA_SEED, 5),
        "se_combined": round(se, 5),
        "ci_lo": round(clustered["ci_lo"], 5),
        "ci_hi": round(clustered["ci_hi"], 5),
        "bootstrap_p_one_sided": round(float(clustered["p_wrong_sign"]), 4),
        "bar_single_arm": round(bars["single_arm"], 5),
        "clears_its_own_bar": bool(abs(clustered["mean"]) > bars["single_arm"]),
        "clears_the_project_bar": bool(abs(clustered["mean"]) > PROJECT_BAR),
        "folds_favouring_the_candidate": int(sum(v < 0 for v in per_fold.values())),
        "delta_by_fold_year": per_fold,
    }


def audit(pairs=PAIRS, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    results = [audit_pair(p, n_boot, seed) for p in pairs]
    reference = [r for r in results if r["pair"] == REFERENCE_PAIR.pair]
    floor = {
        "measured_on": REFERENCE_PAIR.pair,
        "value": reference[0]["bar_single_arm"] if reference else None,
        "means": "the smallest winner-head log-loss difference this harness "
                 "can distinguish from noise at two standard errors, on "
                 "4,804 pooled fights across 8 folds",
    }
    return {
        "experiment": "what the shipping bar should be, measured rather than assumed",
        "generated_by": "scripts/bar_audit.py",
        "note": "This decides nothing. It reports a standard error and the bar "
                "that follows from it; adopting a bar is a pre-registered act "
                "and has to be committed before the next experiment is scored.",
        "project_bar_in_force": PROJECT_BAR,
        "sigma_seed": SIGMA_SEED,
        "sigma_seed_source": str(NOISE_FLOOR.relative_to(ROOT)),
        "n_boot": n_boot,
        "seed": seed,
        "harness_detection_floor": floor,
        "pairs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    report = audit(PAIRS, args.n_boot, args.seed)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    floor = report["harness_detection_floor"]
    print(f"sigma_seed {SIGMA_SEED:.5f} | project bar {PROJECT_BAR}")
    print(f"detection floor {floor['value']:.5f} on {floor['measured_on']}\n")
    for row in report["pairs"]:
        print(f"  {row['pair']:52s} {row['delta_log_loss']:+.5f} "
              f"se {row['se_combined']:.5f} bar {row['bar_single_arm']:.5f} "
              f"{'CLEARS' if row['clears_its_own_bar'] else 'inside noise'}")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
