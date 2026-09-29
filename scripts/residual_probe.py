"""Does a candidate column group add anything on top of the deployed model?

One instrument, reused, instead of a fresh throwaway script per idea. Every
"what if we also knew X?" question this project has asked reduces to the same
measurement, and doing it the same way each time is what makes the answers
comparable.

**The question.** Take the deployed hybrid's own out-of-fold probability, put
its logit in as a FIXED offset, and ask whether the candidate columns move the
answer in a direction the deployed model does not already know about. Pinning
the offset matters: with a free coefficient the fit could earn a "gain" purely
by re-calibrating the deployed model, which is a real effect but not this one,
and it would be credited to the new columns.

**Asked twice.** Once linearly and once with gradient-boosted trees
(XGBoost's ``base_margin`` is a true offset). The deployed scorer is trees
blended with a net, so a linear-only probe under-states what it could extract.

**Against a noise floor, always.** Every gain is reported beside a SHUFFLED
control that destroys the link to the outcome while keeping the columns'
marginal distributions. Two standard deviations of that control is the
smallest effect the procedure can distinguish -- and the script says whether
that floor is under the project's 0.003 shipping bar, because **a null result
from a procedure that cannot resolve the bar says nothing at all**. That line
is the one this repo most needed: an earlier probe reported a clean negative
whose floor was 0.0045, above the bar, and the negative was worth much less
than it looked.

**Exploration window only.** Fold years 2018-2021. 2022-2025 are never touched
here, so they stay clean for a pre-registered confirmatory run through
`scripts/run_walkforward.py` if a group ever survives. This is a SCREEN, not a
decision: nothing here ships anything, and a group that looks promising has to
go through the harness with a bar fixed in advance like everything else.

  OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/residual_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROCESSED = ROOT / "data" / "processed"
OUT_PATH = ROOT / "models" / "residual_probes.json"
#: The deployed scorer's own out-of-fold dump -- the one that carries fight_id.
OOF_PREDICTIONS = ROOT / "models" / "walkforward" / "preds" / "hybrid_e2_cells.json"

EXPLORE_YEARS = (2018, 2019, 2020, 2021)
HELD_BACK = (2022, 2023, 2024, 2025)
PROJECT_BAR = 0.003
MIN_ROWS = 200


# --- the offset tests ---------------------------------------------------------

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_offset_logistic(x, y, off, lam: float = 1.0):
    from scipy.optimize import minimize

    n, k = x.shape

    def nll(theta):
        b, w = theta[0], theta[1:]
        z = off + b + x @ w
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z)) / n
        loss += lam * float(w @ w) / n
        p = 1.0 / (1.0 + np.exp(-z))
        return loss, np.concatenate([[float(np.sum(p - y)) / n],
                                     (x.T @ (p - y)) / n + 2.0 * lam * w / n])

    return minimize(nll, np.zeros(k + 1), jac=True, method="L-BFGS-B",
                    options={"maxiter": 500}).x


def linear_gain(x, y, off, seed: int, folds: int = 5) -> float:
    from sklearn.model_selection import KFold

    base = log_loss(y, 1 / (1 + np.exp(-off)))
    preds = np.empty_like(y, dtype=float)
    for train, test in KFold(n_splits=folds, shuffle=True,
                             random_state=seed).split(x):
        theta = _fit_offset_logistic(x[train], y[train], off[train])
        preds[test] = 1.0 / (1.0 + np.exp(
            -(off[test] + theta[0] + x[test] @ theta[1:])))
    return base - log_loss(y, preds)


def tree_gain(x, y, off, seed: int, folds: int = 5) -> float:
    import xgboost as xgb
    from sklearn.model_selection import KFold

    base = log_loss(y, 1 / (1 + np.exp(-off)))
    preds = np.empty_like(y, dtype=float)
    for train, test in KFold(n_splits=folds, shuffle=True,
                             random_state=seed).split(x):
        booster = xgb.train(
            {"objective": "binary:logistic", "eta": 0.05, "max_depth": 3,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
             "seed": seed, "verbosity": 0},
            xgb.DMatrix(x[train], label=y[train], base_margin=off[train]),
            num_boost_round=60)
        preds[test] = booster.predict(xgb.DMatrix(x[test], base_margin=off[test]))
    return base - log_loss(y, preds)


ESTIMATORS = (("linear", linear_gain, 20), ("trees", tree_gain, 12))


# --- accumulated damage: the one group registered so far ----------------------

class Wear:
    """A fighter's accumulated punishment, pre-fight.

    Every feature the model has about being hit is a RATE --
    `sig_absorbed_pm_diff` is strikes absorbed per minute -- and a rate
    normalises exposure away: two fighters at the same absorbed-per-minute are
    identical to the model whether one has 40 career minutes or 400. These are
    absolute totals, on the hypothesis that damage does not reset.

    One column here is not a rate-versus-total question at all. **Knockdowns
    suffered is absent from the model entirely**: `mma.history` accumulates
    `kd` (knockdowns SCORED) and never the opponent's, so the clearest
    durability signal the scrape records has never been a feature.
    """

    FIELDS = ("kd_suffered", "kd_suffered_pf", "head_absorbed", "sig_absorbed",
              "career_minutes", "head_absorbed_365d", "age_x_head_absorbed")

    def __init__(self) -> None:
        self.fights = 0
        self.seconds = 0.0
        self.kd_suffered = 0.0
        self.head_absorbed = 0.0
        self.sig_absorbed = 0.0
        self.recent: list[tuple[pd.Timestamp, float]] = []

    def snapshot(self, as_of, age) -> dict:
        cutoff = as_of - pd.Timedelta(days=365)
        return {
            "kd_suffered": self.kd_suffered,
            "kd_suffered_pf": (self.kd_suffered / self.fights
                               if self.fights else None),
            "head_absorbed": self.head_absorbed,
            "sig_absorbed": self.sig_absorbed,
            "career_minutes": self.seconds / 60.0,
            "head_absorbed_365d": sum(v for d, v in self.recent if d >= cutoff),
            "age_x_head_absorbed": (self.head_absorbed * age
                                    if age is not None and not np.isnan(age)
                                    else None),
        }

    def fold(self, opp, seconds, date) -> None:
        self.fights += 1
        self.seconds += float(seconds or 0.0)
        head = float(opp.get("head_landed", 0) or 0)
        self.kd_suffered += float(opp.get("kd", 0) or 0)
        self.head_absorbed += head
        self.sig_absorbed += float(opp.get("sig_landed", 0) or 0)
        self.recent.append((date, head))


def build_wear(fights, fight_stats, fighters) -> pd.DataFrame:
    """One row per (fight, fighter) of PRE-fight wear.

    Single chronological pass, emit then fold -- the same contract as
    `mma.history`, so a fight never sees itself or anything after it.
    """
    stats = fight_stats.set_index(["fight_id", "fighter_id"])
    dob = fighters.set_index("fighter_id")["dob"] if "dob" in fighters else None
    states: dict[str, Wear] = {}
    rows = []
    cols = ["fight_id", "date", "duration_sec", "fighter_a_id", "fighter_b_id"]
    for fid, date, dur, a_id, b_id in fights.sort_values(
            ["date", "fight_id"])[cols].itertuples(index=False):
        date = pd.Timestamp(date)
        for own in (a_id, b_id):
            age = None
            if dob is not None and own in dob.index and pd.notna(dob.loc[own]):
                age = (date - pd.Timestamp(dob.loc[own])).days / 365.25
            rows.append({"fight_id": fid, "fighter_id": own,
                         **states.setdefault(own, Wear()).snapshot(date, age)})
        for own, opp in ((a_id, b_id), (b_id, a_id)):
            try:
                opp_row = stats.loc[(fid, opp)]
            except KeyError:
                continue
            if isinstance(opp_row, pd.DataFrame):
                continue
            states[own].fold(opp_row, dur, date)
    return pd.DataFrame(rows)


#: name -> (builder, field tuple, sub-groups reported separately)
GROUPS = {
    "accumulated_damage": (
        build_wear, Wear.FIELDS,
        {
            "knockdowns suffered (absent from the model entirely)":
                ("kd_suffered", "kd_suffered_pf"),
            "cumulative absorbed punishment":
                ("head_absorbed", "sig_absorbed", "career_minutes"),
            "recent wear and the age interaction":
                ("head_absorbed_365d", "age_x_head_absorbed"),
            "every damage column at once": Wear.FIELDS,
        },
    ),
}


# --- the market: what is our odds-free signal worth on top of the line? -------

MARKET_TABLE = ROOT / "data" / "external" / "market_odds.parquet"


def load_market_frame() -> pd.DataFrame:
    """The deployed model's out-of-fold probability beside the devigged line.

    Joined on `fight_id` and validated one-to-one -- never by row position.
    Both columns are already in the feature table's corner frame: the dump
    because the harness wrote it there, the market because
    `build_odds_benchmark.build_market_table` oriented it through the same
    `swapped` flag.

    Coverage is 68% of walk-forward rows and thins sharply in recent years
    (86% in 2018, 21% in 2025), so the exploration window this probe uses is
    also where the odds are densest -- about 1,830 fights.
    """
    dump = json.loads(OOF_PREDICTIONS.read_text())
    oof = pd.DataFrame({
        "fight_id": [str(v) for v in dump["fight_id"]],
        "fold_year": np.asarray(dump["fold_year"], dtype=int),
        "y": np.asarray(dump["y_winner"], dtype=float),
        "p": np.asarray(dump["p_winner"], dtype=float),
    })
    market = pd.read_parquet(MARKET_TABLE)[["fight_id", "market_implied_a"]]
    data = oof.merge(market, on="fight_id", how="inner", validate="one_to_one")
    data["market_p"] = data["market_implied_a"]
    data["model_logit"] = logit(data["p"])
    data["market_logit"] = logit(data["market_p"])
    return data


#: Groups whose columns are already per-fight, so there is no per-fighter
#: builder and no `orient` step. A subgroup here names the PINNED side as well
#: as the candidate columns, because the two directions below are different
#: questions and a gain means the opposite thing in each.
#: A subgroup is (pinned side, candidate columns, role). `role` matters for
#: the artifact's `any_signal`, which exists to say "something turned up that
#: needs a pre-registered harness run". Only a SCREEN can do that. The second
#: direction below is a DIAGNOSTIC: that the closing line adds on top of us is
#: already known from the benchmark, it is guaranteed positive, and the
#: actionable version of it -- odds as a model feature -- is a separate
#: experiment needing its own pre-registration rather than a louder screen.
FIGHT_LEVEL_GROUPS = {
    "market": (
        load_market_frame,
        {
            "our signal on top of the closing line":
                ("market_p", ("model_logit",), "screen"),
            "the closing line on top of our signal":
                ("p", ("market_logit",), "diagnostic"),
        },
    ),
}


def run_fight_level(group_name: str, rng) -> dict:
    loader, subgroups = FIGHT_LEVEL_GROUPS[group_name]
    data = loader()
    explore = data[data["fold_year"].isin(EXPLORE_YEARS)].copy()
    results, roles = {}, {}
    for label, (offset, columns, role) in subgroups.items():
        results[label] = dict(assess_subgroup(explore, list(columns), rng,
                                              offset=offset), role=role)
        roles[label] = role

    def showed(role: str) -> bool:
        return any(results[label].get(est, {}).get("signal", False)
                   for label, r in roles.items() if r == role
                   for est, _fn, _d in ESTIMATORS)

    return {
        "n_exploration_rows": int(len(explore)),
        "n_held_back_and_untouched": int(
            (~data["fold_year"].isin(EXPLORE_YEARS)).sum()),
        "subgroups": results,
        # Screens only: a diagnostic showing signal is the expected reading,
        # not a prompt to run the harness.
        "any_signal": showed("screen"),
        "diagnostic_signal": showed("diagnostic"),
    }


# --- running a group ----------------------------------------------------------

def orient(per_fighter: pd.DataFrame, fights: pd.DataFrame,
           features: pd.DataFrame, fields) -> pd.DataFrame:
    """Per-fighter rows -> A-minus-B differentials in the FEATURE TABLE's frame.

    Oriented through the same `swapped` flag the feature table's own labels
    use, and joined on ids rather than by position -- this repo has mis-paired
    prediction dumps positionally twice.
    """
    corners = fights[["fight_id", "fighter_a_id", "fighter_b_id"]].merge(
        features[["fight_id", "swapped"]], on="fight_id", how="inner")
    a = np.where(corners["swapped"], corners["fighter_b_id"],
                 corners["fighter_a_id"])
    b = np.where(corners["swapped"], corners["fighter_a_id"],
                 corners["fighter_b_id"])
    idx = per_fighter.set_index(["fight_id", "fighter_id"])
    side_a = idx.reindex(list(zip(corners["fight_id"], a))).reset_index(drop=True)
    side_b = idx.reindex(list(zip(corners["fight_id"], b))).reset_index(drop=True)
    out = pd.DataFrame({"fight_id": corners["fight_id"].values})
    for field in fields:
        out[field + "_diff"] = side_a[field].values - side_b[field].values
    return out


def assess_subgroup(frame: pd.DataFrame, columns, rng, offset: str = "p") -> dict:
    """Do `columns` add on top of the probability in `offset`?

    `offset` names the column whose logit is PINNED. It defaults to "p", the
    deployed model's own out-of-fold probability, which is the question every
    column group asks. The market probe passes the devigged line instead and
    puts the model's logit in `columns`, which inverts the question into "what
    is our odds-free signal worth to someone who already has the line?" -- so
    the pinned side is reported, because a gain means opposite things depending
    on which side it was measured against.
    """
    needed = list(columns) + [offset]
    usable = frame.dropna(subset=needed)
    if len(usable) < MIN_ROWS:
        return {"n": len(usable), "skipped": "too few usable rows"}
    y = usable["y"].to_numpy(dtype=float)
    off = logit(usable[offset].to_numpy(dtype=float))
    x = usable[list(columns)].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)

    out = {"n": int(len(usable)), "n_columns": len(columns),
           "offset": offset,
           "baseline_log_loss": round(log_loss(y, usable[offset]), 4),
           "columns": list(columns)}
    for label, fn, draws in ESTIMATORS:
        real = float(np.mean([fn(x, y, off, seed=s) for s in range(5)]))
        null = [fn(x[rng.permutation(len(x))], y, off, seed=s % 5)
                for s in range(draws)]
        mean, sd = float(np.mean(null)), float(np.std(null))
        floor = 2 * sd
        out[label] = {
            "gain": round(real, 5),
            "shuffled_null_mean": round(mean, 5),
            "shuffled_null_sd": round(sd, 5),
            "sd_from_null": round((real - mean) / (sd + 1e-12), 2),
            "detection_floor": round(floor, 5),
            "floor_resolves_the_bar": bool(floor <= PROJECT_BAR),
            "signal": bool(real - mean > floor),
        }
    return out


def run(group_name: str, rng) -> dict:
    builder, fields, subgroups = GROUPS[group_name]
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    fight_stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    features = pd.read_parquet(PROCESSED / "features.parquet")
    dump = json.loads(OOF_PREDICTIONS.read_text())

    diffs = orient(builder(fights, fight_stats, fighters), fights, features, fields)
    oof = pd.DataFrame({
        "fight_id": [str(v) for v in dump["fight_id"]],
        "fold_year": np.asarray(dump["fold_year"], dtype=int),
        "y": np.asarray(dump["y_winner"], dtype=float),
        "p": np.asarray(dump["p_winner"], dtype=float),
    })
    data = oof.merge(diffs, on="fight_id", how="inner")
    explore = data[data["fold_year"].isin(EXPLORE_YEARS)].copy()
    results = {
        name: assess_subgroup(explore, [c + "_diff" for c in cols], rng)
        for name, cols in subgroups.items()
    }
    return {
        "n_exploration_rows": int(len(explore)),
        "n_held_back_and_untouched": int(
            len(data[data["fold_year"].isin(HELD_BACK)])),
        "subgroups": results,
        "any_signal": any(
            v.get(label, {}).get("signal") for v in results.values()
            for label, _, _ in ESTIMATORS),
    }


def render(name: str, block: dict) -> None:
    print(f"\n{'=' * 74}\n{name}  --  {block['n_exploration_rows']:,} "
          f"exploration rows ({block['n_held_back_and_untouched']:,} held back)")
    for label, sub in block["subgroups"].items():
        if sub.get("skipped"):
            print(f"\n  {label}: {sub['skipped']} (n={sub['n']})")
            continue
        print(f"\n  {label}")
        print(f"    n={sub['n']:,}  cols={sub['n_columns']}  "
              f"baseline {sub['baseline_log_loss']}")
        for est, _, _ in ESTIMATORS:
            r = sub[est]
            print(f"      {est:7s} gain {r['gain']:+.5f}  null "
                  f"{r['shuffled_null_mean']:+.5f}  -> {r['sd_from_null']:+.1f} sd;"
                  f"  floor {r['detection_floor']:.5f} "
                  f"({'resolves' if r['floor_resolves_the_bar'] else 'CANNOT resolve'}"
                  f" the {PROJECT_BAR} bar)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", action="append",
                        choices=sorted(set(GROUPS) | set(FIGHT_LEVEL_GROUPS)))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    names = args.group or sorted(set(GROUPS) | set(FIGHT_LEVEL_GROUPS))
    report = {
        "experiment": "residual probes -- does a column group add on top of "
                      "the deployed model?",
        "generated_by": "scripts/residual_probe.py",
        "offset": str(OOF_PREDICTIONS.relative_to(ROOT)),
        "offset_note": ("the default pinned side. The `market` group pins the "
                        "devigged closing line instead and puts the model's "
                        "logit in the candidate columns, so each subgroup "
                        "records which side it pinned."),
        "exploration_years": list(EXPLORE_YEARS),
        "held_back_years": list(HELD_BACK),
        "bar": PROJECT_BAR,
        "note": ("a SCREEN, not a decision. A group that showed signal would "
                 "still have to clear the bar through run_walkforward.py with "
                 "the bar fixed in advance."),
        "groups": {},
    }
    for name in names:
        block = run_fight_level(name, rng) if name in FIGHT_LEVEL_GROUPS \
            else run(name, rng)
        report["groups"][name] = block
        render(name, block)

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
