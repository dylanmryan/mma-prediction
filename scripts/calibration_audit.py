"""Is the deployed model's confidence honest, and is there a prize in fixing it?

Two questions, in that order, and the second one is the point.

**Is it miscalibrated?** Fold the out-of-fold predictions onto a single axis --
the probability assigned to whichever corner the model favoured, against
whether that corner actually won -- and test each confidence band against the
realised rate with an exact binomial test. Testing seven bands is seven looks,
so the threshold is Bonferroni-corrected and fixed by the number of bands, not
chosen after seeing which one looks worst.

**And if it is, how much is fixing it worth?** Asked two ways, because one is
not enough.

*The oracle bound.* For every band, recalibrate it using the realised rate
ITSELF -- the answer, which no honest model can have -- and re-score. That
bounds repairing the thing the reliability table diagnoses: a band whose
stated confidence does not match what happened. It does NOT bound every
calibrator, because collapsing a band to one number discards the model's
ordering inside it, and a smooth monotone remedy keeps that ordering while
moving the level. A fitted temperature can and does beat it on synthetic data
(`tests/test_calibration_audit.py`), which is why the second question exists.

*The direct trial.* Fit temperature, Platt and isotonic on four fifths of the
out-of-fold rows, score the fifth they have not seen, and compare against the
deployed probabilities -- which already carry a fitted temperature.

They agree, from different directions. The reliability table alone would have
justified building something: the extreme band [0.80,1.00] promises 0.843 and
delivers 0.785, which reads as an overconfidence to go and fix. But it is one
of seven bands at p=0.010 against a Bonferroni threshold of 0.0071, its
per-year gap changes sign, repairing it with the answers is worth 0.0004
against a bar of 0.003, and no fitted calibrator improves on what is deployed
(temperature -0.00001, Platt -0.00021, isotonic -0.00413). The band is 6% of
the rows and log-loss is not sensitive enough there for any remedy to clear.

Two things worth keeping from the negative. Repairing EVERY band at once with
hindsight makes the model worse (-0.00070): flattening a band buys calibration
by spending discrimination, which is exactly why SP2.2's isotonic remediation
scored 0.7049 against the temperature form's 0.6437
(`models/walkforward/sp2_2_decision.json` `remediation_isotonic`). And the
deployed temperature of 0.85 is confirmed to be the right one -- refitting it
out of sample moves nothing.

Why this is in the repo rather than a scratch file: it closes a question that
otherwise stays open and keeps looking attractive every time someone reads a
reliability table.

  OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/calibration_audit.py
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

MODELS = ROOT / "models"
WALKFORWARD = MODELS / "walkforward"
OUT_PATH = MODELS / "calibration_audit.json"

#: The deployed scorer's own out-of-fold dump -- the same one
#: `scripts/build_odds_benchmark.py` and `scripts/market_edge_analysis.py`
#: read, and the only one that carries `fight_id`. Fold year Y's
#: probabilities come from a model fitted on fights before Y-1, so nothing
#: scored here is in that model's training set.
OOF_PREDICTIONS = WALKFORWARD / "preds" / "hybrid_e2_cells.json"

#: Confidence bands over the FAVOURED side's probability. Fixed here, in the
#: source, so the count that sets the Bonferroni threshold is a property of
#: the audit rather than of its results. The top band runs to 1.0 because the
#: model almost never goes past 0.9 -- splitting it further would test bands
#: of twenty fights.
BAND_EDGES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0)

#: The v3 spec section 5 shipping bar: a change must beat the incumbent's
#: pooled winner log-loss by more than max(0.003, 2 sigma_seed). Nothing here
#: proposes a change, but the bar is what the oracle bound is measured against.
PROJECT_BAR = 0.003

#: Below this many fights a band's realised rate is too noisy to test.
MIN_BAND_N = 20


def load_predictions(path: Path = OOF_PREDICTIONS) -> dict:
    dump = json.loads(Path(path).read_text())
    missing = [k for k in ("p_winner", "y_winner", "fold_year") if k not in dump]
    if missing:
        raise SystemExit(f"{path} is missing {missing}; re-run "
                         "scripts/run_walkforward.py --dump-predictions")
    return dump


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def favoured_view(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold both corners onto one axis: the probability the model put on the
    side it favoured, and whether that side won.

    Without this, a symmetric model looks calibrated by construction -- every
    over-confident prediction at 0.85 is paired with its mirror at 0.15, and
    averaging the two hides the error. Reliability has to be read from the
    favourite's point of view or it is not read at all.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.where(p >= 0.5, p, 1 - p), np.where(p >= 0.5, y, 1 - y)


def binomial_p(k: int, n: int, expected: float) -> float:
    from scipy import stats
    return float(stats.binomtest(k, n, min(max(expected, 1e-9), 1 - 1e-9)).pvalue)


def proportion_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    from scipy import stats
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=level)
    return float(ci.low), float(ci.high)


def band_masks(fav_p: np.ndarray, edges=BAND_EDGES) -> list[tuple[float, float, np.ndarray]]:
    out = []
    for lo, hi in zip(edges, edges[1:]):
        top = hi >= 1.0
        mask = (fav_p >= lo) & (fav_p <= hi if top else fav_p < hi)
        out.append((lo, hi, mask))
    return out


def oracle_gain(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Pooled log-loss saved by recalibrating `mask`'s rows WITH THE ANSWER.

    Every row in the band is re-predicted at the band's realised rate, in the
    orientation the model favoured -- the best single number that band could
    possibly be assigned, chosen with hindsight.

    **What this does and does not bound.** It is an upper bound on repairing
    the thing the reliability table actually diagnoses: a band whose stated
    confidence does not match its realised rate. It is NOT an upper bound on
    every calibrator, because collapsing a band to one number throws away the
    model's ordering INSIDE that band, and a smooth monotone remedy keeps that
    ordering while moving the level. A fitted temperature can therefore beat
    this bound, and on synthetic data it does -- see
    `tests/test_calibration_audit.py`.

    So this number answers "how much is the visible miscalibration worth?" and
    `recalibrator_trial` answers "can any actual calibrator do better than the
    one already deployed?". Both are needed; neither alone settles it.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if not mask.any():
        return 0.0
    _, fav_win = favoured_view(p, y)
    realised = float(fav_win[mask].mean())
    repaired = p.copy()
    favours_a = mask & (p >= 0.5)
    favours_b = mask & (p < 0.5)
    repaired[favours_a] = realised
    repaired[favours_b] = 1.0 - realised
    return log_loss(y, p) - log_loss(y, repaired)


#: Cross-validation used by `recalibrator_trial`. Small parameter counts on
#: ~4,856 rows barely overfit, but the split is there so the reported gain is
#: an out-of-sample one rather than a fit quality.
CV_FOLDS = 5
CV_SEEDS = (0, 1, 2, 3, 4)


def _fit_affine_on_favoured(fav_p, fav_win, free_intercept: bool):
    """Least-squares-free logistic fit of `a * logit(fav_p) + b` to `fav_win`.

    Fitted on the FAVOURED axis, which keeps the remedy symmetric by
    construction: the deployed scorer is symmetrised (`predict_symmetrized`),
    so a calibrator that treated corner A differently from corner B would be
    repairing an asymmetry the model does not have.
    """
    from scipy.optimize import minimize

    z = np.log(np.clip(fav_p, 1e-9, 1 - 1e-9) / (1 - np.clip(fav_p, 1e-9, 1 - 1e-9)))

    def nll(theta):
        a = theta[0]
        b = theta[1] if free_intercept else 0.0
        u = a * z + b
        return float(np.mean(np.logaddexp(0.0, u) - fav_win * u))

    x0 = [1.0, 0.0] if free_intercept else [1.0]
    out = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
    a = float(out.x[0])
    b = float(out.x[1]) if free_intercept else 0.0
    return a, b


def _apply_affine(p, a, b):
    """Map a calibrated favoured-side rule back onto both corners."""
    p = np.asarray(p, dtype=float)
    fav_p = np.where(p >= 0.5, p, 1 - p)
    z = np.log(np.clip(fav_p, 1e-9, 1 - 1e-9) / (1 - np.clip(fav_p, 1e-9, 1 - 1e-9)))
    fav_new = 1.0 / (1.0 + np.exp(-(a * z + b)))
    return np.where(p >= 0.5, fav_new, 1 - fav_new)


def _fit_isotonic_on_favoured(fav_p, fav_win):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        fav_p, fav_win)


def _apply_isotonic(p, model):
    p = np.asarray(p, dtype=float)
    fav_p = np.where(p >= 0.5, p, 1 - p)
    fav_new = np.clip(model.predict(fav_p), 1e-6, 1 - 1e-6)
    return np.where(p >= 0.5, fav_new, 1 - fav_new)


def recalibrator_trial(p: np.ndarray, y: np.ndarray, bar: float = PROJECT_BAR) -> dict:
    """Can any ACTUAL calibrator, fitted honestly, beat what is deployed?

    The oracle bound answers a narrower question -- how much the visible
    band-level miscalibration is worth -- and cannot rule out a smooth remedy
    that keeps the model's ordering. This asks the practical question directly:
    fit each family on four fifths of the out-of-fold rows, score the fifth it
    has not seen, and report the gain over the deployed probabilities.

    The deployed scorer ALREADY carries a fitted temperature (0.85,
    post-average, `models/blend.json`), so a temperature here is asking for a
    second correction on top of the first, and a gain near zero is the
    expected and healthy answer.
    """
    from sklearn.model_selection import KFold

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    base = log_loss(y, p)
    families = {
        "temperature (1 param)": ("affine", False),
        "platt on the favoured axis (2 params)": ("affine", True),
        "isotonic on the favoured axis": ("isotonic", None),
    }
    results = {}
    for name, (kind, free) in families.items():
        gains, params = [], []
        for seed in CV_SEEDS:
            preds = np.empty_like(p)
            for train, test in KFold(n_splits=CV_FOLDS, shuffle=True,
                                     random_state=seed).split(p):
                fav_p, fav_win = favoured_view(p[train], y[train])
                if kind == "affine":
                    a, b = _fit_affine_on_favoured(fav_p, fav_win, free)
                    preds[test] = _apply_affine(p[test], a, b)
                    params.append((round(a, 4), round(b, 4)))
                else:
                    model = _fit_isotonic_on_favoured(fav_p, fav_win)
                    preds[test] = _apply_isotonic(p[test], model)
            gains.append(base - log_loss(y, preds))
        mean_gain = float(np.mean(gains))
        results[name] = {
            "cv_gain": round(mean_gain, 5),
            "per_seed": [round(g, 5) for g in gains],
            "clears_bar": bool(mean_gain > bar),
            **({"median_fitted_params": list(np.median(np.array(params), axis=0).round(4))}
               if params else {}),
        }
    return {
        "question": ("can a calibrator fitted out of sample beat the deployed "
                     "probabilities, which already carry a fitted temperature?"),
        "cv": f"{CV_FOLDS}-fold, {len(CV_SEEDS)} seeds",
        "baseline_log_loss": round(base, 4),
        "bar": bar,
        "families": results,
        "any_family_clears_the_bar": any(v["clears_bar"] for v in results.values()),
    }


def audit(dump: dict, bar: float = PROJECT_BAR) -> dict:
    p = np.asarray(dump["p_winner"], dtype=float)
    y = np.asarray(dump["y_winner"], dtype=float)
    years = np.asarray(dump["fold_year"], dtype=int)
    fav_p, fav_win = favoured_view(p, y)

    bands = band_masks(fav_p)
    tested = [(lo, hi, m) for lo, hi, m in bands if int(m.sum()) >= MIN_BAND_N]
    n_looks = len(tested)
    threshold = 0.05 / n_looks if n_looks else 0.05

    rows = []
    for lo, hi, mask in tested:
        n = int(mask.sum())
        k = int(fav_win[mask].sum())
        predicted = float(fav_p[mask].mean())
        observed = k / n
        pv = binomial_p(k, n, predicted)
        lo_ci, hi_ci = proportion_ci(k, n)
        gain = oracle_gain(p, y, mask)
        rows.append({
            "band": f"[{lo:.2f},{hi:.2f}{']' if hi >= 1.0 else ')'}",
            "lo": lo, "hi": hi, "n": n, "share_of_rows": n / len(p),
            "predicted": round(predicted, 4),
            "observed": round(observed, 4),
            "gap": round(observed - predicted, 4),
            "observed_ci95": [round(lo_ci, 4), round(hi_ci, 4)],
            "binomial_p": round(pv, 4),
            "nominally_off": bool(pv < 0.05),
            "survives_bonferroni": bool(pv < threshold),
            "oracle_gain": round(gain, 5),
            "oracle_clears_bar": bool(gain > bar),
        })

    flagged = [r for r in rows if r["nominally_off"]]
    per_year = {}
    for r in flagged:
        mask = (fav_p >= r["lo"]) & (fav_p <= r["hi"] if r["hi"] >= 1.0
                                     else fav_p < r["hi"])
        per_year[r["band"]] = {
            str(year): {
                "n": int((mask & (years == year)).sum()),
                "predicted": round(float(fav_p[mask & (years == year)].mean()), 4),
                "observed": round(float(fav_win[mask & (years == year)].mean()), 4),
            }
            for year in sorted(set(years.tolist()))
            if int((mask & (years == year)).sum()) >= 10
        }

    everything = np.ones(len(p), dtype=bool)
    all_bands_oracle = float(sum(r["oracle_gain"] for r in rows))
    return {
        "experiment": "calibration audit of the deployed scorer, out of fold",
        "generated_by": "scripts/calibration_audit.py",
        "describes": (
            "The DEPLOYED model (the SP3 hybrid) read through its winner "
            "marginal, scored strictly out of fold. Betting odds appear "
            "nowhere here; this asks only whether the model's own stated "
            "confidence matches what happened."
        ),
        "provenance": {
            "predictions": str(OOF_PREDICTIONS.relative_to(ROOT)),
            "n": int(len(p)),
            "pooled_winner_log_loss": round(log_loss(y, p), 4),
            "fold_years": sorted(set(years.tolist())),
        },
        "method": {
            "axis": ("the FAVOURED side's probability against whether that "
                     "side won -- a symmetric model averages its own errors "
                     "away when read per-corner"),
            "bands": [f"[{lo:.2f},{hi:.2f})" for lo, hi in
                      zip(BAND_EDGES, BAND_EDGES[1:])],
            "min_band_n": MIN_BAND_N,
            "n_looks": n_looks,
            "bonferroni_threshold": round(threshold, 5),
            "threshold_fixed_by": ("the number of bands, which is a constant "
                                   "in this script, not a choice made after "
                                   "seeing the p-values"),
            "oracle": ("each band re-predicted at its own realised rate -- "
                       "the answer itself. An upper bound on any buildable "
                       "calibrator, used only to rule remedies out."),
            "bar": bar,
        },
        "bands": rows,
        "flagged_bands_by_year": per_year,
        "recalibrator_trial": recalibrator_trial(p, y, bar=bar),
        "oracle_ceiling": {
            "best_single_band": max((r["oracle_gain"] for r in rows), default=0.0),
            "all_bands_at_once": round(oracle_gain_all(p, y, tested), 5),
            "sum_of_single_bands": round(all_bands_oracle, 5),
            "bar": bar,
            "any_band_worth_a_remedy": any(r["oracle_clears_bar"] for r in rows),
        },
        "verdict": verdict(rows, threshold, bar),
    }


def oracle_gain_all(p, y, tested) -> float:
    """Every tested band repaired with its own realised rate, simultaneously.

    The generous version of the generous bound: not "fix the worst band" but
    "fix every band at once, all with hindsight". If even this does not clear
    the bar there is no calibration remedy worth building.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    _, fav_win = favoured_view(p, y)
    repaired = p.copy()
    for _, _, mask in tested:
        if not mask.any():
            continue
        realised = float(fav_win[mask].mean())
        repaired[mask & (p >= 0.5)] = realised
        repaired[mask & (p < 0.5)] = 1.0 - realised
    return log_loss(y, p) - log_loss(y, repaired)


def verdict(rows, threshold, bar) -> dict:
    survivors = [r["band"] for r in rows if r["survives_bonferroni"]]
    nominal = [r["band"] for r in rows if r["nominally_off"]
               and not r["survives_bonferroni"]]
    worth = [r["band"] for r in rows if r["oracle_clears_bar"]]
    return {
        "miscalibrated_bands": survivors,
        "nominally_off_but_not_past_correction": nominal,
        "bands_where_even_an_oracle_fix_clears_the_bar": worth,
        "calibration_is_sound": not survivors,
        "a_remedy_is_worth_building": bool(worth),
        "outcome": (
            f"{len(survivors)} of {len(rows)} bands are miscalibrated past a "
            f"Bonferroni threshold of {threshold:.5f}"
            + (f"; {len(nominal)} are nominally off ({', '.join(nominal)}) and "
               "do not survive it" if nominal else "")
            + ". "
            + ("No band's ORACLE repair -- recalibration with the realised "
               f"answer -- reaches the {bar} bar, so no calibration remedy is "
               "worth building whether or not the miscalibration is real."
               if not worth else
               f"Bands {', '.join(worth)} clear the bar even under an oracle, "
               "so a remedy there is worth specifying.")
        ),
    }


def render(report: dict) -> None:
    prov, method = report["provenance"], report["method"]
    print(f"Calibration audit -- {prov['n']:,} out-of-fold fights, "
          f"pooled log-loss {prov['pooled_winner_log_loss']}")
    print(f"  {method['n_looks']} bands tested, Bonferroni threshold "
          f"p < {method['bonferroni_threshold']}\n")
    print(f"  {'band':>13s} {'n':>5s} {'pred':>7s} {'obs':>7s} {'gap':>7s} "
          f"{'p':>7s} {'oracle':>8s}  verdict")
    for r in report["bands"]:
        v = ("MISCALIBRATED" if r["survives_bonferroni"]
             else "nominal only" if r["nominally_off"] else "calibrated")
        print(f"  {r['band']:>13s} {r['n']:5d} {r['predicted']:7.3f} "
              f"{r['observed']:7.3f} {r['gap']:+7.3f} {r['binomial_p']:7.3f} "
              f"{r['oracle_gain']:+8.5f}  {v}")
    for band, years in report["flagged_bands_by_year"].items():
        print(f"\n  {band} by fold year:")
        for year, row in years.items():
            print(f"    {year}  n={row['n']:4d}  predicted {row['predicted']:.3f}  "
                  f"observed {row['observed']:.3f}  "
                  f"gap {row['observed'] - row['predicted']:+.3f}")
    ceil = report["oracle_ceiling"]
    print(f"\n  ORACLE CEILING -- how much the visible band-level gap is worth")
    print(f"  (recalibrating each band with the realised answer itself):")
    print(f"    best single band     {ceil['best_single_band']:+.5f}")
    print(f"    every band at once   {ceil['all_bands_at_once']:+.5f}")
    print(f"    shipping bar         {ceil['bar']:.5f}")

    trial = report["recalibrator_trial"]
    print(f"\n  REAL CALIBRATORS, fitted out of sample ({trial['cv']}),")
    print(f"  against the deployed probabilities at {trial['baseline_log_loss']}:")
    for name, row in trial["families"].items():
        print(f"    {name:40s} {row['cv_gain']:+.5f}"
              + ("  CLEARS THE BAR" if row["clears_bar"] else ""))
    print(f"\n  {report['verdict']['outcome']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=OOF_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--bar", type=float, default=PROJECT_BAR)
    args = parser.parse_args()

    report = audit(load_predictions(args.predictions), bar=args.bar)
    render(report)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
