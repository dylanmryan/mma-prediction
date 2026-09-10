"""Where, if anywhere, does this model beat the book?

EVALUATION-ONLY, like `scripts/build_odds_benchmark.py`: betting odds are a
comparator here and never a model feature.

Two questions, both asked strictly out of fold, both pre-registered in
`docs/superpowers/plans/2026-09-09-market-edge-analysis.md` BEFORE any number
was computed.

**Part A -- subset edges on the moneyline.** The pooled answer is already
known and negative: the market beats the deployed hybrid's winner marginal by
+0.036 log-loss (`models/market_benchmark_oof.json`). This asks whether any
pre-registered SUBSET flips that, and reports all eight, losers included, with
the multiple-comparisons arithmetic attached.

**Part B -- the method-prop market.** Never compared against before. The book
prices six outcomes -- each corner by KO/TKO, submission, decision -- and
overrounds around 1.22 doing it, where its moneyline sits near 1.05. Since SP3
the model emits a coherent joint over outcome cells, so it has a matching
six-way distribution: collapse the cells over rounds. Scored as a six-way
multiclass problem, and again as method-only (three-way, ignoring who wins),
which isolates the simulator's method skill from its winner skill.

**The guards, restated because they are the whole value of the exercise.**

* Out of fold only. Fold year Y's probabilities come from a model fitted on
  fights before Y-1 and early-stopped on Y-1. The deployed model trains
  through the latest event, so its own predictions on these fights are
  in-sample; an in-sample comparison here already nearly published a "+17.8%
  ROI" on this repo.
* Slices fixed in advance. Ten pre-registered looks in total, so the
  Bonferroni threshold is p < 0.005 and it was fixed before the first p-value
  existed.
* Log-loss is not profit. Every headline gets an ROI at three edge thresholds,
  settled at the ACTUAL OFFERED decimal odds with the vig in them, with the bet
  count beside it.
* Every result here is retrospective. These fights have been reused across
  five experiments (SP1, SP2, SP2.1, SP2.2, SP3). Nothing here is prospective
  and nothing here is a betting strategy.

  OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/market_edge_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mma.evaluate import accuracy, brier_score, log_loss, n_joint_cells  # noqa: E402
from mma.joint import marginals_from_cells  # noqa: E402
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES  # noqa: E402
from mma.odds import decimal_to_implied, devig_multiway  # noqa: E402
from mma.versioning import model_version  # noqa: E402
from mma.walkforward import slice_masks  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

#: The out-of-fold dump carrying the joint cells (see the plan doc: the
#: original `hybrid_e2.json` dump has only `p_winner`, which cannot answer
#: anything about method).
OOF_PREDICTIONS = MODELS / "walkforward" / "preds" / "hybrid_e2_cells.json"
#: The walk-forward report those predictions came from -- the same candidate,
#: same seeds and the same model matrix as `hybrid_e2.json`, re-run only
#: because that run's dump carried no cells. (`hybrid_e2`'s `--drop-columns`
#: named three more columns, but `mma.tensors` and `mma.models.xgb` exclude
#: those three permanently, so naming them drops nothing.) The re-run
#: reproduces `hybrid_e2.json`'s out-of-fold winner probabilities to ZERO
#: difference on all 4,804 rows the two share, and every metric of folds
#: 2018-2024 exactly; it differs only in carrying 52 more fights, which a data
#: refresh added to the (unbounded) 2025 fold between the two runs. Since that
#: refresh it is also the dump `models/market_benchmark_oof.json` is built on,
#: so both market artifacts describe one set of out-of-fold predictions.
OOF_REPORT = MODELS / "walkforward" / "hybrid_e2_cells.json"
OUT_PATH = MODELS / "market_edge_analysis.json"

#: The prop book's six outcomes, corner-major then method. `f1_*`/`f2_*` in
#: the odds CSV map onto this once the row's corner orientation is resolved.
SIX_WAY = (
    "a_ko_tko", "a_submission", "a_decision",
    "b_ko_tko", "b_submission", "b_decision",
)
THREE_WAY = ("ko_tko", "submission", "decision")
_PROP_COLUMNS = (
    "f1_ko_odds", "f1_sub_odds", "f1_dec_odds",
    "f2_ko_odds", "f2_sub_odds", "f2_dec_odds",
)

#: Pre-registered, in the plan doc's order. Four inherited from the harness,
#: four defined here.
PREREGISTERED_SLICES = (
    "debut", "womens", "five_round", "external_missing",
    "youth_gap_5", "home_advantage", "title_fight", "short_notice",
)
#: 8 moneyline slices + 2 prop markets. Fixed before measurement.
N_PREREGISTERED_LOOKS = 10
#: Pre-registered sanity bounds on the six-way overround. A book cannot
#: underround a market it makes; anything below 1.0 is a data error, and 1.6 is
#: far outside anything a real prop book prices.
OVERROUND_MIN, OVERROUND_MAX = 1.00, 1.60
#: Youth slice: "younger by at least this many years", both ages known.
YOUTH_GAP_YEARS = 5.0
THRESHOLDS = (0.00, 0.05, 0.10)
_EPS = 1e-12


# --------------------------------------------------------------------------
# the six-way collapse
# --------------------------------------------------------------------------


def collapse_cells_to_six_way(cells, method_classes=METHOD_CLASSES,
                              round_classes=ROUND_CLASSES) -> np.ndarray:
    """`(n, 18)` outcome cells -> `(n, 6)` corner x method, summing over rounds.

    The prop market does not price the round, so the round axis is exactly what
    has to go. `mma.evaluate`'s layout is
    ``winner * n_finish_methods * n_rounds + method * n_rounds + round``,
    followed by the two decision cells, so a corner's KO block and submission
    block are contiguous runs and the collapse is a reshape plus a sum -- no
    re-derivation of the layout, and no second reading of it that could drift
    from `mma.joint`'s.

    Verified rather than asserted: the tests pin that this reproduces the
    winner and method marginals `mma.joint.marginals_from_cells` reports off
    the same cells, and that a joint composed from independent marginals
    collapses back to their outer product.
    """
    cells = np.asarray(cells, dtype=float)
    expected = n_joint_cells(method_classes, round_classes)
    if cells.ndim != 2 or cells.shape[1] != expected:
        raise ValueError(f"cells must have shape (n, {expected}); got {cells.shape}")
    n_methods = len(method_classes) - 1
    n_rounds = len(round_classes)
    block = n_methods * n_rounds
    finish = cells[:, : 2 * block].reshape(len(cells), 2, n_methods, n_rounds).sum(axis=3)
    cards = cells[:, 2 * block:]
    return np.concatenate(
        [finish[:, 0], cards[:, [0]], finish[:, 1], cards[:, [1]]], axis=1
    )


def three_way_from_six(six) -> np.ndarray:
    """Method only, ignoring who won: add the two corners' matching outcomes.

    This is the comparison that isolates the simulator's method skill from its
    winner skill -- the winner half is already known to lose to the market, so
    a six-way number mixes a known loss into whatever the method half does.
    """
    six = np.asarray(six, dtype=float)
    return six[:, :3] + six[:, 3:]


def six_way_realised_index(y_winner, y_method) -> np.ndarray:
    """The realised `SIX_WAY` class per row, or None where it cannot be scored.

    A fight whose method was never recorded (or was recorded as something
    outside the three the market prices, e.g. a DQ) has no cell in this market
    and is dropped rather than guessed at -- the same rule
    `mma.evaluate._realised_cells` applies.
    """
    y_winner = np.asarray(y_winner, dtype=float)
    out = np.empty(len(y_winner), dtype=object)
    for i, method in enumerate(y_method):
        label = None if method is None or (isinstance(method, float) and math.isnan(method)) else str(method)
        if label not in THREE_WAY:
            out[i] = None
            continue
        corner = 0 if y_winner[i] == 1.0 else 1
        out[i] = corner * 3 + THREE_WAY.index(label)
    return out


def scorable_mask(realised_index) -> np.ndarray:
    return np.array([i is not None for i in realised_index], dtype=bool)


# --------------------------------------------------------------------------
# multiclass metrics
# --------------------------------------------------------------------------


def multiclass_log_loss(realised_index, probs) -> float:
    """Mean `-log P(realised class)`. Agrees with `mma.evaluate.log_loss` on two
    classes, which is what keeps the six-way and moneyline numbers on one scale."""
    probs = np.asarray(probs, dtype=float)
    index = np.asarray(realised_index, dtype=int)
    picked = np.clip(probs[np.arange(len(index)), index], _EPS, None)
    return float(-np.mean(np.log(picked)))


def multiclass_accuracy(realised_index, probs) -> float:
    probs = np.asarray(probs, dtype=float)
    return float(np.mean(probs.argmax(axis=1) == np.asarray(realised_index, dtype=int)))


def multiclass_brier(realised_index, probs) -> float:
    """Summed squared error over the classes, averaged over rows.

    The multiclass generalisation of `mma.evaluate.brier_score`; on two classes
    it is twice the binary Brier, so it is comparable ACROSS this file's
    multiclass blocks but not against the moneyline Brier, and is labelled as
    such in the artifact.
    """
    probs = np.asarray(probs, dtype=float)
    index = np.asarray(realised_index, dtype=int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(index)), index] = 1.0
    return float(np.mean(((probs - onehot) ** 2).sum(axis=1)))


def multiclass_block(realised_index, probs) -> dict:
    return {
        "log_loss": round(multiclass_log_loss(realised_index, probs), 4),
        "accuracy": round(multiclass_accuracy(realised_index, probs), 4),
        "brier_multiclass": round(multiclass_brier(realised_index, probs), 4),
    }


def per_row_log_loss(realised_index, probs) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    index = np.asarray(realised_index, dtype=int)
    return -np.log(np.clip(probs[np.arange(len(index)), index], _EPS, None))


def per_row_binary_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --------------------------------------------------------------------------
# the paired test and the multiple-comparisons arithmetic
# --------------------------------------------------------------------------


def _normal_two_sided_p(z: float) -> float:
    """Two-sided tail of the standard normal at |z|, via `math.erfc`.

    A normal approximation to the paired t-test. Every slice reported here has
    n in the hundreds or thousands, where the two agree to well past the
    precision anyone should read off a retrospective p-value; using it keeps
    scipy out of the dependency set it is not declared in. The artifact says
    so beside every p-value it carries.
    """
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def paired_log_loss_test(model_losses, market_losses) -> dict:
    """Paired two-sided test on per-fight (model - market) log-loss.

    Pairing is the point: the same fights are scored by both, so the fight-level
    difficulty that dominates the variance of either loss on its own cancels.
    A negative `mean_delta` is the model winning.

    Returns nulls for `t_stat`/`p_value` when there is nothing to test -- fewer
    than two rows, or a difference with no variance at all -- rather than a
    fabricated number.
    """
    model = np.asarray(model_losses, dtype=float)
    market = np.asarray(market_losses, dtype=float)
    delta = model - market
    n = int(len(delta))
    out = {
        "n": n,
        "mean_delta": round(float(delta.mean()), 6) if n else None,
        "t_stat": None,
        "p_value": None,
        "p_value_method": "normal approximation to the paired t-test",
    }
    if n < 2:
        return out
    sd = float(delta.std(ddof=1))
    if sd <= 0.0:
        return out
    t = float(delta.mean() / (sd / math.sqrt(n)))
    out["t_stat"] = round(t, 4)
    out["p_value"] = round(_normal_two_sided_p(t), 6)
    return out


def multiple_comparison_thresholds(n_looks: int = N_PREREGISTERED_LOOKS,
                                   alpha: float = 0.05) -> dict:
    """The bar a nominal p-value has to clear once `n_looks` were taken.

    Stated as a function rather than as prose because the number of looks is
    the thing most easily forgotten between computing a p-value and quoting it.
    """
    if n_looks < 1:
        raise ValueError(f"there must be at least one look to correct for; got {n_looks}")
    return {
        "n_preregistered_looks": int(n_looks),
        "family_wise_alpha": alpha,
        "bonferroni_alpha": alpha / n_looks,
        "sidak_alpha": 1.0 - (1.0 - alpha) ** (1.0 / n_looks),
    }


# --------------------------------------------------------------------------
# the pre-registered slices
# --------------------------------------------------------------------------


def edge_slice_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """The eight pre-registered row subsets, in the plan doc's order.

    The first four are taken FROM `mma.walkforward.slice_masks` rather than
    reimplemented, so `debut` here is the same `debut` every candidate report
    names. The other four are defined here because they are specific to this
    analysis and adding them to the harness would change every report's shape.
    """
    harness = slice_masks(features)
    masks: dict[str, np.ndarray] = {
        name: harness[name] for name in ("debut", "womens", "five_round")
    }
    masks["external_missing"] = (
        harness["external_missing"] if "external_missing" in harness
        else np.zeros(len(features), dtype=bool)
    )

    age_a = pd.to_numeric(features["age_a"], errors="coerce")
    age_b = pd.to_numeric(features["age_b"], errors="coerce")
    gap = (age_a - age_b).abs()
    masks["youth_gap_5"] = (gap >= YOUTH_GAP_YEARS).fillna(False).to_numpy(dtype=bool)

    home_a = features["home_country_a"].astype(bool).to_numpy()
    home_b = features["home_country_b"].astype(bool).to_numpy()
    known = ~features["home_country_unknown"].astype(bool).to_numpy()
    masks["home_advantage"] = known & (home_a != home_b)

    masks["title_fight"] = features["title_fight"].astype(bool).to_numpy()

    notice_known = ~features["notice_unknown"].astype(bool).to_numpy()
    short = (features["short_notice_30_a"].astype(bool)
             | features["short_notice_30_b"].astype(bool)).to_numpy()
    masks["short_notice"] = notice_known & short

    return {name: masks[name] for name in PREREGISTERED_SLICES}


# --------------------------------------------------------------------------
# the prop market
# --------------------------------------------------------------------------


def prop_decimals(row, orientation: str) -> tuple[float, ...]:
    """The six offered decimal prices in `SIX_WAY` order.

    `orientation` is the odds row's corner order relative to fights.parquet's,
    resolved by exactly the same helpers the moneyline alignment uses -- so a
    prop row and its moneyline row can never disagree about which corner is
    which.
    """
    values = tuple(float(row[column]) for column in _PROP_COLUMNS)
    if orientation == "same":
        return values
    if orientation == "swapped":
        return values[3:] + values[:3]
    raise ValueError(f"orientation must be 'same' or 'swapped'; got {orientation!r}")


def prop_overround(decimals) -> float:
    """Sum of the six raw (vig-included) implied probabilities."""
    return float(math.fsum(decimal_to_implied(float(d)) for d in decimals))


def passes_overround_filter(overround: float) -> bool:
    """The pre-registered sanity bounds, inclusive at both ends."""
    return OVERROUND_MIN <= overround <= OVERROUND_MAX


def prop_probabilities(row, orientation: str) -> tuple[float, ...]:
    """Devigged six-way market probabilities in `SIX_WAY` order."""
    decimals = prop_decimals(row, orientation)
    return devig_multiway([decimal_to_implied(d) for d in decimals])


def prop_roi(model_probs, decimals, realised_index, threshold: float,
             reference: str = "offered") -> dict:
    """Flat 1-unit ROI over the six prop outcomes at one edge threshold.

    A bet is placed on every (fight, outcome) whose model probability exceeds
    the reference probability by more than `threshold`, and is settled at that
    outcome's ACTUAL offered decimal price.

    `reference="offered"` (the default, and the pre-registered headline) uses
    the raw implied probability of the offered price -- vig included -- so the
    model has to beat the price it would actually pay. `reference="devigged"`
    uses the normalised market probability, which is the looser bar the
    moneyline sweep in `scripts/build_odds_benchmark.py` uses; it is reported
    alongside for continuity, never instead.

    `roi_pct` is None when nothing was bet: "never bet" and "broke even" are
    different claims and must not look alike.

    Every ROI comes with an error bar, because on this market the point
    estimate alone is actively misleading: a flat stake on outcomes priced
    around 6.5 decimal has a per-bet profit spread of roughly 2.5 units, so
    even 800 bets carry a standard error near 9 percentage points. A
    positive-looking ROI whose interval spans zero is not an edge, and
    reporting the two numbers together is the only way that is visible.
    """
    model_probs = np.asarray(model_probs, dtype=float)
    decimals = np.asarray(decimals, dtype=float)
    raw = 1.0 / decimals
    if reference == "offered":
        market = raw
    elif reference == "devigged":
        market = raw / raw.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"reference must be 'offered' or 'devigged'; got {reference!r}")

    bet = (model_probs - market) > threshold
    won = np.zeros_like(bet)
    won[np.arange(len(decimals)), np.asarray(realised_index, dtype=int)] = True
    profit = np.where(won, decimals - 1.0, -1.0)[bet]
    n_bets = int(bet.sum())
    net = float(profit.sum()) if n_bets else 0.0
    out = {
        "n_bets": n_bets,
        "n_fights_with_a_bet": int(bet.any(axis=1).sum()),
        "staked": float(n_bets),
        "net": round(net, 3),
        "roi_pct": round(100.0 * net / n_bets, 3) if n_bets else None,
        "roi_se_pct": None,
        "roi_95ci_pct": None,
        "t_stat": None,
        "p_value": None,
    }
    if n_bets < 2:
        return out
    se = float(profit.std(ddof=1)) / math.sqrt(n_bets)
    out["roi_se_pct"] = round(100.0 * se, 3)
    mean = float(profit.mean())
    out["roi_95ci_pct"] = [round(100.0 * (mean - 1.96 * se), 3),
                           round(100.0 * (mean + 1.96 * se), 3)]
    if se > 0.0:
        t = mean / se
        out["t_stat"] = round(t, 4)
        out["p_value"] = round(_normal_two_sided_p(t), 6)
    return out


# --------------------------------------------------------------------------
# Part A: moneyline slices
# --------------------------------------------------------------------------


def moneyline_roi(frame: pd.DataFrame, threshold: float, reference: str = "offered") -> dict:
    """Flat 1-unit ROI on the moneyline, both corners, at one edge threshold.

    Same shape as `prop_roi` and the same `reference` choice, so Part A and
    Part B's profit numbers are computed by the same rule and can be read
    against each other.
    """
    p_a = frame["model_p_a"].to_numpy(dtype=float)
    decimals = np.stack([frame["decimal_a"].to_numpy(dtype=float),
                         frame["decimal_b"].to_numpy(dtype=float)], axis=1)
    model = np.stack([p_a, 1.0 - p_a], axis=1)
    realised = np.where(frame["y_winner"].to_numpy(dtype=float) == 1.0, 0, 1)
    return prop_roi(model, decimals, realised, threshold, reference=reference)


def moneyline_block(frame: pd.DataFrame) -> dict:
    """Model vs market on one row set: metrics, the paired test, and ROI."""
    y = frame["y_winner"].to_numpy(dtype=float)
    p_model = frame["model_p_a"].to_numpy(dtype=float)
    p_market = frame["market_implied_a"].to_numpy(dtype=float)
    model = {"log_loss": round(log_loss(y, p_model), 4),
             "accuracy": round(accuracy(y, p_model), 4),
             "brier": round(brier_score(y, p_model), 4)}
    market = {"log_loss": round(log_loss(y, p_market), 4),
              "accuracy": round(accuracy(y, p_market), 4),
              "brier": round(brier_score(y, p_market), 4)}
    return {
        "n": int(len(frame)),
        "date_range": [str(frame["date"].min().date()), str(frame["date"].max().date())],
        "model": model,
        "market": market,
        "delta_model_minus_market": {k: round(model[k] - market[k], 4) for k in model},
        "paired_log_loss_test": paired_log_loss_test(
            per_row_binary_log_loss(y, p_model), per_row_binary_log_loss(y, p_market)
        ),
        "roi_vs_offered_price": {
            f"{t:.2f}": moneyline_roi(frame, t) for t in THRESHOLDS
        },
        "roi_vs_devigged_price": {
            f"{t:.2f}": moneyline_roi(frame, t, reference="devigged") for t in THRESHOLDS
        },
    }


# --------------------------------------------------------------------------
# Part B: the prop market
# --------------------------------------------------------------------------


def prop_block(model_probs, market_probs, decimals, realised_index, dates) -> dict:
    model_probs = np.asarray(model_probs, dtype=float)
    market_probs = np.asarray(market_probs, dtype=float)
    model = multiclass_block(realised_index, model_probs)
    market = multiclass_block(realised_index, market_probs)
    block = {
        "n": int(len(model_probs)),
        "date_range": [str(pd.Timestamp(min(dates)).date()),
                       str(pd.Timestamp(max(dates)).date())],
        "model": model,
        "market": market,
        "delta_model_minus_market": {k: round(model[k] - market[k], 4) for k in model},
        "paired_log_loss_test": paired_log_loss_test(
            per_row_log_loss(realised_index, model_probs),
            per_row_log_loss(realised_index, market_probs),
        ),
    }
    if decimals is not None:
        block["roi_vs_offered_price"] = {
            f"{t:.2f}": prop_roi(model_probs, decimals, realised_index, t)
            for t in THRESHOLDS
        }
        block["roi_vs_devigged_price"] = {
            f"{t:.2f}": prop_roi(model_probs, decimals, realised_index, t,
                                 reference="devigged")
            for t in THRESHOLDS
        }
    return block


def per_year(model_probs, market_probs, realised_index, dates) -> dict:
    """The six-way comparison year by year, so a pooled number cannot hide a
    trend (and so a thin year is visibly thin)."""
    years = pd.to_datetime(pd.Series(list(dates))).dt.year.to_numpy()
    model_probs = np.asarray(model_probs, dtype=float)
    market_probs = np.asarray(market_probs, dtype=float)
    index = np.asarray(realised_index, dtype=int)
    out = {}
    for year in sorted(set(int(y) for y in years)):
        mask = years == year
        m_ll = multiclass_log_loss(index[mask], model_probs[mask])
        k_ll = multiclass_log_loss(index[mask], market_probs[mask])
        out[str(year)] = {
            "n": int(mask.sum()),
            "model_log_loss": round(m_ll, 4),
            "market_log_loss": round(k_ll, 4),
            "delta": round(m_ll - k_ll, 4),
        }
    return out


# --------------------------------------------------------------------------
# alignment (I/O side)
# --------------------------------------------------------------------------


def align_props_to_fights(odds_raw: pd.DataFrame, fights: pd.DataFrame,
                          fighters: pd.DataFrame):
    """Per-fight devigged six-way prop probabilities, in fights.parquet's a/b order.

    Corner orientation is resolved by `scripts.build_odds_benchmark`'s own
    helpers -- the id path first, the accent-folding name matcher as fallback
    -- so the props inherit the alignment the moneyline benchmark already
    validates against known historical favourites, rather than a second
    implementation that could disagree with it.
    """
    import scripts.build_odds_benchmark as bob

    fights_idx = fights.set_index("fight_id")
    known = set(fights_idx.index)
    name_index = bob.build_name_index(fighters)

    records = []
    stats = {"n_id": 0, "n_name": 0, "n_unoriented": 0, "n_bad_overround": 0}
    priced = odds_raw.dropna(subset=list(_PROP_COLUMNS))
    for fight_id, group in priced.groupby("fight_id"):
        if fight_id is None or fight_id not in known:
            continue
        row = fights_idx.loc[fight_id]
        orientation = bob._orient_by_id(group, row["fighter_a_id"], row["fighter_b_id"])
        method = "id"
        if orientation is None:
            orientation, _ = bob._orient_by_name(
                group, row["fighter_a_id"], row["fighter_b_id"], name_index)
            method = "name"
        if orientation is None:
            stats["n_unoriented"] += 1
            continue
        odds_row = group.iloc[0]
        decimals = prop_decimals(odds_row, orientation)
        overround = prop_overround(decimals)
        stats["n_id" if method == "id" else "n_name"] += 1
        records.append({
            "fight_id": fight_id,
            "prop_overround": overround,
            "prop_passes_filter": passes_overround_filter(overround),
            **{f"prop_decimal_fights_{label}": value
               for label, value in zip(SIX_WAY, decimals)},
            **{f"prop_market_fights_{label}": value
               for label, value in zip(SIX_WAY, prop_probabilities(odds_row, orientation))},
        })
        if not passes_overround_filter(overround):
            stats["n_bad_overround"] += 1
    return pd.DataFrame.from_records(records), stats


def props_to_features_convention(merged: pd.DataFrame) -> pd.DataFrame:
    """Flip the prop columns onto features.parquet's a/b order.

    Reuses the feature table's own `swapped` column, exactly as
    `scripts.build_odds_benchmark.to_features_convention` does for the
    moneyline: `swapped=True` means features "a" is fights.parquet's "b", so
    corner A's three prop outcomes and corner B's exchange.
    """
    out = merged.copy()
    swapped = out["swapped"].to_numpy(dtype=bool)
    for prefix in ("prop_market", "prop_decimal"):
        a = [f"{prefix}_fights_{label}" for label in SIX_WAY[:3]]
        b = [f"{prefix}_fights_{label}" for label in SIX_WAY[3:]]
        for target, source_same, source_swapped in zip(SIX_WAY[:3], a, b):
            out[f"{prefix}_{target}"] = np.where(
                swapped, out[source_swapped], out[source_same])
        for target, source_same, source_swapped in zip(SIX_WAY[3:], b, a):
            out[f"{prefix}_{target}"] = np.where(
                swapped, out[source_swapped], out[source_same])
    return out


def attach_oof_joint(pooled: pd.DataFrame, dump: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """The dump's out-of-fold rows, with its winner probability AND its cells.

    Returns `(rows, cells)` where `cells[i]` is the joint the dump predicted
    for `rows.iloc[i]`'s fight -- reordered onto the returned rows rather than
    handed back in the dump's own order, so the caller cannot pair them
    positionally by accident.

    Every check `scripts.build_odds_benchmark.attach_oof_predictions` makes is
    made there, including the id-keyed pairing itself. This adds the two the
    cells need: the realised METHOD the dump recorded must still be the method
    the table records for that same fight, and each row's cells must be a
    distribution. Both matter because a mis-paired six-way log-loss looks
    perfectly plausible -- it is not a number anyone could eyeball as wrong.
    """
    import scripts.build_odds_benchmark as bob

    out = bob.attach_oof_predictions(pooled, dump)
    for key in ("joint_cells", "y_method"):
        if key not in dump:
            raise ValueError(
                f"prediction dump {dump.get('name')!r} carries no {key}; re-run "
                "scripts/run_walkforward.py --candidate hybrid --dump-predictions "
                "to produce one (see the plan doc)"
            )
    dump_row = {str(fight_id): row for row, fight_id in enumerate(dump["fight_id"])}
    rows = np.array([dump_row[str(fight_id)] for fight_id in out["fight_id"]])

    dumped_method = [None if v is None else str(v) for v in dump["y_method"]]
    rebuilt = [None if pd.isna(v) else str(v) for v in out["y_method"]]
    n_bad = sum(1 for row, method in zip(rows, rebuilt) if dumped_method[row] != method)
    if n_bad:
        raise ValueError(
            f"the current table's 'y_method' disagrees with the prediction dump on "
            f"{n_bad} of {len(rebuilt)} fights that pair by id; the dump describes "
            "outcomes these fights no longer have and no six-way metric computed "
            "from it would mean anything"
        )
    cells = np.asarray(dump["joint_cells"], dtype=float)
    if cells.shape[0] != int(dump["n"]):
        raise ValueError(
            f"prediction dump says n={dump['n']} but carries {cells.shape[0]} "
            "rows of joint_cells; the dump is malformed"
        )
    cells = cells[rows]
    if not np.allclose(cells.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("out-of-fold joint cells must sum to 1 for every fight")
    return out, cells


def corner_names(matched: pd.DataFrame, fights: pd.DataFrame,
                 fighters: pd.DataFrame) -> pd.DataFrame:
    """Fighter NAMES for each row's corner A and B, in features.parquet's order.

    features.parquet carries only `fight_id`, so the names come from
    fights.parquet via fighters.parquet -- and then through the feature table's
    own `swapped` flag, because a swapped row's corner A is fights.parquet's
    corner B. Getting that flip wrong is precisely the bug the hand
    verification exists to catch, so it is applied here rather than eyeballed.
    """
    name_of = fighters.set_index("fighter_id")["name"]
    pairs = fights.set_index("fight_id")[["fighter_a_id", "fighter_b_id"]]
    joined = matched[["fight_id", "swapped"]].join(pairs, on="fight_id")
    swapped = joined["swapped"].to_numpy(dtype=bool)
    a_id = np.where(swapped, joined["fighter_b_id"], joined["fighter_a_id"])
    b_id = np.where(swapped, joined["fighter_a_id"], joined["fighter_b_id"])
    return pd.DataFrame({
        "name_a": name_of.reindex(a_id).to_numpy(),
        "name_b": name_of.reindex(b_id).to_numpy(),
    }, index=matched.index)


def hand_verified_examples(matched: pd.DataFrame, names: pd.DataFrame, six_model,
                           six_market, realised_index, n: int = 3) -> list[dict]:
    """A few fights written out in full, for a human to check the alignment on.

    A six-way comparison is exactly where a devig or a corner flip would
    masquerade as skill, and no aggregate metric can show that. These are the
    `n` most confidently-priced fights in the set: named fighters, the book's
    six prices and their devigged probabilities, the model's six, and the
    realised outcome -- enough to check corner order and method order against
    the historical record by eye.
    """
    six_market = np.asarray(six_market, dtype=float)
    six_model = np.asarray(six_model, dtype=float)
    order = np.argsort(-six_market.max(axis=1))[:n]
    out = []
    for i in (int(v) for v in order):
        row = matched.iloc[i]
        out.append({
            "fight_id": row["fight_id"],
            "date": str(row["date"].date()),
            "corner_a": names.iloc[i]["name_a"],
            "corner_b": names.iloc[i]["name_b"],
            "realised_outcome": SIX_WAY[int(realised_index[i])],
            "realised_in_words": (
                f"{names.iloc[i]['name_a' if row['y_winner'] == 1.0 else 'name_b']} "
                f"won by {row['y_method']}"
            ),
            "offered_decimals": {label: float(row[f"prop_decimal_{label}"])
                                 for label in SIX_WAY},
            "prop_overround": round(float(row["prop_overround"]), 4),
            "market_six_way_devigged": {label: round(float(v), 4)
                                        for label, v in zip(SIX_WAY, six_market[i])},
            "model_six_way": {label: round(float(v), 4)
                              for label, v in zip(SIX_WAY, six_model[i])},
        })
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=OOF_PREDICTIONS)
    parser.add_argument("--report", type=Path, default=OOF_REPORT)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    import scripts.build_odds_benchmark as bob

    print("Loading fights/features/fighters parquet ...")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    features = bob.load_features()

    print(f"Downloading odds dataset ({bob.ODDS_DATASET}) via kagglehub ...")
    odds_raw = bob.load_odds_raw()
    print(f"  {len(odds_raw)} odds rows across {odds_raw['fight_id'].nunique()} fights")

    print("Aligning the MONEYLINE (the alignment the benchmark already validates) ...")
    aligned, ml_stats = bob.align_odds_to_fights(odds_raw, fights, fighters)
    bob.validate_famous_fights(fights, fighters, aligned)

    print("Aligning the six-way METHOD PROP market ...")
    props, prop_stats = align_props_to_fights(odds_raw, fights, fighters)
    print(f"  {len(props)} fights carry all six prop prices "
          f"({prop_stats['n_bad_overround']} fail the overround filter)")

    print(f"Pairing the out-of-fold dump {args.predictions.name} ...")
    dump = json.loads(args.predictions.read_text())
    wf_report = json.loads(args.report.read_text())
    oof, cells = attach_oof_joint(bob.pooled_frame(features), dump)
    reproduced = bob.check_dump_reproduces_report(oof, wf_report)
    print(f"  {len(oof)} out-of-fold rows; dump reproduces the report's pooled metrics")

    # ---- Part A -----------------------------------------------------------
    matched_ml = bob.to_features_convention(bob.join_odds_by_id(oof, aligned))
    matched_ml["y_winner"] = matched_ml["y_winner"].astype(float)
    print(f"Part A: {len(matched_ml)} fights are out-of-fold AND moneyline-aligned")

    masks = edge_slice_masks(matched_ml)
    part_a = {"all_odds_matched_reference": moneyline_block(matched_ml)}
    for name, mask in masks.items():
        block = matched_ml[mask].reset_index(drop=True)
        part_a[name] = (
            moneyline_block(block) if len(block) >= 2
            else {"n": int(len(block)), "note": "too few rows to score"}
        )

    # ---- Part B -----------------------------------------------------------
    six_all = collapse_cells_to_six_way(cells)
    oof_b = oof.copy()
    for i, label in enumerate(SIX_WAY):
        oof_b[f"model_{label}"] = six_all[:, i]

    prop_columns = (["fight_id", "prop_overround", "prop_passes_filter"]
                    + [f"prop_market_fights_{label}" for label in SIX_WAY]
                    + [f"prop_decimal_fights_{label}" for label in SIX_WAY])
    matched_b = props_to_features_convention(
        oof_b.merge(props[prop_columns], on="fight_id", how="inner", validate="one_to_one")
    ).reset_index(drop=True)
    matched_b["y_winner"] = matched_b["y_winner"].astype(float)

    realised_all = six_way_realised_index(matched_b["y_winner"], matched_b["y_method"])
    scorable = scorable_mask(realised_all)
    kept = scorable & matched_b["prop_passes_filter"].to_numpy(dtype=bool)
    print(f"Part B: {len(matched_b)} out-of-fold fights carry prop prices; "
          f"{int(kept.sum())} are scorable and pass the overround filter")

    def _six_way_view(mask):
        frame = matched_b[mask].reset_index(drop=True)
        model = frame[[f"model_{label}" for label in SIX_WAY]].to_numpy(dtype=float)
        market = frame[[f"prop_market_{label}" for label in SIX_WAY]].to_numpy(dtype=float)
        decimals = frame[[f"prop_decimal_{label}" for label in SIX_WAY]].to_numpy(dtype=float)
        index = np.asarray(realised_all[mask], dtype=int)
        return frame, model, market, decimals, index

    frame, model6, market6, decimals6, index6 = _six_way_view(kept)
    # Re-normalise the model's six-way row: collapsing 18 cells that sum to 1
    # keeps the sum, but float error accumulates and a probability vector that
    # is 1 +/- 1e-15 should be made exact before it is scored.
    model6 = model6 / model6.sum(axis=1, keepdims=True)
    market6 = market6 / market6.sum(axis=1, keepdims=True)

    six_way = prop_block(model6, market6, decimals6, index6, frame["date"])
    six_way["per_year"] = per_year(model6, market6, index6, frame["date"])

    index3 = index6 % 3
    three_way = prop_block(three_way_from_six(model6), three_way_from_six(market6),
                           None, index3, frame["date"])
    three_way["per_year"] = per_year(three_way_from_six(model6),
                                     three_way_from_six(market6), index3, frame["date"])

    # Sensitivity: the same comparison WITHOUT the pre-registered overround
    # filter, so the filter cannot be the thing producing the result.
    frame_u, model6_u, market6_u, decimals6_u, index6_u = _six_way_view(scorable)
    model6_u = model6_u / model6_u.sum(axis=1, keepdims=True)
    market6_u = market6_u / market6_u.sum(axis=1, keepdims=True)
    unfiltered = prop_block(model6_u, market6_u, decimals6_u, index6_u, frame_u["date"])

    # The collapse, verified on the rows actually scored.
    reported = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)
    collapse_check = {
        "six_way_rows_sum_to_one": bool(
            np.allclose(collapse_cells_to_six_way(cells).sum(axis=1), 1.0, atol=1e-9)),
        "max_abs_winner_marginal_error": float(np.abs(
            collapse_cells_to_six_way(cells)[:, :3].sum(axis=1) - reported["winner"]).max()),
        "max_abs_method_marginal_error": float(np.abs(
            three_way_from_six(collapse_cells_to_six_way(cells)) - reported["method"]).max()),
        "note": (
            "The collapse must be the SAME reading of the cells that "
            "mma.joint.marginals_from_cells already reports, or the six-way "
            "comparison is scoring a distribution no other artifact vouches for."
        ),
    }

    results = {
        "computed_on": pd.Timestamp.today().date().isoformat(),
        "plan": "docs/superpowers/plans/2026-09-09-market-edge-analysis.md",
        "describes": (
            "The deployed SP3 hybrid, out of fold, against the same bookmaker's "
            "moneyline (per pre-registered slice) and six-way method-prop market."
        ),
        "odds_dataset": bob.ODDS_DATASET,
        "retrospective_warning": (
            "EVERY number here is retrospective. These fights have been reused "
            "across five experiments (SP1, SP2, SP2.1, SP2.2, SP3), the slice "
            "list was pre-registered but the fights were not fresh, and no "
            "result here has been confirmed prospectively. Prospective "
            "confirmation on fights nobody has looked at is required before any "
            "of this means anything. Nothing here is a betting strategy: no "
            "bankroll management, no line shopping, no closing-line timing, no "
            "transaction costs, no limits."
        ),
        "guards": {
            "out_of_fold": (
                "Every model probability comes from a walk-forward fold model "
                "fitted on fights strictly before the fold's inner-validation "
                "year. The deployed refit model's own predictions on these "
                "fights are in-sample and are used nowhere here."
            ),
            "preregistration": (
                "The eight slices and two markets were fixed in the plan doc "
                "and committed before the first number was computed."
            ),
            "p_values": (
                "Nominal, from a normal approximation to the paired t-test on "
                "per-fight log-loss differences, and to be read only alongside "
                "multiple_comparisons below."
            ),
            "brier_scale": (
                "'brier' on the moneyline blocks is the repo's binary Brier; "
                "'brier_multiclass' on the prop blocks is the summed-over-classes "
                "form, which is twice the binary one on two classes. Compare "
                "them within a block, never across."
            ),
        },
        "provenance": {
            "predictions": str(args.predictions.relative_to(ROOT)),
            "predictions_name": dump["name"],
            "walkforward_report": str(args.report.relative_to(ROOT)),
            "n_pooled_walkforward_rows": int(dump["n"]),
            "dump_reproduces_report_pooled": reproduced,
            "deployed_model_version": model_version(ROOT),
            "moneyline_benchmark": "models/market_benchmark_oof.json",
        },
        "multiple_comparisons": {
            **multiple_comparison_thresholds(),
            "looks": {
                "part_a_slices": list(PREREGISTERED_SLICES),
                "part_b_markets": ["six_way", "method_three_way"],
            },
            "note": (
                "Ten pre-registered looks. A nominal p of 0.05 across ten "
                "independent looks is expected to appear about 40% of the time "
                "under a true null, so a slice is only interesting below the "
                "Bonferroni threshold -- and even then it is one retrospective "
                "look at fights this project has mined five times."
            ),
        },
        "alignment": {
            "moneyline": {"n_by_id": ml_stats["n_id"], "n_by_name": ml_stats["n_name"],
                          "n_skipped": ml_stats["n_skipped"]},
            "props": {
                "n_fights_with_all_six_prices": int(len(props)),
                "n_by_id": prop_stats["n_id"], "n_by_name": prop_stats["n_name"],
                "n_unoriented": prop_stats["n_unoriented"],
                "n_failing_overround_filter": prop_stats["n_bad_overround"],
                "overround_bounds": [OVERROUND_MIN, OVERROUND_MAX],
                "median_overround": round(float(props["prop_overround"].median()), 4),
                "corner_orientation": (
                    "Resolved by scripts/build_odds_benchmark's own id-then-name "
                    "helpers, the same ones whose moneyline output is validated "
                    "against known historical favourites in this run."
                ),
            },
        },
        "part_a_moneyline_slices": part_a,
        "part_b_method_props": {
            "n_out_of_fold_fights_with_prop_prices": int(len(matched_b)),
            "n_scored": int(kept.sum()),
            "n_dropped_unscorable_method": int((~scorable).sum()),
            "n_dropped_overround_filter": int(scorable.sum() - kept.sum()),
            "collapse_verification": collapse_check,
            "six_way": six_way,
            "method_three_way": three_way,
            "six_way_without_the_overround_filter_sensitivity": unfiltered,
            "hand_verified_examples": hand_verified_examples(
                frame, corner_names(frame, fights, fighters), model6, market6, index6),
        },
    }

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {args.out}")

    print("\n=== PART A: moneyline, per pre-registered slice (out of fold) ===")
    print(f"{'slice':22s}{'n':>6s}{'model':>9s}{'market':>9s}{'delta':>9s}{'p':>10s}")
    for name in ("all_odds_matched_reference", *PREREGISTERED_SLICES):
        block = part_a[name]
        if "model" not in block:
            print(f"{name:22s}{block['n']:>6d}   {block['note']}")
            continue
        p = block["paired_log_loss_test"]["p_value"]
        print(f"{name:22s}{block['n']:>6d}{block['model']['log_loss']:>9.4f}"
              f"{block['market']['log_loss']:>9.4f}"
              f"{block['delta_model_minus_market']['log_loss']:>+9.4f}"
              f"{(f'{p:.4f}' if p is not None else 'n/a'):>10s}")

    print("\n=== PART B: the method-prop market (out of fold) ===")
    for label, block in (("six-way", six_way), ("method-only", three_way)):
        print(f"{label:14s} n={block['n']:<6d} "
              f"model={block['model']['log_loss']:.4f} "
              f"market={block['market']['log_loss']:.4f} "
              f"delta={block['delta_model_minus_market']['log_loss']:+.4f} "
              f"p={block['paired_log_loss_test']['p_value']}")

    bonferroni = results["multiple_comparisons"]["bonferroni_alpha"]
    print(f"\nBonferroni threshold over {N_PREREGISTERED_LOOKS} pre-registered "
          f"looks: p < {bonferroni:.4f}")


if __name__ == "__main__":
    main()
