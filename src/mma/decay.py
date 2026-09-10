"""The arithmetic of a REMOVAL decision, and of the coverage decay behind it.

`mma.walkforward.bar_check` states the bar an ADDITION has to clear: improve
pooled winner log-loss by more than max(0.003, 2 sigma_seed) and regress no
fold by more than 0.01. This module states the other direction -- when a group
of columns already in the deployed model has stopped earning its place -- and
the two are deliberately not mirror images.

The asymmetry, pre-registered in
`docs/superpowers/plans/2026-09-09-external-decay-decision.md` §5 and repeated
here so the code carries its own reason:

* A block fed by a STATIC snapshot pays on the folds the snapshot covers and
  does nothing on the ones it does not. Pooled log-loss averages both, so it
  keeps quoting a gain bought on coverage we will never have again. The clause
  that decides a removal is therefore the RECENT-FOLD one, at a tenth of the
  bar; the pooled clause is only do-no-harm, at the same 0.01 an addition's
  fold regression is held to.
* The simpler model has no decay to manage -- no snapshot to refresh, no
  coverage threshold to watch. That is a real reduction in maintenance risk
  that log-loss does not price, and it is why the columns must EARN their
  place on the recent folds rather than merely fail to hurt the average.

Everything here is pure: reports and frames in, dicts and floats out. The
script that reads the committed reports and writes the artifact is
`scripts/external_decay_decision.py`; the weekly coverage check is
`scripts/check_snapshot_coverage.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: The two most recent walk-forward folds -- the ones that represent the rows
#: the deployed model actually serves.
RECENT_YEARS = (2024, 2025)
#: Do-no-harm clause: a removal may not cost more pooled joint log-loss than
#: the 0.01 an addition is allowed to regress a single fold by.
JOINT_BAR = 0.01
#: The deciding clause: keeping the columns must be worth more than a tenth of
#: the bar on the recent folds.
RECENT_BAR = 0.001
#: Winner-marginal tolerance, from models/walkforward/noise_floor.json.
SIGMA_SEED = 0.000346
#: A clause landing this close to its threshold is reported as AMBIGUOUS
#: rather than resolved. No tiebreak is invented after the fact.
AMBIGUITY_MARGIN = 1e-4


def fold_metric(report: dict, year, metric: str) -> float:
    """One fold's metric, accepting an int or str year (JSON keys are str)."""
    folds = report["folds"]
    key = str(year) if str(year) in folds else year
    value = folds[key][metric]
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"fold {year} {metric} is not finite in {report.get('name')!r}")
    return float(value)


def weighted_mean(report: dict, metric: str, years, weights: dict) -> float:
    """Row-weighted mean of `metric` over `years`.

    `weights` maps year -> number of rows the metric was averaged over. It is
    passed in rather than read off the report because the report's `n` is the
    fold's ROW count, and the joint metric is a mean over the SCORED rows
    (method known, and round known when the method is a finish), which is a
    smaller and differently-shaped number. Getting that wrong would silently
    reweight the two folds against each other, so the caller states it.
    """
    years = [str(y) for y in years]
    missing = [y for y in years if y not in {str(k) for k in weights}]
    if missing:
        raise KeyError(f"no weight for fold year(s) {missing}")
    w = {str(k): float(v) for k, v in weights.items()}
    total = sum(w[y] for y in years)
    if total <= 0:
        raise ValueError(f"weights for {years} sum to {total}")
    return sum(w[y] * fold_metric(report, y, metric) for y in years) / total


def removal_verdict(
    candidate: dict,
    incumbent: dict,
    *,
    joint_weights: dict,
    winner_weights: dict | None = None,
    recent_years=RECENT_YEARS,
    joint_bar: float = JOINT_BAR,
    recent_bar: float = RECENT_BAR,
    sigma_seed: float = SIGMA_SEED,
    margin: float = AMBIGUITY_MARGIN,
) -> dict:
    """Apply the pre-registered removal rule to one candidate/incumbent pair.

    `candidate` is the report with the columns REMOVED, `incumbent` the
    deployed model that keeps them, so every delta is
    ``removed - kept`` and NEGATIVE means removing them is better.

    Three clauses, all of which must hold to REMOVE:

    1. ``d_pooled_joint < joint_bar``  -- do no harm to the primary metric;
    2. ``keep_gain_recent < recent_bar`` -- keeping them is not worth a tenth
       of the bar on the recent folds. The GAIN FROM KEEPING is stated in the
       direction English states it: ``removed - kept``, POSITIVE when removing
       raises the loss, i.e. when keeping the columns helps. It is the same
       number as ``d_recent_joint`` and is reported under both names so the
       clause reads the way it is written down;
    3. ``d_pooled_winner <= sigma_seed`` -- the winner marginal does not
       regress past the seed noise floor.

    Clauses 1 and 2 are strict, clause 3 is inclusive: 1 and 2 are bars, 3 is
    a tolerance on a regression, and a tolerance includes its endpoint. The
    pair must be `comparable` -- same fold years, same pooled `n` -- or the
    verdict is KEEP whatever the deltas say, because two reports over
    different rows are not a comparison.

    A clause whose margin is smaller than `margin` makes the verdict
    AMBIGUOUS, which is neither REMOVE nor KEEP: it is the instruction to stop
    and report.
    """
    winner_weights = joint_weights if winner_weights is None else winner_weights
    cand_years, inc_years = set(map(str, candidate["folds"])), set(map(str, incumbent["folds"]))
    missing_folds = sorted(cand_years ^ inc_years)
    comparable = (
        not missing_folds
        and candidate["pooled"].get("n") == incumbent["pooled"].get("n")
    )

    d_pooled_joint = round(
        float(candidate["pooled"]["joint_log_loss"]) - float(incumbent["pooled"]["joint_log_loss"]), 6
    )
    d_pooled_winner = round(
        float(candidate["pooled"]["winner_log_loss"]) - float(incumbent["pooled"]["winner_log_loss"]), 6
    )
    recent_cand = weighted_mean(candidate, "joint_log_loss", recent_years, joint_weights)
    recent_inc = weighted_mean(incumbent, "joint_log_loss", recent_years, joint_weights)
    d_recent_joint = round(recent_cand - recent_inc, 6)
    # The gain from KEEPING, in the direction the rule is written: positive
    # when the removed model loses more, i.e. when the columns are pulling
    # their weight on the folds that represent the future. Identical to
    # `d_recent_joint`; named separately because the clause below is about
    # keeping and a reader should not have to negate anything in their head.
    keep_gain_recent = d_recent_joint

    recent_cand_w = weighted_mean(candidate, "winner_log_loss", recent_years, winner_weights)
    recent_inc_w = weighted_mean(incumbent, "winner_log_loss", recent_years, winner_weights)
    d_recent_winner = round(recent_cand_w - recent_inc_w, 6)

    clauses = {
        "pooled_joint_do_no_harm": {
            "value": d_pooled_joint, "threshold": joint_bar, "comparison": "<",
            "holds": bool(d_pooled_joint < joint_bar),
            "margin": round(abs(d_pooled_joint - joint_bar), 6),
        },
        "recent_fold_contribution_of_keeping": {
            "value": keep_gain_recent, "threshold": recent_bar, "comparison": "<",
            "holds": bool(keep_gain_recent < recent_bar),
            "margin": round(abs(keep_gain_recent - recent_bar), 6),
        },
        "winner_marginal_tolerance": {
            "value": d_pooled_winner, "threshold": sigma_seed, "comparison": "<=",
            "holds": bool(d_pooled_winner <= sigma_seed),
            "margin": round(abs(d_pooled_winner - sigma_seed), 6),
        },
    }
    ambiguous = [name for name, c in clauses.items() if c["margin"] < margin]
    all_hold = all(c["holds"] for c in clauses.values())
    if not comparable:
        verdict = "KEEP"
    elif ambiguous:
        verdict = "AMBIGUOUS"
    else:
        verdict = "REMOVE" if all_hold else "KEEP"

    return {
        "candidate": candidate.get("name"),
        "incumbent": incumbent.get("name"),
        "comparable": bool(comparable),
        "missing_folds": missing_folds,
        "recent_years": [int(y) for y in recent_years],
        "pooled": {
            "candidate_joint": float(candidate["pooled"]["joint_log_loss"]),
            "incumbent_joint": float(incumbent["pooled"]["joint_log_loss"]),
            "candidate_winner": float(candidate["pooled"]["winner_log_loss"]),
            "incumbent_winner": float(incumbent["pooled"]["winner_log_loss"]),
            "d_joint": d_pooled_joint,
            "d_winner": d_pooled_winner,
        },
        "recent": {
            "candidate_joint": round(recent_cand, 6),
            "incumbent_joint": round(recent_inc, 6),
            "candidate_winner": round(recent_cand_w, 6),
            "incumbent_winner": round(recent_inc_w, 6),
            "d_joint": d_recent_joint,
            "d_winner": d_recent_winner,
            "keep_gain_recent": keep_gain_recent,
        },
        "fold_deltas_joint": {
            str(y): round(fold_metric(candidate, y, "joint_log_loss")
                          - fold_metric(incumbent, y, "joint_log_loss"), 6)
            for y in candidate["folds"] if str(y) in {str(k) for k in incumbent["folds"]}
        },
        "fold_deltas_winner": {
            str(y): round(fold_metric(candidate, y, "winner_log_loss")
                          - fold_metric(incumbent, y, "winner_log_loss"), 6)
            for y in candidate["folds"] if str(y) in {str(k) for k in incumbent["folds"]}
        },
        "clauses": clauses,
        "ambiguous_clauses": ambiguous,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# coverage decay
# --------------------------------------------------------------------------
# The decision above is only as good as the assumption that the folds it was
# measured on still describe the rows being served. These two helpers are what
# lets that assumption be checked rather than assumed -- the second one weekly,
# in the Action.


def coverage_by_year(dates, missing) -> dict[str, dict]:
    """`missing` share per calendar year of `dates`, plus the row counts.

    Both arguments are row-aligned; `missing` is the fight-level
    `external_missing` flag (True = at least one corner the snapshot has never
    seen), so the reported share RISES as the static snapshot ages.
    """
    d = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    m = pd.Series(list(missing)).reset_index(drop=True).astype(bool)
    if len(d) != len(m):
        raise ValueError(f"dates and missing differ in length ({len(d)} vs {len(m)})")
    out = {}
    for year, index in d.groupby(d.dt.year).groups.items():
        rows = m.loc[index]
        out[str(int(year))] = {
            "n": int(len(rows)),
            "missing": int(rows.sum()),
            "share": round(float(rows.mean()), 4),
        }
    return out


def trailing_coverage(dates, missing, *, as_of=None, months: int = 12) -> dict:
    """The `missing` share over the trailing `months` of the table.

    The window ends at `as_of` (default: the latest date present) and is
    half-open on the left, `(as_of - months, as_of]`, so a row exactly
    `months` back is outside it. Measuring on the TABLE's own last date rather
    than on today's date is deliberate: the check is about how much of the
    data the model was fitted on the snapshot covers, and wall-clock time
    would make the number drift between runs that read the same table.
    """
    if months <= 0:
        raise ValueError(f"months must be positive (got {months})")
    d = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    m = pd.Series(list(missing)).reset_index(drop=True).astype(bool)
    if len(d) != len(m):
        raise ValueError(f"dates and missing differ in length ({len(d)} vs {len(m)})")
    end = pd.Timestamp(d.max() if as_of is None else as_of)
    start = end - pd.DateOffset(months=months)
    window = (d > start) & (d <= end)
    rows = m[window]
    return {
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "months": int(months),
        "n": int(len(rows)),
        "missing": int(rows.sum()),
        "share": round(float(rows.mean()), 4) if len(rows) else float("nan"),
    }
