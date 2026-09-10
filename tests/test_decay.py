"""The removal rule and the coverage arithmetic behind the decay decision.

`mma.decay` states the mirror image of `mma.walkforward.bar_check` -- when a
group of columns already in the deployed model has stopped earning its place.
What is worth pinning:

* the DIRECTION of every delta. `removal_verdict` takes the report WITHOUT the
  columns as the candidate, so a negative delta means REMOVING is better,
  while `keep_gain_recent` is positive when KEEPING is better. The two read in
  opposite directions on purpose, because the clause is written about keeping;
  getting either sign backwards would flip the verdict silently and read
  perfectly plausibly, which is why the first four tests here are about
  nothing else;
* the asymmetric comparisons -- two strict bars and one inclusive tolerance --
  because a candidate sitting exactly on a threshold must not cross it;
* that the weights for the joint metric are PASSED IN rather than read off the
  report's `n`, since the joint metric is a mean over the scored rows;
* that a non-comparable pair is KEEP whatever the deltas say;
* the trailing window's half-open left edge, which is the difference between
  a stable weekly number and one that wobbles on a boundary fight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mma import decay


def report(name, *, pooled_joint, pooled_winner, folds, n=4804) -> dict:
    return {
        "name": name,
        "pooled": {"n": n, "joint_log_loss": pooled_joint, "winner_log_loss": pooled_winner},
        "folds": {
            str(year): {"n": 500, "joint_log_loss": j, "winner_log_loss": w}
            for year, (j, w) in folds.items()
        },
    }


BASE_FOLDS = {2018: (2.20, 0.65), 2024: (2.10, 0.64), 2025: (2.14, 0.65)}
WEIGHTS = {"2018": 500, "2024": 600, "2025": 400}


def kept(**over):
    folds = dict(BASE_FOLDS) | over.pop("folds", {})
    return report("kept", pooled_joint=2.1432, pooled_winner=0.6437, folds=folds, **over)


def removed(*, pooled_joint=2.1432, pooled_winner=0.6437, folds=None, **over):
    return report("removed", pooled_joint=pooled_joint, pooled_winner=pooled_winner,
                  folds=dict(BASE_FOLDS) | (folds or {}), **over)


def verdict(candidate, incumbent=None, **kwargs):
    return decay.removal_verdict(
        candidate, incumbent if incumbent is not None else kept(),
        joint_weights=WEIGHTS, **kwargs
    )


# --- direction ------------------------------------------------------------

def test_identical_reports_remove_because_the_columns_buy_nothing():
    """Every delta is zero: nothing is harmed and keeping is worth nothing."""
    out = verdict(removed())
    assert out["verdict"] == "REMOVE"
    assert out["pooled"]["d_joint"] == 0.0
    assert out["recent"]["keep_gain_recent"] == 0.0


def test_a_candidate_that_is_worse_recently_means_keeping_pays_and_keeps():
    """Removing costs 0.004 on the recent folds, so keeping GAINS 0.004."""
    out = verdict(removed(folds={2024: (2.104, 0.64), 2025: (2.144, 0.65)}))
    assert out["recent"]["keep_gain_recent"] == pytest.approx(0.004, abs=1e-9)
    assert out["clauses"]["recent_fold_contribution_of_keeping"]["holds"] is False
    assert out["verdict"] == "KEEP"


def test_a_candidate_that_is_better_recently_removes():
    """Removing IMPROVES the recent folds, so keeping gains a NEGATIVE amount."""
    out = verdict(removed(folds={2024: (2.09, 0.64), 2025: (2.13, 0.65)}))
    assert out["recent"]["keep_gain_recent"] == pytest.approx(-0.01, abs=1e-9)
    assert out["verdict"] == "REMOVE"


def test_recent_is_row_weighted_not_a_plain_mean():
    """2024 carries 600 rows and 2025 400, so the folds do not weigh equally."""
    out = verdict(removed(folds={2024: (2.20, 0.64), 2025: (2.14, 0.65)}))
    # candidate recent = (600*2.20 + 400*2.14)/1000 = 2.176; incumbent 2.116
    assert out["recent"]["candidate_joint"] == pytest.approx(2.176, abs=1e-9)
    assert out["recent"]["d_joint"] == pytest.approx(0.06, abs=1e-9)


# --- the three clauses ----------------------------------------------------

def test_pooled_clause_is_do_no_harm_at_the_bar_not_a_requirement_to_improve():
    """A removal that costs 0.009 pooled joint still passes clause 1."""
    out = verdict(removed(pooled_joint=2.1432 + 0.009))
    assert out["clauses"]["pooled_joint_do_no_harm"]["holds"] is True
    assert out["verdict"] == "REMOVE"


def test_pooled_clause_fails_past_the_bar():
    out = verdict(removed(pooled_joint=2.1432 + 0.02))
    assert out["clauses"]["pooled_joint_do_no_harm"]["holds"] is False
    assert out["verdict"] == "KEEP"


def test_the_pooled_bar_is_strict_so_a_candidate_on_it_does_not_clear_it():
    out = verdict(removed(pooled_joint=2.1432 + decay.JOINT_BAR), margin=0.0)
    assert out["clauses"]["pooled_joint_do_no_harm"]["holds"] is False


def test_the_recent_bar_is_strict_so_a_candidate_on_it_does_not_clear_it():
    """A keep-gain of exactly 0.001 -> the columns earn their place."""
    out = verdict(removed(folds={2024: (2.101, 0.64), 2025: (2.141, 0.65)}), margin=0.0)
    assert out["recent"]["keep_gain_recent"] == pytest.approx(decay.RECENT_BAR, abs=1e-9)
    assert out["clauses"]["recent_fold_contribution_of_keeping"]["holds"] is False


def test_the_winner_tolerance_is_inclusive_so_a_candidate_on_it_passes():
    """Clause 3 is a tolerance on a regression, not a bar to clear."""
    out = verdict(removed(pooled_winner=0.6437 + decay.SIGMA_SEED), margin=0.0)
    assert out["clauses"]["winner_marginal_tolerance"]["holds"] is True
    assert out["verdict"] == "REMOVE"


def test_a_winner_regression_past_sigma_seed_vetoes_an_otherwise_clean_removal():
    out = verdict(removed(pooled_winner=0.6437 + 0.002))
    assert out["clauses"]["pooled_joint_do_no_harm"]["holds"] is True
    assert out["clauses"]["recent_fold_contribution_of_keeping"]["holds"] is True
    assert out["clauses"]["winner_marginal_tolerance"]["holds"] is False
    assert out["verdict"] == "KEEP"


def test_a_winner_improvement_never_trips_the_tolerance():
    out = verdict(removed(pooled_winner=0.6437 - 0.01))
    assert out["clauses"]["winner_marginal_tolerance"]["holds"] is True


# --- ambiguity and comparability -----------------------------------------

def test_a_clause_landing_on_its_threshold_is_ambiguous_not_resolved():
    out = verdict(removed(pooled_joint=2.1432 + decay.JOINT_BAR))
    assert out["verdict"] == "AMBIGUOUS"
    assert out["ambiguous_clauses"] == ["pooled_joint_do_no_harm"]


def test_ambiguity_is_reported_even_when_every_clause_holds():
    """A hair inside the recent bar is not a REMOVE anyone should act on."""
    out = verdict(removed(folds={2024: (2.10095, 0.64), 2025: (2.14095, 0.65)}))
    assert all(c["holds"] for c in out["clauses"].values())
    assert out["verdict"] == "AMBIGUOUS"


def test_a_pair_over_different_rows_is_never_a_removal():
    out = verdict(removed(n=4000))
    assert out["comparable"] is False
    assert out["verdict"] == "KEEP"


def test_a_pair_over_different_fold_years_is_never_a_removal():
    candidate = removed()
    candidate["folds"].pop("2018")
    out = verdict(candidate)
    assert out["missing_folds"] == ["2018"]
    assert out["verdict"] == "KEEP"


def test_a_non_finite_fold_metric_raises_rather_than_reading_as_a_failure():
    candidate = removed()
    candidate["folds"]["2025"]["joint_log_loss"] = float("nan")
    with pytest.raises(ValueError, match="fold 2025 joint_log_loss is not finite"):
        verdict(candidate)


# --- weights --------------------------------------------------------------

def test_weights_are_required_for_every_recent_year():
    with pytest.raises(KeyError, match="no weight for fold year"):
        decay.removal_verdict(removed(), kept(), joint_weights={"2024": 600})


def test_winner_weights_default_to_the_joint_ones_but_can_differ():
    """The joint metric is a mean over the SCORED rows, the winner over all of
    them, so the two aggregates may be weighted differently."""
    candidate = removed(folds={2024: (2.10, 0.60), 2025: (2.14, 0.70)})
    same = decay.removal_verdict(candidate, kept(), joint_weights=WEIGHTS)
    other = decay.removal_verdict(
        candidate, kept(), joint_weights=WEIGHTS,
        winner_weights={"2018": 500, "2024": 400, "2025": 600},
    )
    assert same["recent"]["candidate_winner"] == pytest.approx(0.64, abs=1e-9)
    assert other["recent"]["candidate_winner"] == pytest.approx(0.66, abs=1e-9)


def test_zero_total_weight_raises():
    with pytest.raises(ValueError, match="sum to 0"):
        decay.removal_verdict(removed(), kept(),
                              joint_weights={"2018": 1, "2024": 0, "2025": 0})


# --- coverage -------------------------------------------------------------

def _dates(*values) -> pd.Series:
    return pd.to_datetime(pd.Series(values))


def test_coverage_by_year_counts_and_shares():
    dates = _dates("2024-03-01", "2024-09-01", "2025-02-01", "2025-06-01", "2025-11-01")
    out = decay.coverage_by_year(dates, [False, True, True, True, False])
    assert out["2024"] == {"n": 2, "missing": 1, "share": 0.5}
    assert out["2025"] == {"n": 3, "missing": 2, "share": 0.6667}


def test_coverage_by_year_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="differ in length"):
        decay.coverage_by_year(_dates("2025-01-01"), [True, False])


def test_trailing_coverage_ends_at_the_tables_own_last_date():
    dates = _dates("2024-01-01", "2025-01-01", "2025-08-01", "2026-08-01")
    out = decay.trailing_coverage(dates, [False, False, True, True])
    assert out["window_end"] == "2026-08-01"
    assert out["window_start"] == "2025-08-01"
    # Half-open on the left: the 2025-08-01 row sits ON the start and is out.
    assert out["n"] == 1 and out["missing"] == 1 and out["share"] == 1.0


def test_trailing_coverage_window_can_be_pinned_with_as_of():
    dates = _dates("2025-01-01", "2025-07-01", "2026-01-01")
    out = decay.trailing_coverage(dates, [True, False, True], as_of="2025-12-31", months=12)
    assert out["n"] == 2 and out["missing"] == 1 and out["share"] == 0.5


def test_trailing_coverage_of_an_empty_window_is_nan_not_zero():
    """Zero would read as perfect coverage; there is simply nothing to say."""
    out = decay.trailing_coverage(_dates("2020-01-01"), [True], as_of="2026-01-01")
    assert out["n"] == 0
    assert np.isnan(out["share"])


def test_trailing_coverage_rejects_a_non_positive_window():
    with pytest.raises(ValueError, match="months must be positive"):
        decay.trailing_coverage(_dates("2025-01-01"), [True], months=0)


# --- the fresh-seed conjunction -------------------------------------------

def test_both_seed_sets_must_say_remove():
    """SP2.1 shipped a false positive on one seed set; this is why the rule
    is a conjunction and not a vote."""
    assert decay.confirmed_verdict("REMOVE", "REMOVE") == "REMOVE"


@pytest.mark.parametrize("fresh", ["KEEP", "AMBIGUOUS", None])
def test_a_removal_that_the_fresh_seeds_do_not_confirm_is_ambiguous(fresh):
    assert decay.confirmed_verdict("REMOVE", fresh) == "AMBIGUOUS"


def test_a_fresh_seed_removal_does_not_override_a_primary_keep():
    """Neither direction gets to win on one seed set."""
    assert decay.confirmed_verdict("KEEP", "REMOVE") == "AMBIGUOUS"


@pytest.mark.parametrize("fresh", ["KEEP", None])
def test_keeping_needs_no_confirmation_run(fresh):
    """Leaving the model alone is not a change, so there is nothing to confirm
    -- and an unrun confirmation must not turn a KEEP into a stop."""
    assert decay.confirmed_verdict("KEEP", fresh) == "KEEP"


def test_an_ambiguous_primary_stays_ambiguous_however_the_fresh_seeds_land():
    for fresh in ("REMOVE", "KEEP", "AMBIGUOUS", None):
        assert decay.confirmed_verdict("AMBIGUOUS", fresh) == "AMBIGUOUS"


# --- fold coverage is not year coverage -----------------------------------

def test_coverage_by_fold_is_not_coverage_by_year_for_the_unbounded_last_fold():
    """The last fold absorbs every later fight, so quoting the calendar year
    understates how uncovered the evidence for that fold's delta was -- the
    exact slip the decay plan's threshold justification made."""
    from mma.walkforward import make_folds

    dates = pd.Series(pd.to_datetime(
        ["2024-06-01"] * 4 + ["2025-06-01"] * 4 + ["2026-06-01"] * 4
    ))
    missing = [False] * 4 + [False] * 4 + [True] * 4
    folds = make_folds(dates, fold_years=(2024, 2025))
    by_fold = decay.coverage_by_fold(folds, missing)
    by_year = decay.coverage_by_year(dates, missing)
    assert by_year["2025"]["share"] == 0.0
    assert by_fold["2025"]["n"] == 8 and by_fold["2025"]["share"] == 0.5
    assert by_fold["2024"] == {"n": 4, "missing": 0, "share": 0.0}


def test_coverage_by_fold_of_an_empty_fold_is_nan():
    from mma.walkforward import make_folds

    dates = pd.Series(pd.to_datetime(["2025-06-01"] * 3))
    folds = make_folds(dates, fold_years=(2024, 2025))
    by_fold = decay.coverage_by_fold(folds, [True] * 3)
    assert by_fold["2024"]["n"] == 0
    assert np.isnan(by_fold["2024"]["share"])
