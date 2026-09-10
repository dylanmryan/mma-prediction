"""Pure helpers behind scripts/market_edge_analysis.py.

Everything here runs on synthetic frames: no network, no kagglehub, no model
artifacts, and nothing recomputes a committed artifact. The point of the file
is that the arithmetic which decides "does the model beat the book on this
slice / in this market" is executable and pinned, rather than prose applied by
hand inside a script.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import scripts.market_edge_analysis as mea
from mma.evaluate import log_loss, n_joint_cells
from mma.joint import compose_joint_cells, marginals_from_cells
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES


# --- the six-way collapse ---------------------------------------------------


def test_six_way_labels_are_corner_major_and_method_minor():
    assert mea.SIX_WAY == (
        "a_ko_tko", "a_submission", "a_decision",
        "b_ko_tko", "b_submission", "b_decision",
    )


def test_collapse_sums_to_one_for_every_row():
    rng = np.random.default_rng(0)
    cells = rng.random((7, n_joint_cells(METHOD_CLASSES, ROUND_CLASSES)))
    cells /= cells.sum(axis=1, keepdims=True)
    six = mea.collapse_cells_to_six_way(cells, METHOD_CLASSES, ROUND_CLASSES)
    assert six.shape == (7, 6)
    assert six.sum(axis=1) == pytest.approx(np.ones(7))


def test_collapse_reproduces_the_winner_marginal_the_joint_module_reports():
    """The collapse must not be a second, subtly different reading of the same
    cells -- corner A's three prop outcomes ARE its winner marginal."""
    rng = np.random.default_rng(1)
    cells = rng.random((25, n_joint_cells(METHOD_CLASSES, ROUND_CLASSES)))
    cells /= cells.sum(axis=1, keepdims=True)
    six = mea.collapse_cells_to_six_way(cells, METHOD_CLASSES, ROUND_CLASSES)
    expected = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)["winner"]
    assert six[:, :3].sum(axis=1) == pytest.approx(expected)


def test_collapse_reproduces_the_method_marginal_the_joint_module_reports():
    rng = np.random.default_rng(2)
    cells = rng.random((25, n_joint_cells(METHOD_CLASSES, ROUND_CLASSES)))
    cells /= cells.sum(axis=1, keepdims=True)
    six = mea.collapse_cells_to_six_way(cells, METHOD_CLASSES, ROUND_CLASSES)
    expected = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)["method"]
    assert mea.three_way_from_six(six) == pytest.approx(expected)


def test_collapse_of_a_composed_joint_is_the_outer_product_of_its_marginals():
    """A joint built from independent marginals must collapse back to
    P(winner) x P(method) -- the case where the right answer is known in
    closed form, so a transposed or mis-strided reshape cannot pass."""
    winner = np.array([0.6, 0.25])
    method = np.array([[0.5, 0.2, 0.3], [0.1, 0.1, 0.8]])
    rounds = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    cells = compose_joint_cells(winner, method, rounds, METHOD_CLASSES, ROUND_CLASSES)
    six = mea.collapse_cells_to_six_way(cells, METHOD_CLASSES, ROUND_CLASSES)
    expected = np.concatenate(
        [winner[:, None] * method, (1.0 - winner)[:, None] * method], axis=1
    )
    assert six == pytest.approx(expected)


def test_collapse_rejects_the_wrong_cell_width():
    with pytest.raises(ValueError, match="18"):
        mea.collapse_cells_to_six_way(np.zeros((3, 12)), METHOD_CLASSES, ROUND_CLASSES)


def test_three_way_from_six_adds_the_two_corners():
    six = np.array([[0.1, 0.2, 0.3, 0.05, 0.15, 0.2]])
    assert mea.three_way_from_six(six) == pytest.approx(np.array([[0.15, 0.35, 0.5]]))


# --- the realised outcome ---------------------------------------------------


def test_realised_index_maps_each_corner_and_method_to_its_own_class():
    y_winner = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    y_method = ["ko_tko", "submission", "decision", "ko_tko", "submission", "decision"]
    assert list(mea.six_way_realised_index(y_winner, y_method)) == [0, 1, 2, 3, 4, 5]


def test_realised_index_is_none_for_an_unrecorded_method():
    idx = mea.six_way_realised_index(np.array([1.0, 0.0]), ["dq", None])
    assert list(idx) == [None, None]


def test_scorable_mask_drops_exactly_the_unscorable_rows():
    idx = np.array([0, None, 4], dtype=object)
    assert list(mea.scorable_mask(idx)) == [True, False, True]


# --- multiclass metrics -----------------------------------------------------


def test_multiclass_log_loss_matches_binary_log_loss_on_two_classes():
    """The multiclass scorer must agree with the repo's existing binary one
    where the two overlap, or the six-way and moneyline numbers are not on the
    same scale and cannot be discussed together."""
    p_a = np.array([0.7, 0.2, 0.55, 0.9])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    probs = np.stack([p_a, 1.0 - p_a], axis=1)
    index = np.where(y == 1.0, 0, 1)
    assert mea.multiclass_log_loss(index, probs) == pytest.approx(log_loss(y, p_a))


def test_multiclass_log_loss_of_a_confident_correct_call_is_near_zero():
    probs = np.array([[0.999, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002]])
    assert mea.multiclass_log_loss(np.array([0]), probs) == pytest.approx(
        -math.log(0.999), abs=1e-9
    )


def test_multiclass_log_loss_of_a_uniform_six_way_is_log_six():
    probs = np.full((5, 6), 1 / 6)
    assert mea.multiclass_log_loss(np.arange(5) % 6, probs) == pytest.approx(math.log(6))


def test_multiclass_accuracy_is_argmax_agreement():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    assert mea.multiclass_accuracy(np.array([0, 0, 2]), probs) == pytest.approx(2 / 3)


def test_multiclass_brier_is_the_summed_squared_error_over_classes():
    probs = np.array([[0.7, 0.2, 0.1]])
    expected = (0.7 - 1) ** 2 + 0.2**2 + 0.1**2
    assert mea.multiclass_brier(np.array([0]), probs) == pytest.approx(expected)


def test_multiclass_brier_of_a_perfect_prediction_is_zero():
    assert mea.multiclass_brier(np.array([1]), np.array([[0.0, 1.0, 0.0]])) == 0.0


# --- the paired test --------------------------------------------------------


def test_paired_test_reports_a_negative_mean_when_the_model_loses_less():
    model = np.array([0.4, 0.5, 0.3, 0.45])
    market = np.array([0.6, 0.7, 0.5, 0.65])
    out = mea.paired_log_loss_test(model, market)
    assert out["mean_delta"] < 0
    assert out["n"] == 4


def test_paired_test_on_identical_losses_is_a_flat_null():
    losses = np.array([0.5, 0.6, 0.7, 0.8])
    out = mea.paired_log_loss_test(losses, losses)
    assert out["mean_delta"] == 0.0
    assert out["t_stat"] is None and out["p_value"] is None


def test_paired_test_p_value_is_two_sided():
    """Flipping the sign of every difference must not change the p-value."""
    a = np.array([0.1, 0.4, 0.2, 0.5, 0.3, 0.35])
    b = np.array([0.5, 0.6, 0.55, 0.7, 0.5, 0.62])
    assert mea.paired_log_loss_test(a, b)["p_value"] == pytest.approx(
        mea.paired_log_loss_test(b, a)["p_value"]
    )


def test_paired_test_needs_at_least_two_rows():
    out = mea.paired_log_loss_test(np.array([0.5]), np.array([0.6]))
    assert out["p_value"] is None


def test_paired_test_p_value_falls_as_a_consistent_gap_repeats():
    """Same mean gap, same spread, twenty times the fights -- the evidence is
    what changes, so the p-value must be the thing that moves."""
    delta = -0.02 + np.linspace(-0.15, 0.15, 10)
    small = mea.paired_log_loss_test(0.5 + delta, np.full(10, 0.5))
    tiled = np.tile(delta, 20)
    large = mea.paired_log_loss_test(0.5 + tiled, np.full(200, 0.5))
    assert small["mean_delta"] == pytest.approx(large["mean_delta"])
    assert 0.0 < large["p_value"] < small["p_value"] < 1.0


def test_multiple_comparison_thresholds_for_ten_looks():
    out = mea.multiple_comparison_thresholds(n_looks=10, alpha=0.05)
    assert out["bonferroni_alpha"] == pytest.approx(0.005)
    assert out["sidak_alpha"] == pytest.approx(1 - 0.95 ** (1 / 10))


def test_multiple_comparison_thresholds_reject_zero_looks():
    with pytest.raises(ValueError, match="at least one"):
        mea.multiple_comparison_thresholds(n_looks=0)


# --- the pre-registered slices ----------------------------------------------


def _slice_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "debut_a": [True, False, False, False, False, False],
        "debut_b": [False, False, False, False, False, False],
        "weight_class": ["Women's Bantamweight", "Lightweight", "Lightweight",
                         "Lightweight", "Lightweight", "Lightweight"],
        "scheduled_rounds": pd.array([3, 5, 3, 3, 3, 3], dtype="Int64"),
        "external_missing": [False, False, True, False, False, False],
        "age_a": [30.0, 30.0, 30.0, 36.0, 30.0, np.nan],
        "age_b": [30.0, 30.0, 30.0, 30.0, 30.0, 20.0],
        "home_country_a": [False, False, False, False, True, True],
        "home_country_b": [False, False, False, False, False, True],
        "home_country_unknown": [False, False, False, False, False, False],
        "title_fight": [False, True, False, False, False, False],
        "short_notice_30_a": [False, False, False, False, False, True],
        "short_notice_30_b": [False, False, False, False, False, False],
        "notice_unknown": [True, True, True, True, True, False],
    })


def test_the_slice_list_is_exactly_the_eight_that_were_pre_registered():
    assert list(mea.edge_slice_masks(_slice_frame())) == list(mea.PREREGISTERED_SLICES)
    assert len(mea.PREREGISTERED_SLICES) == 8


def test_the_four_harness_slices_keep_the_harness_definition():
    """`debut`, `womens`, `five_round` and `external_missing` are inherited, not
    reimplemented, so a change to the harness cannot silently desynchronise
    them from what every candidate report calls by the same name."""
    from mma.walkforward import slice_masks

    frame = _slice_frame()
    ours = mea.edge_slice_masks(frame)
    theirs = slice_masks(frame)
    for name in ("debut", "womens", "five_round", "external_missing"):
        assert list(ours[name]) == list(theirs[name])


def test_youth_gap_needs_five_years_and_both_ages_known():
    masks = mea.edge_slice_masks(_slice_frame())
    # row 3 is a 6-year gap; row 5 has a 10-year gap but a missing age_a.
    assert list(masks["youth_gap_5"]) == [False, False, False, True, False, False]


def test_home_advantage_needs_the_corners_to_differ_and_the_flag_to_be_known():
    masks = mea.edge_slice_masks(_slice_frame())
    # row 4 has exactly one corner at home; row 5 has both at home.
    assert list(masks["home_advantage"]) == [False, False, False, False, True, False]


def test_home_advantage_excludes_rows_where_the_flag_is_unknown():
    frame = _slice_frame()
    frame.loc[4, "home_country_unknown"] = True
    assert not mea.edge_slice_masks(frame)["home_advantage"].any()


def test_short_notice_excludes_rows_whose_notice_is_unknown():
    masks = mea.edge_slice_masks(_slice_frame())
    assert list(masks["short_notice"]) == [False, False, False, False, False, True]


def test_title_fight_slice_is_the_column():
    assert list(mea.edge_slice_masks(_slice_frame())["title_fight"]) == [
        False, True, False, False, False, False
    ]


# --- prop odds alignment and sanity filter ----------------------------------


def test_prop_probabilities_are_devigged_in_the_six_way_order():
    row = {"f1_ko_odds": 4.0, "f1_sub_odds": 8.0, "f1_dec_odds": 4.0,
           "f2_ko_odds": 4.0, "f2_sub_odds": 8.0, "f2_dec_odds": 4.0}
    out = mea.prop_probabilities(row, orientation="same")
    assert sum(out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(out[3])  # symmetric book, symmetric answer


def test_prop_probabilities_swap_the_corners_when_the_odds_row_is_reversed():
    row = {"f1_ko_odds": 2.0, "f1_sub_odds": 10.0, "f1_dec_odds": 10.0,
           "f2_ko_odds": 20.0, "f2_sub_odds": 20.0, "f2_dec_odds": 20.0}
    same = mea.prop_probabilities(row, orientation="same")
    swapped = mea.prop_probabilities(row, orientation="swapped")
    assert swapped == pytest.approx(tuple(same[3:]) + tuple(same[:3]))


def test_prop_decimals_follow_the_same_orientation_as_the_probabilities():
    row = {"f1_ko_odds": 2.0, "f1_sub_odds": 10.0, "f1_dec_odds": 11.0,
           "f2_ko_odds": 20.0, "f2_sub_odds": 21.0, "f2_dec_odds": 22.0}
    assert mea.prop_decimals(row, "same") == (2.0, 10.0, 11.0, 20.0, 21.0, 22.0)
    assert mea.prop_decimals(row, "swapped") == (20.0, 21.0, 22.0, 2.0, 10.0, 11.0)


def test_overround_is_the_raw_implied_sum():
    decimals = (4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    assert mea.prop_overround(decimals) == pytest.approx(1.5)


def test_the_sanity_filter_keeps_a_normal_book_and_drops_an_underround():
    assert mea.passes_overround_filter(1.22)
    assert mea.passes_overround_filter(1.00)
    assert mea.passes_overround_filter(1.60)
    assert not mea.passes_overround_filter(0.999)
    assert not mea.passes_overround_filter(1.601)


# --- ROI --------------------------------------------------------------------


def test_prop_roi_bets_only_where_the_model_beats_the_offered_price():
    probs = np.array([[0.60, 0.05, 0.10, 0.10, 0.05, 0.10]])
    decimals = np.array([[2.0, 20.0, 10.0, 10.0, 20.0, 10.0]])  # raw implied .5/.05/.1/.1/.05/.1
    realised = np.array([0])
    out = mea.prop_roi(probs, decimals, realised, threshold=0.0)
    assert out["n_bets"] == 1  # only outcome 0 has model .60 > offered .50
    assert out["net"] == pytest.approx(1.0)  # won at 2.0 decimal


def test_prop_roi_settles_a_loser_at_minus_one_unit():
    probs = np.array([[0.60, 0.05, 0.10, 0.10, 0.05, 0.10]])
    decimals = np.array([[2.0, 20.0, 10.0, 10.0, 20.0, 10.0]])
    out = mea.prop_roi(probs, decimals, np.array([3]), threshold=0.0)
    assert out["n_bets"] == 1 and out["net"] == pytest.approx(-1.0)
    assert out["roi_pct"] == pytest.approx(-100.0)


def test_prop_roi_threshold_removes_thin_edges():
    probs = np.array([[0.52, 0.05, 0.10, 0.10, 0.05, 0.18]])
    decimals = np.array([[2.0, 20.0, 10.0, 10.0, 20.0, 10.0]])
    realised = np.array([0])
    assert mea.prop_roi(probs, decimals, realised, threshold=0.0)["n_bets"] == 2
    assert mea.prop_roi(probs, decimals, realised, threshold=0.05)["n_bets"] == 1


def test_prop_roi_with_no_bets_reports_a_null_roi_not_a_zero():
    """A 0% ROI and 'never bet' are different claims and must not look alike."""
    probs = np.full((1, 6), 0.05)
    decimals = np.full((1, 6), 2.0)
    out = mea.prop_roi(probs, decimals, np.array([0]), threshold=0.0)
    assert out["n_bets"] == 0 and out["roi_pct"] is None


def test_prop_roi_against_the_devigged_reference_bets_more_often():
    """Betting off the devigged probability ignores the book's margin, so it is
    the looser bar -- reported alongside, never instead of, the vig-inclusive
    one."""
    # A book overrounding to 1.28 -- outcome 2 is priced at a raw .1667 but a
    # devigged .1301, and the model's .14 sits between the two.
    probs = np.array([[0.65, 0.04, 0.14, 0.05, 0.04, 0.08]])
    decimals = np.array([[1.6, 12.0, 6.0, 6.0, 12.0, 6.0]])
    realised = np.array([0])
    vig = mea.prop_roi(probs, decimals, realised, threshold=0.0)
    fair = mea.prop_roi(probs, decimals, realised, threshold=0.0, reference="devigged")
    assert vig["n_bets"] == 1 and fair["n_bets"] == 2


def test_prop_roi_reports_an_error_bar_not_just_a_point_estimate():
    """A flat-stake ROI on long-odds props has an enormous standard error, so
    the point estimate on its own is not a finding. The interval is what says
    whether a positive-looking ROI is distinguishable from zero at all."""
    probs = np.tile(np.array([0.60, 0.05, 0.10, 0.10, 0.05, 0.10]), (40, 1))
    decimals = np.tile(np.array([2.0, 20.0, 10.0, 10.0, 20.0, 10.0]), (40, 1))
    realised = np.array([0, 3] * 20)
    out = mea.prop_roi(probs, decimals, realised, threshold=0.0)
    assert out["n_bets"] == 40
    assert out["roi_se_pct"] > 0
    lo, hi = out["roi_95ci_pct"]
    assert lo < out["roi_pct"] < hi
    assert out["p_value"] is not None


def test_prop_roi_error_bar_is_none_when_a_single_bet_was_placed():
    """One bet has no spread to estimate, and inventing one would be worse than
    saying so."""
    probs = np.array([[0.60, 0.05, 0.10, 0.10, 0.05, 0.10]])
    decimals = np.array([[2.0, 20.0, 10.0, 10.0, 20.0, 10.0]])
    out = mea.prop_roi(probs, decimals, np.array([0]), threshold=0.0)
    assert out["n_bets"] == 1
    assert out["roi_se_pct"] is None and out["p_value"] is None


def test_prop_roi_error_bar_shrinks_as_the_same_bets_repeat():
    probs = np.tile(np.array([0.60, 0.05, 0.10, 0.10, 0.05, 0.10]), (20, 1))
    decimals = np.tile(np.array([2.0, 20.0, 10.0, 10.0, 20.0, 10.0]), (20, 1))
    small = mea.prop_roi(probs, decimals, np.array([0, 3] * 10), threshold=0.0)
    big = mea.prop_roi(np.tile(probs, (10, 1)), np.tile(decimals, (10, 1)),
                       np.array([0, 3] * 100), threshold=0.0)
    assert big["roi_pct"] == pytest.approx(small["roi_pct"])
    assert big["roi_se_pct"] < small["roi_se_pct"]


def test_prop_roi_of_a_break_even_book_is_not_significant():
    """Betting a fair coin at fair odds must not look like an edge."""
    # Only outcome 0 is bet (model .55 over a raw .50); the rest are priced at
    # a raw .125 the model never reaches. It is settled at evens and lands
    # exactly half the time.
    probs = np.tile(np.array([0.55, 0.09, 0.09, 0.09, 0.09, 0.09]), (100, 1))
    decimals = np.tile(np.array([2.0, 8.0, 8.0, 8.0, 8.0, 8.0]), (100, 1))
    realised = np.array([0, 3] * 50)
    out = mea.prop_roi(probs, decimals, realised, threshold=0.0)
    assert out["roi_pct"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)


# --- pairing the joint dump to the table ------------------------------------


def _joint_pooled(n: int = 6) -> pd.DataFrame:
    """A stand-in for the walk-forward evaluation rows, with a method label."""
    return pd.DataFrame(
        {
            "fight_id": [f"{i:016x}" for i in range(n)],
            "fold_year": [2018 + i % 2 for i in range(n)],
            "y_winner": [float(i % 2) for i in range(n)],
            "y_method": [["ko_tko", "submission", "decision"][i % 3] for i in range(n)],
            "swapped": [bool(i % 2) for i in range(n)],
            "date": pd.date_range("2018-01-01", periods=n, freq="400D"),
        }
    )


def _joint_dump(pooled: pd.DataFrame) -> dict:
    n = len(pooled)
    n_cells = n_joint_cells(METHOD_CLASSES, ROUND_CLASSES)
    cells = np.tile(np.arange(1, n + 1, dtype=float)[:, None], (1, n_cells))
    cells /= cells.sum(axis=1, keepdims=True)
    return {
        "name": "synthetic",
        "n": n,
        "fight_id": [str(v) for v in pooled["fight_id"]],
        "fold_year": [int(v) for v in pooled["fold_year"]],
        "y_winner": [float(v) for v in pooled["y_winner"]],
        "p_winner": list(np.linspace(0.2, 0.8, n)),
        "y_method": [str(v) for v in pooled["y_method"]],
        "joint_cells": [list(row) for row in cells],
    }


def test_attach_oof_joint_returns_cells_aligned_to_the_rows_it_returns():
    pooled = _joint_pooled()
    dump = _joint_dump(pooled)
    out, cells = mea.attach_oof_joint(pooled, dump)
    assert list(out["fight_id"]) == dump["fight_id"]
    assert cells.shape == (len(out), n_joint_cells(METHOD_CLASSES, ROUND_CLASSES))
    assert cells == pytest.approx(np.asarray(dump["joint_cells"], dtype=float))


def test_attach_oof_joint_keeps_each_fights_own_cells_when_the_table_grows():
    """The row a fight's cells sit on is the dump's, not the table's. A table
    that has gained fights since the dump would shift every later row by one
    under a positional read, and the six-way log-loss would still look
    plausible -- so the cells travel with the fight id."""
    pooled = _joint_pooled(6)
    dump = _joint_dump(pooled)
    grown = pd.concat(
        [pooled.iloc[:2],
         pooled.iloc[:1].assign(fight_id=["ffffffffffffffff"]),
         pooled.iloc[2:]]
    ).reset_index(drop=True)
    out, cells = mea.attach_oof_joint(grown, dump)
    assert list(out["fight_id"]) == dump["fight_id"]
    by_id = dict(zip(dump["fight_id"], np.asarray(dump["joint_cells"], dtype=float)))
    for row, fight_id in enumerate(out["fight_id"]):
        assert cells[row] == pytest.approx(by_id[fight_id])


def test_attach_oof_joint_rejects_a_relabelled_method():
    pooled = _joint_pooled()
    dump = _joint_dump(pooled)
    dump["y_method"][0] = "submission" if dump["y_method"][0] != "submission" else "decision"
    with pytest.raises(ValueError, match="y_method"):
        mea.attach_oof_joint(pooled, dump)


def test_attach_oof_joint_rejects_a_dump_with_no_cells():
    pooled = _joint_pooled()
    dump = _joint_dump(pooled)
    del dump["joint_cells"]
    with pytest.raises(ValueError, match="joint_cells"):
        mea.attach_oof_joint(pooled, dump)


def test_attach_oof_joint_rejects_cells_that_are_not_a_distribution():
    pooled = _joint_pooled()
    dump = _joint_dump(pooled)
    dump["joint_cells"][0] = [v * 2 for v in dump["joint_cells"][0]]
    with pytest.raises(ValueError, match="sum to 1"):
        mea.attach_oof_joint(pooled, dump)
