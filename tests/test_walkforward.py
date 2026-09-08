import json

import numpy as np
import pandas as pd
import pytest

from mma.walkforward import FOLD_YEARS, make_folds, recency_weights
from mma.walkforward import bar_check, build_report, paired_delta, pool, score_rows
from mma.walkforward import slice_comparison, slice_masks

METHOD = ["ko_tko", "submission", "decision"]
ROUND = ["1", "2", "3", "45"]


def _dates():
    return pd.Series(pd.to_datetime([
        "2005-06-01", "2015-03-01", "2016-07-01", "2017-01-15", "2017-12-31",
        "2018-01-01", "2018-06-01", "2024-05-05", "2025-02-02", "2026-08-08",
    ]))


def test_fold_years_are_2018_to_2025():
    assert FOLD_YEARS == (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)


def test_fold_2018_masks():
    folds = {f.year: f for f in make_folds(_dates())}
    f = folds[2018]
    d = _dates()
    assert f.train.tolist() == (d < "2017-01-01").tolist()
    assert f.inner_val.tolist() == ((d >= "2017-01-01") & (d < "2018-01-01")).tolist()
    assert f.eval.tolist() == ((d >= "2018-01-01") & (d < "2019-01-01")).tolist()


def test_no_eval_row_in_train_or_inner_val_and_masks_disjoint():
    for f in make_folds(_dates()):
        assert not (f.train & f.eval).any()
        assert not (f.inner_val & f.eval).any()
        assert not (f.train & f.inner_val).any()


def test_last_fold_absorbs_partial_2026():
    folds = {f.year: f for f in make_folds(_dates())}
    d = _dates()
    assert folds[2025].eval.tolist() == (d >= "2025-01-01").tolist()
    assert folds[2025].eval.sum() == 2


def test_train_start_drops_old_fights_from_train_only():
    folds = {f.year: f for f in make_folds(_dates(), train_start="2010-01-01")}
    f = folds[2018]
    assert not f.train[0]  # 2005 row excluded from training
    assert f.train.sum() == 2  # 2015, 2016
    assert f.inner_val.sum() == 2


def test_eval_start_property():
    f = make_folds(_dates(), fold_years=(2019,))[0]
    assert f.eval_start == pd.Timestamp("2019-01-01")


def test_recency_weights_half_life():
    dates = pd.Series(pd.to_datetime(["2018-01-01", "2014-01-01", "2010-01-01"]))
    w = recency_weights(dates, reference=pd.Timestamp("2018-01-01"), half_life_years=4.0)
    assert w == pytest.approx([1.0, 0.5, 0.25], rel=1e-2)


def test_recency_weights_none_is_uniform():
    dates = pd.Series(pd.to_datetime(["2018-01-01", "2014-01-01"]))
    assert recency_weights(dates, pd.Timestamp("2018-01-01"), None).tolist() == [1.0, 1.0]


def _rows(n=8, seed=0):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame({
        "date": pd.to_datetime(["2018-03-01"] * 4 + ["2019-03-01"] * 4),
        "y_winner": rng.integers(0, 2, size=n),
        "y_method": pd.array(["ko_tko", "decision", "submission", "decision"] * 2, dtype="string"),
        "y_finish_round": pd.array(["1", None, "3", None] * 2, dtype="string"),
        "weight_class": pd.array(["Lightweight", "Women's Strawweight"] * 4, dtype="string"),
        "scheduled_rounds": pd.array([3, 5, 3, 3] * 2, dtype="Int64"),
        "debut_a": [True, False] * 4, "debut_b": [False] * 8,
    })
    pred = {
        "winner": rng.uniform(0.2, 0.8, size=n),
        "method": np.full((n, 3), 1 / 3),
        "round": np.full((n, 4), 0.25),
    }
    return feats, pred


def test_score_rows_reports_all_metrics():
    feats, pred = _rows()
    out = score_rows(feats, pred, METHOD, ROUND)
    for key in ("n", "winner_log_loss", "accuracy", "brier", "ece",
                "joint_log_loss", "method_macro_f1", "round_macro_f1", "n_method", "n_round"):
        assert key in out
    assert out["n"] == 8 and out["n_method"] == 8 and out["n_round"] == 4
    assert out["joint_log_loss"] is not None and out["method_macro_f1"] is not None


def test_score_rows_joint_matches_direct_computation():
    feats, pred = _rows()
    out = score_rows(feats, pred, METHOD, ROUND)
    # uniform heads: finishes cost -log(pw * 1/3 * 1/4), decisions -log(pw * 1/3)
    y = feats["y_winner"].to_numpy()
    pw = np.where(y == 1, pred["winner"], 1 - pred["winner"])
    is_finish = feats["y_finish_round"].notna().to_numpy()
    expected = -np.mean(np.log(pw / 3 / np.where(is_finish, 4, 1)))
    assert out["joint_log_loss"] == pytest.approx(expected, abs=1e-4)


def test_score_rows_without_method_head():
    feats, pred = _rows()
    out = score_rows(feats, {"winner": pred["winner"], "method": None, "round": None}, METHOD, ROUND)
    assert out["joint_log_loss"] is None and out["method_macro_f1"] is None and out["round_macro_f1"] is None


def test_slice_masks():
    feats, _ = _rows()
    masks = slice_masks(feats)
    assert set(masks) == {"debut", "womens", "five_round"}
    assert masks["debut"].sum() == 4
    assert masks["womens"].sum() == 4
    assert masks["five_round"].sum() == 2
    feats["external_missing"] = [True] + [False] * 7
    assert slice_masks(feats)["external_missing"].sum() == 1


def test_pool_concatenates_fold_predictions_in_row_order():
    feats, pred = _rows()
    a = np.array([True] * 4 + [False] * 4)
    b = ~a
    sub = lambda m: {k: (v[m] if v is not None else None) for k, v in pred.items()}  # noqa: E731
    pooled_feats, pooled_pred = pool(feats, [(a, sub(a)), (b, sub(b))])
    assert len(pooled_feats) == 8
    assert pooled_pred["winner"].tolist() == pred["winner"].tolist()
    assert pooled_pred["method"].shape == (8, 3)


def test_pool_without_heads():
    feats, pred = _rows()
    a = np.array([True] * 4 + [False] * 4)
    _, pooled = pool(feats, [(a, {"winner": pred["winner"][a], "method": None, "round": None}),
                             (~a, {"winner": pred["winner"][~a], "method": None, "round": None})])
    assert pooled["method"] is None and pooled["round"] is None


def test_bar_check_arithmetic():
    cand = {"pooled": {"winner_log_loss": 0.640}, "folds": {"2018": {"winner_log_loss": 0.65}, "2019": {"winner_log_loss": 0.63}}}
    inc = {"pooled": {"winner_log_loss": 0.650}, "folds": {"2018": {"winner_log_loss": 0.64}, "2019": {"winner_log_loss": 0.66}}}
    out = bar_check(cand, inc, sigma_seed=0.002)
    assert out["bar"] == pytest.approx(0.004)          # max(0.003, 2*sigma)
    assert out["delta"] == pytest.approx(-0.010)
    assert out["worst_fold_delta"] == pytest.approx(0.010)  # 2018 got worse by exactly 0.01, allowed
    assert out["clears_delta"] is True and out["no_fold_regression"] is True and out["ships"] is True
    inc["folds"]["2018"]["winner_log_loss"] = 0.635
    assert bar_check(cand, inc, 0.002)["ships"] is False


def test_bar_check_uses_min_bar_when_sigma_small():
    cand = {"pooled": {"winner_log_loss": 0.6475}, "folds": {}}
    inc = {"pooled": {"winner_log_loss": 0.650}, "folds": {}}
    out = bar_check(cand, inc, sigma_seed=0.0005)
    assert out["bar"] == pytest.approx(0.003)
    assert out["clears_delta"] is False  # delta -0.0025 does not beat -0.003
    # exactly on the bar: 0.650 - 0.0030 is not representable exactly, and the
    # strict comparison must not be decided by float noise
    cand["pooled"]["winner_log_loss"] = 0.650 - 0.0030
    out = bar_check(cand, inc, sigma_seed=0.0005)
    assert out["delta"] == pytest.approx(-0.003)
    assert out["clears_delta"] is False
    assert out["ships"] is False


def test_bar_check_not_comparable_when_fold_sets_differ():
    cand = {"pooled": {"winner_log_loss": 0.630, "n": 100},
            "folds": {"2018": {"winner_log_loss": 0.63}, "2019": {"winner_log_loss": 0.63}}}
    inc = {"pooled": {"winner_log_loss": 0.650, "n": 100},
           "folds": {"2018": {"winner_log_loss": 0.65}}}
    out = bar_check(cand, inc, sigma_seed=0.002)
    assert out["comparable"] is False
    assert out["missing_folds"] == ["2019"]
    assert out["clears_delta"] is True and out["no_fold_regression"] is True
    assert out["ships"] is False  # would clear on the numbers, but the reports do not line up
    # same fold years but different pooled n is also not comparable
    inc["folds"]["2019"] = {"winner_log_loss": 0.65}
    inc["pooled"]["n"] = 99
    out = bar_check(cand, inc, sigma_seed=0.002)
    assert out["comparable"] is False and out["missing_folds"] == [] and out["ships"] is False
    inc["pooled"]["n"] = 100
    out = bar_check(cand, inc, sigma_seed=0.002)
    assert out["comparable"] is True and out["ships"] is True


def test_bar_check_rejects_nan_sigma():
    cand = {"pooled": {"winner_log_loss": 0.630, "n": 10}, "folds": {}}
    inc = {"pooled": {"winner_log_loss": 0.650, "n": 10}, "folds": {}}
    with pytest.raises(ValueError, match="sigma_seed"):
        bar_check(cand, inc, sigma_seed=float("nan"))
    with pytest.raises(ValueError, match="sigma_seed"):
        bar_check(cand, inc, sigma_seed=None)


def test_paired_delta_arithmetic():
    a = {"pooled": {"winner_log_loss": 0.650}, "folds": {"2018": {"winner_log_loss": 0.64}, "2019": {"winner_log_loss": 0.66}}}
    b = {"pooled": {"winner_log_loss": 0.640}, "folds": {"2018": {"winner_log_loss": 0.65}, "2019": {"winner_log_loss": 0.63}}}
    out = paired_delta(a, b, sigma_seed=0.002)
    assert out["A_pooled"] is a["pooled"] and out["B_pooled"] is b["pooled"]
    assert out["delta_B_minus_A"] == pytest.approx(-0.010)
    assert out["per_fold_B_minus_A"] == {"2018": pytest.approx(0.01), "2019": pytest.approx(-0.03)}
    assert out["B_not_worse_than_A_by_sigma"] is True

    # B worse than A by more than sigma_seed -> gate fails
    b["pooled"]["winner_log_loss"] = 0.653
    out = paired_delta(a, b, sigma_seed=0.002)
    assert out["delta_B_minus_A"] == pytest.approx(0.003)
    assert out["B_not_worse_than_A_by_sigma"] is False


def test_paired_delta_folds_restricted_to_common_years():
    a = {"pooled": {"winner_log_loss": 0.5}, "folds": {"2018": {"winner_log_loss": 0.5}, "2019": {"winner_log_loss": 0.5}}}
    b = {"pooled": {"winner_log_loss": 0.5}, "folds": {"2019": {"winner_log_loss": 0.4}, "2020": {"winner_log_loss": 0.5}}}
    out = paired_delta(a, b, sigma_seed=0.002)
    assert out["per_fold_B_minus_A"] == {"2019": pytest.approx(-0.1)}


def test_slice_comparison_pairs_slices_and_flags_missing_ones():
    cand = {"slices": {
        "debut": {"n": 100, "winner_log_loss": 0.640},
        "external_missing": {"n": 20, "winner_log_loss": 0.700},
    }}
    inc = {"slices": {
        "debut": {"n": 100, "winner_log_loss": 0.650},
        "womens": {"n": 50, "winner_log_loss": 0.660},
    }}
    rows = {r["slice"]: r for r in slice_comparison(cand, inc)}
    assert set(rows) == {"debut", "external_missing", "womens"}

    shared = rows["debut"]
    assert shared["n"] == 100
    assert shared["candidate_winner_log_loss"] == pytest.approx(0.640)
    assert shared["incumbent_winner_log_loss"] == pytest.approx(0.650)
    assert shared["delta"] == pytest.approx(-0.010)

    # candidate-only slice: no incumbent number, so no delta
    only_cand = rows["external_missing"]
    assert only_cand["n"] == 20
    assert only_cand["candidate_winner_log_loss"] == pytest.approx(0.700)
    assert only_cand["incumbent_winner_log_loss"] is None
    assert only_cand["delta"] is None

    # incumbent-only slice is still reported, from the other side
    only_inc = rows["womens"]
    assert only_inc["n"] == 50
    assert only_inc["candidate_winner_log_loss"] is None
    assert only_inc["delta"] is None

    # candidate slices come first, in the candidate report's own order
    assert [r["slice"] for r in slice_comparison(cand, inc)] == [
        "debut", "external_missing", "womens"]


def test_slice_comparison_handles_reports_without_slices():
    assert slice_comparison({}, {}) == []


def test_build_report_shape():
    feats, pred = _rows()
    a = np.array([True] * 4 + [False] * 4)
    sub = lambda m: {k: (v[m] if v is not None else None) for k, v in pred.items()}  # noqa: E731
    report = build_report(
        name="toy", config={"k": 1}, features=feats,
        fold_results=[(2018, a, sub(a), {"best_iteration": 10}),
                      (2019, ~a, sub(~a), {"best_iteration": 20})],
        method_classes=METHOD, round_classes=ROUND,
    )
    assert set(report) >= {"name", "config", "fold_years", "folds", "pooled", "slices", "fit_info"}
    assert set(report["folds"]) == {"2018", "2019"}
    assert report["fold_years"] == [2018, 2019]
    assert report["fit_info"]["best_iteration"] == [10, 20]
    assert "debut" in report["slices"] and "womens" in report["slices"]
    assert report["pooled"]["n"] == 8


def test_build_report_casts_numpy_scalars_and_skips_empty_folds(capsys):
    feats, pred = _rows()
    a = np.array([True] * 4 + [False] * 4)
    empty = np.zeros(8, dtype=bool)
    sub = lambda m: {k: (v[m] if v is not None else None) for k, v in pred.items()}  # noqa: E731
    report = build_report(
        name="toy", config={"k": 1}, features=feats,
        fold_results=[(2018, a, sub(a), {"best_iteration": np.int64(10), "temperature": [np.float32(1.5)]}),
                      (2019, empty, sub(empty), {"best_iteration": np.int64(99), "temperature": [np.float32(2.0)]}),
                      (2020, ~a, sub(~a), {"best_iteration": np.int64(20), "temperature": [np.float32(1.0)]})],
        method_classes=METHOD, round_classes=ROUND,
    )
    json.dumps(report)  # numpy scalars would raise TypeError here
    assert set(report["folds"]) == {"2018", "2020"}
    assert report["fold_years"] == [2018, 2020]
    assert report["fit_info"]["best_iteration"] == [10, 20]
    assert all(type(v) is int for v in report["fit_info"]["best_iteration"])
    assert report["fit_info"]["temperature"] == [[1.5], [1.0]]
    assert report["pooled"]["n"] == 8
    assert "2019" in capsys.readouterr().out


from scripts.noise_floor import check_disjoint_seed_sets, sigma_from_reports


def test_sigma_from_reports_is_sample_std():
    reports = [{"pooled": {"winner_log_loss": v}} for v in (0.650, 0.652, 0.648)]
    out = sigma_from_reports(reports)
    assert out["sigma_seed"] == pytest.approx(np.std([0.650, 0.652, 0.648], ddof=1))
    assert out["bar"] == pytest.approx(max(0.003, 2 * out["sigma_seed"]))
    assert out["n_reports"] == 3


def test_noise_floor_rejects_overlapping_seed_sets():
    # disjoint sets pass silently
    check_disjoint_seed_sets(["0,1,2,3,4", "5,6,7,8,9", "10,11,12,13,14"])
    # a shared seed (4) between two sets raises
    with pytest.raises(SystemExit, match="4"):
        check_disjoint_seed_sets(["0,1,2,3,4", "4,5,6,7,8"])
    # null seed sets (non-torch reports) are skipped, not a failure
    check_disjoint_seed_sets(["0,1,2,3,4", None, "5,6,7,8,9"])


from scripts.run_walkforward import build_candidate, fixed_budget_from


def test_fixed_budget_from_per_head_xgb_medians():
    report = {"fit_info": {"best_iteration": [
        {"winner": 80, "method": 70, "round": 90},
        {"winner": 81, "method": 79, "round": 75},
        {"winner": 100, "method": 85, "round": 60},
    ]}}
    # medians 81, 79, 75 -- best_iteration is 0-based, so +1 tree count
    assert fixed_budget_from(report) == {"fixed_rounds": {"winner": 82, "method": 80, "round": 76}}


def test_fixed_budget_from_torch_per_fold_seed_lists():
    report = {"fit_info": {
        "best_epoch": [[7, 6], [26, 4], [19, 25]],
        "temperature": [[1.05, 1.10], [1.2, 1.0], [1.15, 1.1]],
    }}
    out = fixed_budget_from(report)
    # per-fold medians 6.5, 15, 22 -> median 15 -> +1 (best_epoch is 0-based)
    assert out["fixed_epochs"] == 16
    # per-fold medians 1.075, 1.1, 1.125 -> 1.1
    assert out["temperature"] == pytest.approx(1.1)
    assert "fixed_rounds" not in out


def test_fixed_budget_from_legacy_flat_best_iteration():
    report = {"fit_info": {"best_iteration": [50, 70, 90]}}
    # median 70 -- best_iteration is 0-based, so +1 tree count
    assert fixed_budget_from(report) == {"fixed_rounds": 71}


def test_build_candidate_rejects_mismatched_budget():
    torch_budget = {"fixed_epochs": 14, "temperature": 1.1}
    xgb_budget = {"fixed_rounds": {"winner": 81, "method": 79, "round": 75}}
    with pytest.raises(SystemExit, match="fixed_rounds"):
        build_candidate("xgb", "x", "0", {}, torch_budget)
    with pytest.raises(SystemExit, match="fixed_epochs"):
        build_candidate("torch", "t", "0", {}, xgb_budget)
    with pytest.raises(SystemExit, match="elo"):
        build_candidate("elo", "e", "0", {}, xgb_budget)
    # no --fixed-budget-from at all: every learner builds in early-stopping mode
    assert build_candidate("xgb", "x", "0", {}, None).fixed_rounds is None
    assert build_candidate("torch", "t", "0,1", {}, None).fixed_epochs is None
    # matching budgets are applied
    assert build_candidate("xgb", "x", "0", {}, xgb_budget).fixed_rounds == xgb_budget["fixed_rounds"]
    torch_cand = build_candidate("torch", "t", "0,1", {}, torch_budget)
    assert torch_cand.fixed_epochs == 14 and torch_cand.temperature == 1.1 and torch_cand.seeds == (0, 1)
