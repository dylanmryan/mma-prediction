import numpy as np
import pandas as pd
import pytest

from mma.candidates import BlendCandidate, EloCandidate, TorchCandidate, XGBCandidate
from mma.walkforward import Fold, make_folds


def _table(n=600, seed=0):
    """Tiny feature table shaped like features.parquet: dates 2014-2019, a
    signal column, and the identifier/target columns the candidates need."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2014-01-01") + pd.to_timedelta(rng.integers(0, 6 * 365, size=n), unit="D")
    elo_diff = rng.normal(0, 120, size=n)
    p = 1 / (1 + 10 ** (-elo_diff / 400))
    y = (rng.uniform(size=n) < p).astype(int)
    methods = rng.choice(["ko_tko", "submission", "decision"], size=n)
    rounds = np.where(methods == "decision", None, rng.choice(["1", "2", "3", "45"], size=n))
    return pd.DataFrame({
        "fight_id": [f"f{i}" for i in range(n)], "date": dates, "swapped": False,
        "y_winner": y, "y_method": pd.array(methods, dtype="string"),
        "y_finish_round": pd.array(rounds, dtype="string"),
        "weight_class": pd.array(rng.choice(["Lightweight", "Women's Strawweight"], size=n), dtype="string"),
        "title_fight": False, "scheduled_rounds": pd.array(rng.choice([3, 5], size=n), dtype="Int64"),
        "elo_diff": elo_diff, "noise": rng.normal(size=n),
        "debut_a": False, "debut_b": False,
    }).sort_values("date", kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def table():
    return _table()


@pytest.fixture(scope="module")
def fold(table):
    return make_folds(table["date"], fold_years=(2019,))[0]


def test_elo_candidate_is_expected_score(table, fold):
    pred, info = EloCandidate().fit_predict(table, fold, None)
    diff = table.loc[fold.eval, "elo_diff"].to_numpy()
    assert pred["winner"] == pytest.approx(1 / (1 + 10 ** (-diff / 400)))
    assert pred["method"] is None and pred["round"] is None


def test_xgb_candidate_shapes_and_signal(table, fold):
    pred, info = XGBCandidate(params={"max_depth": 2}).fit_predict(table, fold, None)
    n = int(fold.eval.sum())
    assert pred["winner"].shape == (n,) and pred["method"].shape == (n, 3) and pred["round"].shape == (n, 4)
    assert set(info["best_iteration"]) == {"winner", "method", "round"}
    assert all(isinstance(v, int) for v in info["best_iteration"].values())
    assert info["n_train"] == int(fold.train.sum())
    y = table.loc[fold.eval, "y_winner"].to_numpy()
    assert np.mean((pred["winner"] > 0.5) == (y == 1)) > 0.6


def test_xgb_candidate_fixed_budget_uses_inner_val_for_training(table, fold):
    cand = XGBCandidate(params={"max_depth": 2}, fixed_rounds=20)
    pred, info = cand.fit_predict(table, fold, None)
    assert info["best_iteration"] == {"winner": 20, "method": 20, "round": 20}
    assert info["n_train"] == int((fold.train | fold.inner_val).sum())


def test_xgb_candidate_per_head_fixed_rounds(table, fold):
    cand = XGBCandidate(params={"max_depth": 2},
                        fixed_rounds={"winner": 15, "method": 12, "round": 8})
    pred, info = cand.fit_predict(table, fold, None)
    n = int(fold.eval.sum())
    assert pred["winner"].shape == (n,) and pred["method"].shape == (n, 3) and pred["round"].shape == (n, 4)
    assert info["best_iteration"] == {"winner": 15, "method": 12, "round": 8}
    assert info["n_train"] == int((fold.train | fold.inner_val).sum())


def test_xgb_candidate_rejects_fold_missing_a_class(table, fold):
    broken = table.copy()
    broken.loc[fold.train, "y_method"] = "decision"
    with pytest.raises(ValueError, match="method"):
        XGBCandidate(params={"max_depth": 2}).fit_predict(broken, fold, None)


def test_torch_candidate_shapes_and_determinism(table, fold):
    cand = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=3)
    pred_a, info_a = cand.fit_predict(table, fold, None)
    pred_b, _ = cand.fit_predict(table, fold, None)
    n = int(fold.eval.sum())
    assert pred_a["winner"].shape == (n,) and pred_a["round"].shape == (n, 4)
    assert np.array_equal(pred_a["winner"], pred_b["winner"])
    assert len(info_a["temperature"]) == 1 and len(info_a["best_epoch"]) == 1


def test_torch_candidate_masks_round_45_for_three_round_fights(table, fold):
    pred, _ = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=2).fit_predict(table, fold, None)
    three = (table.loc[fold.eval, "scheduled_rounds"].fillna(3) <= 3).to_numpy()
    assert np.all(pred["round"][three, 3] < 1e-6)
    assert np.allclose(pred["round"].sum(axis=1), 1.0)


def test_torch_candidate_pins_threads(table, fold):
    import torch

    cand = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=3)
    pred_a, info_a = cand.fit_predict(table, fold, None)
    torch.set_num_threads(4)
    pred_b, info_b = cand.fit_predict(table, fold, None)
    assert np.array_equal(pred_a["winner"], pred_b["winner"])
    assert info_a["torch_threads"] == 1 and info_b["torch_threads"] == 1


def test_torch_candidate_fixed_epochs(table, fold):
    cand = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, fixed_epochs=2, temperature=1.2)
    pred, info = cand.fit_predict(table, fold, None)
    assert info["best_epoch"] == [1] and info["temperature"] == [1.2]
    assert info["n_train"] == int((fold.train | fold.inner_val).sum())


def test_sample_weight_reaches_both_learners(table, fold):
    w = np.where(table["noise"] > 0, 3.0, 1.0)
    a, _ = XGBCandidate(params={"max_depth": 2}).fit_predict(table, fold, None)
    b, _ = XGBCandidate(params={"max_depth": 2}).fit_predict(table, fold, w)
    assert not np.allclose(a["winner"], b["winner"])
    ta, _ = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=2).fit_predict(table, fold, None)
    tb, _ = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=2).fit_predict(table, fold, w)
    assert not np.allclose(ta["winner"], tb["winner"])


# --- XGBoost seed ensembling (SP2.2) ----------------------------------------


def test_xgb_seed_ensemble_of_one_matches_the_single_fit(table, fold):
    """seeds=(0,) must be bit-identical to the single-fit path at random_state 0.

    This is the guard that the ensemble did not change what a plain
    `--candidate xgb` run means: every committed XGB report was produced by
    the seeds=None path at BASE_PARAMS' random_state 0.
    """
    single, single_info = XGBCandidate(params={"max_depth": 2}).fit_predict(table, fold, None)
    ensembled, ens_info = XGBCandidate(params={"max_depth": 2}, seeds=(0,)).fit_predict(table, fold, None)
    assert np.array_equal(single["winner"], ensembled["winner"])
    assert np.array_equal(single["method"], ensembled["method"])
    assert np.array_equal(single["round"], ensembled["round"])
    # fit_info too: a one-seed ensemble reports scalars, not one-element lists,
    # so a re-run reproduces a committed report's fit_info as well.
    assert single_info == ens_info


def test_xgb_seed_ensemble_averages_predict_proba_across_seeds(table, fold):
    members = [
        XGBCandidate(params={"max_depth": 2, "random_state": s}).fit_predict(table, fold, None)[0]
        for s in (0, 1, 2)
    ]
    blend, info = XGBCandidate(params={"max_depth": 2}, seeds=(0, 1, 2)).fit_predict(table, fold, None)
    for head in ("winner", "method", "round"):
        assert blend[head] == pytest.approx(np.mean([m[head] for m in members], axis=0))
    assert not np.allclose(blend["winner"], members[0]["winner"])
    assert all(len(v) == 3 for v in info["best_iteration"].values())


def test_xgb_seed_ensemble_rejects_a_random_state_in_params(table, fold):
    with pytest.raises(ValueError, match="random_state"):
        XGBCandidate(params={"max_depth": 2, "random_state": 7}, seeds=(0, 1)).fit_predict(table, fold, None)


# --- BlendCandidate (SP2.2) --------------------------------------------------

BLEND_KWARGS = dict(seeds=(0,), params={"max_depth": 2}, config={"hidden": (16, 8)}, max_epochs=3)


def test_blend_candidate_shapes_and_row_alignment(table, fold):
    pred, info = BlendCandidate(**BLEND_KWARGS).fit_predict(table, fold, None)
    n = int(fold.eval.sum())
    assert pred["winner"].shape == (n,) and pred["method"].shape == (n, 3) and pred["round"].shape == (n, 4)
    assert np.all((pred["winner"] > 0) & (pred["winner"] < 1))
    assert pred["method"].sum(axis=1) == pytest.approx(1.0)
    assert pred["round"].sum(axis=1) == pytest.approx(1.0)
    assert info["blend_weight"] == 0.5
    assert info["n_train"] == int(fold.train.sum())
    assert set(info) >= {"xgb", "torch", "blend_weight", "temperature", "calibrated"}
    assert "best_iteration" in info["xgb"] and "best_epoch" in info["torch"]


def test_blend_candidate_is_deterministic(table, fold):
    cand = BlendCandidate(**BLEND_KWARGS)
    a, info_a = cand.fit_predict(table, fold, None)
    b, info_b = cand.fit_predict(table, fold, None)
    for head in ("winner", "method", "round"):
        assert np.array_equal(a[head], b[head])
    assert info_a == info_b


def _widened(fold):
    """The fold the blend's members actually see: eval widened to inner_val |
    eval, so one fit scores both the calibration rows and the evaluation rows.
    Training and early stopping are untouched (train / inner_val are the same
    masks), which is what makes the widening free."""
    return Fold(year=fold.year, train=fold.train, inner_val=fold.inner_val,
                eval=fold.inner_val | fold.eval), fold.eval[fold.inner_val | fold.eval]


def test_blend_weight_one_reproduces_the_xgb_member(table, fold):
    """The strongest correctness check: a degenerate weight must collapse the
    blend onto that member exactly (calibration off, which is the only other
    thing the blend does to the winner head)."""
    wide, is_eval = _widened(fold)
    blend, _ = BlendCandidate(weight=1.0, calibrate=False, **BLEND_KWARGS).fit_predict(table, fold, None)
    xgb, _ = XGBCandidate(params={"max_depth": 2}, seeds=(0,)).fit_predict(table, wide, None)
    assert np.array_equal(blend["winner"], xgb["winner"][is_eval].astype(float))
    assert np.array_equal(blend["method"], xgb["method"][is_eval].astype(float))
    # The round head is the one place the blend does more than average: the
    # 45 column is masked for three-round fights, which the XGB head alone
    # does not do. Everything else about it is the member's.
    three = (table.loc[fold.eval, "scheduled_rounds"].fillna(3) <= 3).to_numpy()
    assert np.all(blend["round"][three, 3] == 0.0)
    assert blend["round"][~three] == pytest.approx(xgb["round"][is_eval][~three])


def test_blend_weight_zero_reproduces_the_torch_member(table, fold):
    wide, is_eval = _widened(fold)
    blend, _ = BlendCandidate(weight=0.0, calibrate=False, **BLEND_KWARGS).fit_predict(table, fold, None)
    torch_pred, _ = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=3).fit_predict(table, wide, None)
    for head in ("winner", "method", "round"):
        # the torch member predicts in float32 and the blend averages in
        # float64; the widening is exact, so equality still has to be exact.
        assert np.array_equal(blend[head], np.asarray(torch_pred[head], dtype=float)[is_eval])
        assert blend[head].dtype == np.float64


def test_widening_the_eval_mask_does_not_move_the_members(table, fold):
    """The members are fitted once on `inner_val | eval` rather than twice, so
    the widening must not itself change what they predict on the eval rows.

    XGBoost is exactly batch-invariant. The torch member is not, quite: its
    float32 matmuls block differently at 339 rows than at 113, which moves a
    handful of probabilities by one float32 ulp (~6e-8) -- five orders of
    magnitude below the 4-dp log-loss the reports carry, and the reason the
    degenerate-weight tests above compare against the widened fold.
    """
    wide, is_eval = _widened(fold)
    for member in (XGBCandidate(params={"max_depth": 2}, seeds=(0,)),
                   TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=3)):
        plain, _ = member.fit_predict(table, fold, None)
        widened, _ = member.fit_predict(table, wide, None)
        for head in ("winner", "method", "round"):
            moved = np.abs(np.asarray(plain[head], dtype=float)
                           - np.asarray(widened[head], dtype=float)[is_eval])
            assert moved.max() < 1e-6


def test_blend_temperature_is_fitted_on_inner_val_and_never_on_eval(table, fold, monkeypatch):
    import mma.candidates as candidates

    seen = {}
    real = candidates.fit_temperature

    def spy(logits, y):
        seen["n"] = len(logits)
        seen["y"] = np.asarray(y).copy()
        return real(logits, y)

    monkeypatch.setattr(candidates, "fit_temperature", spy)
    pred, info = BlendCandidate(**BLEND_KWARGS).fit_predict(table, fold, None)
    assert seen["n"] == int(fold.inner_val.sum())
    assert np.array_equal(seen["y"], table.loc[fold.inner_val, "y_winner"].to_numpy(dtype=float))
    assert 0.5 <= info["temperature"] <= 3.0


def test_blend_predictions_ignore_the_evaluation_labels(table, fold):
    """Nothing the blend fits -- neither member nor the temperature -- may
    read a target on the evaluation rows."""
    scrambled = table.copy()
    scrambled.loc[fold.eval, "y_winner"] = 1 - scrambled.loc[fold.eval, "y_winner"]
    a, info_a = BlendCandidate(**BLEND_KWARGS).fit_predict(table, fold, None)
    b, info_b = BlendCandidate(**BLEND_KWARGS).fit_predict(scrambled, fold, None)
    assert np.array_equal(a["winner"], b["winner"])
    assert info_a["temperature"] == info_b["temperature"]


def test_blend_calibration_rescales_the_winner_head(table, fold):
    raw, raw_info = BlendCandidate(calibrate=False, **BLEND_KWARGS).fit_predict(table, fold, None)
    cal, cal_info = BlendCandidate(calibrate=True, **BLEND_KWARGS).fit_predict(table, fold, None)
    t = cal_info["temperature"]
    assert raw_info["temperature"] == 1.0 and raw_info["calibrated"] is False
    logits = np.log(np.clip(raw["winner"], 1e-9, 1 - 1e-9) / (1 - np.clip(raw["winner"], 1e-9, 1 - 1e-9)))
    assert cal["winner"] == pytest.approx(1 / (1 + np.exp(-logits / t)))
    # the method/round heads are untouched by the winner temperature
    assert np.array_equal(raw["method"], cal["method"])


def test_blend_masks_round_45_for_three_round_fights(table, fold):
    pred, _ = BlendCandidate(**BLEND_KWARGS).fit_predict(table, fold, None)
    three = (table.loc[fold.eval, "scheduled_rounds"].fillna(3) <= 3).to_numpy()
    assert np.all(pred["round"][three, 3] == 0.0)
    assert pred["round"].sum(axis=1) == pytest.approx(1.0)


def test_blend_rejects_a_weight_outside_the_unit_interval(table, fold):
    with pytest.raises(ValueError, match="blend weight"):
        BlendCandidate(weight=1.5, **BLEND_KWARGS).fit_predict(table, fold, None)
