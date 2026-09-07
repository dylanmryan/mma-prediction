import numpy as np
import pandas as pd
import pytest

from mma.candidates import EloCandidate, TorchCandidate, XGBCandidate
from mma.walkforward import make_folds


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
    assert "best_iteration" in info and info["n_train"] == int(fold.train.sum())
    y = table.loc[fold.eval, "y_winner"].to_numpy()
    assert np.mean((pred["winner"] > 0.5) == (y == 1)) > 0.6


def test_xgb_candidate_fixed_budget_uses_inner_val_for_training(table, fold):
    cand = XGBCandidate(params={"max_depth": 2}, fixed_rounds=20)
    pred, info = cand.fit_predict(table, fold, None)
    assert info["best_iteration"] == 20 and info["n_train"] == int((fold.train | fold.inner_val).sum())


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
