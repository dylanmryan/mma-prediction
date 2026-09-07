import numpy as np
import pandas as pd
import pytest

from mma.walkforward import FOLD_YEARS, Fold, make_folds, recency_weights


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
