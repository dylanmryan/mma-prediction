import numpy as np
import pandas as pd
import pytest

from mma.tensors import DROPPED, Preprocessor


def _features():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2", "f3"],
            "date": pd.to_datetime(["2019-01-01", "2020-01-01", "2022-01-01"]),
            "swapped": [False, True, False],
            "y_winner": [1, 0, 1],
            "y_method": ["ko_tko", None, "decision"],
            "y_finish_round": ["2", None, None],
            "weight_class": pd.array(["Lightweight", None, "Heavyweight"], dtype="string"),
            "title_fight": [False, True, False],
            "scheduled_rounds": pd.array([3, 5, 3], dtype="Int64"),
            "elo_diff": [50.0, None, -20.0],
            "reach_diff": [5.0, 2.0, None],
            "reach_missing_a": [False, False, True],
            "reach_missing_b": [False, False, False],
            "dob_missing_a": [False, False, False],
            "dob_missing_b": [False, False, False],
        }
    )


def test_missing_flags_dropped_and_ids_excluded():
    prep = Preprocessor.fit(_features(), train_mask=np.array([True, True, False]))
    assert set(DROPPED) & set(prep.numeric_columns) == set()
    for column in ("fight_id", "date", "swapped", "y_winner", "weight_class"):
        assert column not in prep.numeric_columns


def test_impute_and_standardize_fit_on_train_only():
    features = _features()
    prep = Preprocessor.fit(features, train_mask=np.array([True, True, False]))
    x, wc = prep.transform(features)
    elo = features["elo_diff"]
    # train rows: [50, NaN] -> median 50, mean after impute 50, std 0 -> guarded to 1
    column = prep.numeric_columns.index("elo_diff")
    assert x[0, column] == pytest.approx(0.0)   # (50-50)/1
    assert x[1, column] == pytest.approx(0.0)   # imputed to train median 50
    assert x[2, column] == pytest.approx(-70.0) # (-20-50)/1 -- unseen val row


def test_weight_class_indexing_unknown_to_zero():
    features = _features()
    prep = Preprocessor.fit(features, train_mask=np.array([True, True, False]))
    _, wc = prep.transform(features)
    assert wc[1] == 0                      # NaN -> unknown bucket
    assert wc[0] != wc[2] and wc[0] > 0    # two known classes, distinct


def test_round_trip_json(tmp_path):
    features = _features()
    prep = Preprocessor.fit(features, train_mask=np.array([True, True, False]))
    path = tmp_path / "preprocess.json"
    prep.save(path)
    loaded = Preprocessor.load(path)
    x1, wc1 = prep.transform(features)
    x2, wc2 = loaded.transform(features)
    assert np.allclose(x1, x2) and (wc1 == wc2).all()
