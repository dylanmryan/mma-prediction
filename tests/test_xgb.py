import numpy as np
import pandas as pd

from mma.models.xgb import feature_frame, train_binary


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "elo_diff": rng.normal(0, 100, n),
            "age_diff": rng.normal(0, 5, n),
            "weight_class": pd.array(["Lightweight"] * n, dtype="string"),
            "title_fight": [False] * n,
        }
    )
    y = (x["elo_diff"] + rng.normal(0, 50, n) > 0).astype(int)
    return x, y


def test_feature_frame_drops_ids_and_targets():
    features = pd.DataFrame(
        {
            "fight_id": ["f"], "date": [pd.Timestamp("2020-01-01")],
            "swapped": [True], "y_winner": [1], "y_method": ["decision"],
            "y_finish_round": [None], "elo_diff": [10.0],
            "weight_class": pd.array(["Lightweight"], dtype="string"),
        }
    )
    x = feature_frame(features)
    assert list(x.columns) == ["elo_diff", "weight_class"]
    assert str(x["weight_class"].dtype) == "category"


def test_binary_model_learns_signal():
    x, y = _synthetic()
    xf = feature_frame(pd.concat([x], axis=1).assign(fight_id="f", date=pd.Timestamp("2020-01-01"), swapped=False, y_winner=0, y_method=None, y_finish_round=None))
    model = train_binary(xf[:300], y[:300], xf[300:], y[300:])
    p = model.predict_proba(xf[300:])[:, 1]
    assert ((p >= 0.5).astype(int) == y[300:]).mean() > 0.7


def _toy_xgb(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((x["a"] + 0.3 * rng.normal(size=n) > 0).astype(int))
    return x, y


def test_train_binary_accepts_params_and_sample_weight():
    x, y = _toy_xgb()
    w = np.where(x["a"] > 0, 5.0, 1.0)
    model = train_binary(x[:150], y[:150], x[150:], y[150:],
                         params={"max_depth": 2}, sample_weight=w[:150])
    assert model.get_params()["max_depth"] == 2
    assert model.predict_proba(x[150:]).shape == (50, 2)


def test_train_binary_fixed_rounds_has_no_early_stopping():
    x, y = _toy_xgb()
    model = train_binary(x, y, None, None, fixed_rounds=37)
    assert model.get_booster().num_boosted_rounds() == 37


def test_train_multiclass_fixed_rounds_and_weights():
    from mma.models.xgb import train_multiclass
    x, y = _toy_xgb()
    # xgboost's sklearn wrapper requires every class to appear in y_train,
    # so sprinkle in the third label.
    labels = pd.Series(np.where(y == 1, "ko_tko", "decision"))
    labels[x["b"] > 1.0] = "submission"
    model = train_multiclass(x, labels, None, None, ["ko_tko", "submission", "decision"],
                             fixed_rounds=10, sample_weight=np.ones(len(x)))
    assert model.get_booster().num_boosted_rounds() == 10
    assert model.predict_proba(x).shape == (len(x), 3)
