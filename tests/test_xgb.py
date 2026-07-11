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
