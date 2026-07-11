"""XGBoost baselines: winner (binary), method (3-class), finish round (4-class)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

TARGETS = ("y_winner", "y_method", "y_finish_round")
NON_FEATURES = {"fight_id", "date", "swapped", *TARGETS}

BASE_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}
MAX_ROUNDS = 2000
EARLY_STOP = 50


def feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    x = features[[c for c in features.columns if c not in NON_FEATURES]].copy()
    x["weight_class"] = x["weight_class"].astype("category")
    for column in x.columns:
        if x[column].dtype == "bool" or str(x[column].dtype) == "boolean":
            x[column] = x[column].astype(int)
        elif str(x[column].dtype) in ("Int64", "Float64"):
            x[column] = x[column].astype(float)
    return x


def train_binary(x_train, y_train, x_val, y_val) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        **BASE_PARAMS,
        n_estimators=MAX_ROUNDS,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP,
        enable_categorical=True,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def train_multiclass(x_train, y_train, x_val, y_val, classes) -> xgb.XGBClassifier:
    mapping = {label: index for index, label in enumerate(classes)}
    model = xgb.XGBClassifier(
        **BASE_PARAMS,
        n_estimators=MAX_ROUNDS,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        early_stopping_rounds=EARLY_STOP,
        enable_categorical=True,
    )
    model.fit(
        x_train, y_train.map(mapping),
        eval_set=[(x_val, y_val.map(mapping))], verbose=False,
    )
    return model
