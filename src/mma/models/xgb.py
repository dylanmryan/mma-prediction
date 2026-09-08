"""XGBoost baselines: winner (binary), method (3-class), finish round (4-class).

Both trainers early-stop on a validation set by default. They also accept a
``params`` override dict (merged over BASE_PARAMS), per-row ``sample_weight``
(e.g. recency weights), and ``fixed_rounds`` -- a fixed-budget mode that trains
exactly that many rounds with no validation set, which the walk-forward
refit-through-latest step needs.
"""
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
    "random_state": 0,
}
MAX_ROUNDS = 2000
EARLY_STOP = 50


def feature_frame(features: pd.DataFrame, drop_columns=()) -> pd.DataFrame:
    """The model matrix: every column that is not an identifier or a target.

    `drop_columns` names further columns to leave out *for this call only* --
    the ablation path behind `scripts/run_walkforward.py --drop-columns`.
    Columns excluded for good belong in `NON_FEATURES`, not here.
    """
    excluded = NON_FEATURES | set(drop_columns)
    x = features[[c for c in features.columns if c not in excluded]].copy()
    x["weight_class"] = x["weight_class"].astype("category")
    for column in x.columns:
        if x[column].dtype == "bool" or str(x[column].dtype) == "boolean":
            x[column] = x[column].astype(int)
        elif str(x[column].dtype) in ("Int64", "Float64"):
            x[column] = x[column].astype(float)
    return x


RESERVED_PARAMS = frozenset({
    "n_estimators", "objective", "early_stopping_rounds", "eval_metric",
    "num_class", "enable_categorical",
})


def _classifier(objective: str, params: dict | None, fixed_rounds: int | None, **extra):
    reserved = sorted(RESERVED_PARAMS & set(params or {}))
    if reserved:
        raise ValueError(
            f"params may not set reserved xgboost keys {reserved}; the trainer "
            "owns them (use fixed_rounds for the round budget)"
        )
    merged = {**BASE_PARAMS, **(params or {})}
    if fixed_rounds is not None:
        return xgb.XGBClassifier(
            **merged, n_estimators=fixed_rounds, objective=objective,
            enable_categorical=True, **extra,
        )
    return xgb.XGBClassifier(
        **merged, n_estimators=MAX_ROUNDS, objective=objective,
        early_stopping_rounds=EARLY_STOP, enable_categorical=True, **extra,
    )


def _fit_kwargs(sample_weight, eval_set):
    kwargs = {"verbose": False}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if eval_set is not None:
        kwargs["eval_set"] = eval_set
    return kwargs


def train_binary(x_train, y_train, x_val, y_val, *, params=None,
                 sample_weight=None, fixed_rounds=None) -> xgb.XGBClassifier:
    """Early-stop on (x_val, y_val); or, with fixed_rounds, train exactly that
    many rounds and ignore the validation set (pass None)."""
    model = _classifier("binary:logistic", params, fixed_rounds, eval_metric="logloss")
    eval_set = None if fixed_rounds is not None else [(x_val, y_val)]
    model.fit(x_train, y_train, **_fit_kwargs(sample_weight, eval_set))
    return model


def train_multiclass(x_train, y_train, x_val, y_val, classes, *, params=None,
                     sample_weight=None, fixed_rounds=None) -> xgb.XGBClassifier:
    """Multi-class twin of train_binary; ``classes`` fixes the label order."""
    mapping = {label: index for index, label in enumerate(classes)}
    model = _classifier("multi:softprob", params, fixed_rounds,
                        num_class=len(classes), eval_metric="mlogloss")
    eval_set = None if fixed_rounds is not None else [(x_val, y_val.map(mapping))]
    model.fit(x_train, y_train.map(mapping), **_fit_kwargs(sample_weight, eval_set))
    return model
