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
# Kept in the feature table, held out of the model matrix: the `external`
# block's two fight-level flags. `external_missing` has to stay in the table
# for `mma.walkforward.slice_masks` to report its slice, but modelling it (or
# `same_country`, which is False whenever a nationality is unknown and so
# encodes the same coverage artifact) makes the block decay as the snapshot
# ages -- +0.0006 row-weighted on the 2024-2025 folds with the flags, -0.0006
# without. The twin exclusion for the torch path is `mma.tensors.DROPPED`,
# which carries the full argument; see also the SP2 plan's Task 11 shipping
# note (docs/superpowers/plans/2026-09-07-sp2-features-v3.md).
# `notice_unknown` and the `home_country_a`/`_b` pair join them for the SP2.1
# restoration: the first is the `notice` block's coverage flag (the source's
# bout list ends 2024-12-14, so modelling it is modelling the calendar) and
# the second FAILS the constant-vector leak check outright -- it is the
# `external` coverage-selection artifact in a second channel. See
# `mma.tensors.DROPPED`, which carries the full argument for all five.
MODEL_EXCLUDED = (
    "external_missing", "same_country",
    "notice_unknown", "home_country_a", "home_country_b",
)
NON_FEATURES = {"fight_id", "date", "swapped", *TARGETS, *MODEL_EXCLUDED}

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


def booster_categories(booster: xgb.Booster) -> dict:
    """{column: [category values]} for the categorical columns a model trained on.

    XGBoost 3.x matches categoricals BY VALUE (it stores the category strings
    in the model), which is why a served row does not have to reproduce the
    training frame's category ORDER. It does have to stay inside the training
    set: a value the model never saw is a hard XGBoostError, not a missing
    value. `align_to_booster` uses this list to make that impossible.
    """
    exported = booster.get_categories(export_to_arrow=True).to_arrow()
    return {
        name: [str(value) for value in values]
        for name, values in exported
        if values is not None
    }


def align_to_booster(x: pd.DataFrame, booster: xgb.Booster) -> pd.DataFrame:
    """A `feature_frame` output reshaped to exactly what one model expects.

    Two things a SERVED row needs that a training-table slice gets for free:

    * **The model's own column list.** The served row follows
      `feature_blocks.table_blocks()`, i.e. whatever the feature table on disk
      was built from; the committed model follows the blocks it was TRAINED on.
      Those are the same set after a redeploy and can differ while a feature
      experiment is running, so this takes the booster's list -- a missing
      column is a real contract break and raises, an extra one is simply not
      part of this model.
    * **A categorical that stays inside the training set.** `feature_frame`
      categorises `weight_class` from the values present, which for a one-row
      matchup is one value. Under XGBoost 3.x that is fine as long as the value
      was seen in training -- but a division the model never saw (a new weight
      class, a Wikipedia card that says "Catchweight") raises XGBoostError and
      would take down the weekly prospective run for the whole card. Rebuilding
      the column with the model's own categories turns that into a MISSING
      value instead, which is what the torch member already does with it
      (`mma.tensors.Preprocessor.transform` maps an unknown weight class to its
      reserved index 0).
    """
    trained = list(booster.feature_names or x.columns)
    missing = [column for column in trained if column not in x.columns]
    if missing:
        raise KeyError(
            f"the committed model needs feature(s) the served row does not "
            f"carry: {missing}; rebuild the feature table for the blocks the "
            "model was trained on, or retrain the model"
        )
    out = x[trained].copy()
    for column, categories in booster_categories(booster).items():
        if column not in out.columns:
            continue
        # Values outside the model's categories are mapped to None EXPLICITLY.
        # Handing them to pd.Categorical(..., categories=...) and letting it
        # coerce is deprecated in pandas and will raise, and the mapping is the
        # point rather than an implementation detail: an unseen weight class
        # has to become a missing value, not an error.
        known = set(categories)
        out[column] = pd.Categorical(
            [value if value in known else None for value in out[column].astype(object)],
            categories=categories,
        )
    return out


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
