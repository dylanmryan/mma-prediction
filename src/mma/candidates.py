"""Candidate models behind one fit/predict protocol for the walk-forward harness.

Every candidate exposes
    fit_predict(features, fold, sample_weight) -> (pred, info)
where pred = {"winner": (n_eval,), "method": (n_eval, 3) | None,
"round": (n_eval, 4) | None} is row-aligned with features.loc[fold.eval]
and info carries fit diagnostics (best iteration / epoch, temperature,
n_train). sample_weight, when given, is aligned with the full feature
table; candidates slice it by their own training mask. Fixed-budget mode
(fixed_rounds / fixed_epochs) trains on fold.train | fold.inner_val with no
early stopping -- the "refit on the freshest year" strategy of spec §4 SP1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from mma.elo import expected_score
from mma.models.net import MultiTaskNet
from mma.models.train_loop import (
    METHOD_CLASSES, ROUND_CLASSES, encode_targets, fit_temperature, predict, train_one,
)
from mma.models.xgb import feature_frame, train_binary, train_multiclass
from mma.tensors import Preprocessor
from mma.walkforward import Fold


def _require_all_classes(y: pd.Series, classes, target: str, fold: Fold) -> None:
    present = set(y.dropna().unique())
    missing = [c for c in classes if c not in present]
    if missing:
        raise ValueError(
            f"fold {fold.year}: training rows lack {target} class(es) {missing}; "
            "xgboost cannot fit a multiclass model without every class present"
        )


def _slice_targets(targets: dict, mask: np.ndarray) -> dict:
    index = torch.tensor(mask)
    return {key: value[index] for key, value in targets.items()}


class EloCandidate:
    """Winner-only floor: the Elo expected score from the pre-fight rating diff."""
    name = "elo"

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        diff = np.nan_to_num(features.loc[fold.eval, "elo_diff"].to_numpy(dtype=float))
        winner = np.array([expected_score(d, 0.0) for d in diff])
        return {"winner": winner, "method": None, "round": None}, {}


@dataclass
class XGBCandidate:
    name: str = "xgb"
    params: dict = field(default_factory=dict)
    fixed_rounds: int | None = None

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        x = feature_frame(features)
        train = (fold.train | fold.inner_val) if self.fixed_rounds is not None else fold.train
        val = fold.inner_val
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        common = {"params": self.params, "fixed_rounds": self.fixed_rounds}

        def weights(mask):
            return None if w is None else w[mask]

        y = features["y_winner"]
        winner = train_binary(x[train], y[train], x[val], y[val],
                              sample_weight=weights(train), **common)

        ym = features["y_method"]
        known = ym.notna().to_numpy()
        _require_all_classes(ym[train & known], METHOD_CLASSES, "method", fold)
        method = train_multiclass(x[train & known], ym[train & known], x[val & known], ym[val & known],
                                  METHOD_CLASSES, sample_weight=weights(train & known), **common)

        yr = features["y_finish_round"]
        finish = yr.notna().to_numpy()
        _require_all_classes(yr[train & finish], ROUND_CLASSES, "finish_round", fold)
        rounds = train_multiclass(x[train & finish], yr[train & finish], x[val & finish], yr[val & finish],
                                  ROUND_CLASSES, sample_weight=weights(train & finish), **common)

        ev = x[fold.eval]
        pred = {
            "winner": winner.predict_proba(ev)[:, 1],
            "method": method.predict_proba(ev),
            "round": rounds.predict_proba(ev),
        }
        info = {
            "best_iteration": (self.fixed_rounds if self.fixed_rounds is not None
                               else int(winner.best_iteration)),
            "n_train": int(train.sum()),
        }
        return pred, info


@dataclass
class TorchCandidate:
    name: str = "torch"
    seeds: tuple = (0, 1, 2, 3, 4)
    config: dict = field(default_factory=dict)
    max_epochs: int = 200
    patience: int = 20
    fixed_epochs: int | None = None
    temperature: float | None = None  # fixed-epoch mode only (no inner val to fit on)

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        fixed = self.fixed_epochs is not None
        train = (fold.train | fold.inner_val) if fixed else fold.train
        prep = Preprocessor.fit(features, train_mask=train)
        x, wc = prep.transform(features)
        targets = encode_targets(features)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        three_round = torch.tensor(
            (features.loc[fold.eval, "scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        )
        val_x, val_wc = (None, None) if fixed else (x[fold.inner_val], wc[fold.inner_val])
        val_targets = None if fixed else _slice_targets(targets, fold.inner_val)

        winners, methods, rounds = [], [], []
        info = {"best_epoch": [], "temperature": [], "n_train": int(train.sum())}
        for seed in self.seeds:
            net, fit_info = train_one(
                seed, x[train], wc[train], _slice_targets(targets, train), val_x, val_wc, val_targets,
                max_epochs=self.max_epochs, patience=self.patience,
                n_weight_classes=prep.n_weight_classes, config=self.config,
                sample_weight=None if w is None else w[train], fixed_epochs=self.fixed_epochs,
            )
            if fixed:
                temperature = float(self.temperature or 1.0)
            else:
                raw = predict(net, val_x, val_wc)
                temperature = fit_temperature(raw["winner_logits"], val_targets["y_winner"].numpy())
            out = predict(net, x[fold.eval], wc[fold.eval], temperature=temperature)
            winners.append(out["winner"])
            methods.append(out["method"])
            rounds.append(
                MultiTaskNet.round_probs(torch.tensor(out["round_logits"]), three_round).numpy()
            )
            info["best_epoch"].append(fit_info["best_epoch"])
            info["temperature"].append(temperature)
        pred = {
            "winner": np.mean(winners, axis=0),
            "method": np.mean(methods, axis=0),
            "round": np.mean(rounds, axis=0),
        }
        return pred, info
