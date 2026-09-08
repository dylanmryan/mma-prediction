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

`features` must be in the same row order as the `dates` used to build the
fold; the CLI sorts by date and resets the index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

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
        winner = 1 / (1 + 10 ** (-diff / 400))
        return {"winner": winner, "method": None, "round": None}, {}


@dataclass
class XGBCandidate:
    name: str = "xgb"
    params: dict = field(default_factory=dict)
    # int applies to all three heads; a dict {"winner": n, "method": n, "round": n}
    # sets a per-head budget (see fixed_budget_from in scripts/run_walkforward.py).
    fixed_rounds: int | dict | None = None
    # Feature columns held out of the model matrix for this run only; the
    # columns stay in `features`, so the walk-forward slices still see them.
    drop_columns: tuple = ()

    def _head_rounds(self, head: str):
        if isinstance(self.fixed_rounds, dict):
            return self.fixed_rounds[head]
        return self.fixed_rounds

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        x = feature_frame(features, self.drop_columns)
        fixed = self.fixed_rounds is not None
        train = (fold.train | fold.inner_val) if fixed else fold.train
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        def weights(mask):
            return None if w is None else w[mask]

        def val_slice(frame, mask):
            return None if fixed else frame[mask]

        winner_rounds = self._head_rounds("winner")
        method_rounds = self._head_rounds("method")
        round_rounds = self._head_rounds("round")

        y = features["y_winner"]
        winner = train_binary(x[train], y[train], val_slice(x, fold.inner_val), val_slice(y, fold.inner_val),
                              params=self.params, sample_weight=weights(train), fixed_rounds=winner_rounds)

        ym = features["y_method"]
        known = ym.notna().to_numpy()
        _require_all_classes(ym[train & known], METHOD_CLASSES, "method", fold)
        method = train_multiclass(x[train & known], ym[train & known],
                                  val_slice(x, fold.inner_val & known), val_slice(ym, fold.inner_val & known),
                                  METHOD_CLASSES, params=self.params, sample_weight=weights(train & known),
                                  fixed_rounds=method_rounds)

        yr = features["y_finish_round"]
        finish = yr.notna().to_numpy()
        _require_all_classes(yr[train & finish], ROUND_CLASSES, "finish_round", fold)
        rounds = train_multiclass(x[train & finish], yr[train & finish],
                                  val_slice(x, fold.inner_val & finish), val_slice(yr, fold.inner_val & finish),
                                  ROUND_CLASSES, params=self.params, sample_weight=weights(train & finish),
                                  fixed_rounds=round_rounds)

        ev = x[fold.eval]
        pred = {
            "winner": winner.predict_proba(ev)[:, 1],
            "method": method.predict_proba(ev),
            "round": rounds.predict_proba(ev),
        }
        info = {
            "best_iteration": {
                "winner": winner_rounds if fixed else int(winner.best_iteration),
                "method": method_rounds if fixed else int(method.best_iteration),
                "round": round_rounds if fixed else int(rounds.best_iteration),
            },
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
    # As XGBCandidate.drop_columns: excluded from the tensor, kept in the table.
    drop_columns: tuple = ()

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        # Torch predictions vary slightly across intra-op thread counts, so
        # the harness pins to 1 thread before any torch op -- reports must
        # not depend on the environment they were produced in.
        torch.set_num_threads(1)
        fixed = self.fixed_epochs is not None
        train = (fold.train | fold.inner_val) if fixed else fold.train
        prep = Preprocessor.fit(features, train_mask=train, drop_columns=self.drop_columns)
        x, wc = prep.transform(features)
        targets = encode_targets(features)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        three_round = torch.tensor(
            (features.loc[fold.eval, "scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        )
        val_x, val_wc = (None, None) if fixed else (x[fold.inner_val], wc[fold.inner_val])
        val_targets = None if fixed else _slice_targets(targets, fold.inner_val)

        winners, methods, rounds = [], [], []
        info = {
            "best_epoch": [], "temperature": [], "epochs_run": [],
            "n_train": int(train.sum()), "torch_threads": torch.get_num_threads(),
        }
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
            info["temperature"].append(round(temperature, 2))
            info["epochs_run"].append(fit_info["epochs_run"])
        pred = {
            "winner": np.mean(winners, axis=0),
            "method": np.mean(methods, axis=0),
            "round": np.mean(rounds, axis=0),
        }
        return pred, info
