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


def _mean_over_members(members: list[dict], head: str) -> np.ndarray:
    """Row-wise mean of one head across members; a single member is returned
    untouched so a one-member ensemble is bit-identical to that member."""
    if len(members) == 1:
        return members[0][head]
    return np.mean([m[head] for m in members], axis=0)


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
    # Seed ensemble (SP2.2): fit one model per `random_state` and average
    # `predict_proba` over them, on all three heads. `None` -- the default --
    # is the historical single fit at whatever `random_state` `params` (or
    # `mma.models.xgb.BASE_PARAMS`) carries, and every committed XGB report
    # was produced that way. A one-seed tuple is bit-identical to the single
    # fit at that seed, fit_info included, so `--seeds 0` reproduces those
    # reports rather than merely resembling them. XGBoost has no ensemble of
    # its own -- its stochasticity is subsample/colsample under one seed --
    # and SP2.1 measured the price of scoring a single fit at sigma 0.00087
    # to 0.00125, three times the torch ensemble's, which is why the blend
    # candidate below needs this.
    seeds: tuple | None = None

    def _head_rounds(self, head: str):
        if isinstance(self.fixed_rounds, dict):
            return self.fixed_rounds[head]
        return self.fixed_rounds

    def _seed_params(self, seed: int | None) -> dict:
        if seed is None:
            return self.params
        if "random_state" in self.params:
            raise ValueError(
                "XGBCandidate(seeds=...) sets random_state per member, but params "
                f"already fixes random_state={self.params['random_state']!r}; pass one of them"
            )
        return {**self.params, "random_state": int(seed)}

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
        ym = features["y_method"]
        known = ym.notna().to_numpy()
        _require_all_classes(ym[train & known], METHOD_CLASSES, "method", fold)
        yr = features["y_finish_round"]
        finish = yr.notna().to_numpy()
        _require_all_classes(yr[train & finish], ROUND_CLASSES, "finish_round", fold)
        ev = x[fold.eval]

        members, iterations = [], []
        for seed in (self.seeds if self.seeds is not None else (None,)):
            params = self._seed_params(seed)
            winner = train_binary(x[train], y[train], val_slice(x, fold.inner_val), val_slice(y, fold.inner_val),
                                  params=params, sample_weight=weights(train), fixed_rounds=winner_rounds)
            method = train_multiclass(x[train & known], ym[train & known],
                                      val_slice(x, fold.inner_val & known), val_slice(ym, fold.inner_val & known),
                                      METHOD_CLASSES, params=params, sample_weight=weights(train & known),
                                      fixed_rounds=method_rounds)
            rounds = train_multiclass(x[train & finish], yr[train & finish],
                                      val_slice(x, fold.inner_val & finish), val_slice(yr, fold.inner_val & finish),
                                      ROUND_CLASSES, params=params, sample_weight=weights(train & finish),
                                      fixed_rounds=round_rounds)
            members.append({
                "winner": winner.predict_proba(ev)[:, 1],
                "method": method.predict_proba(ev),
                "round": rounds.predict_proba(ev),
            })
            iterations.append({
                "winner": winner_rounds if fixed else int(winner.best_iteration),
                "method": method_rounds if fixed else int(method.best_iteration),
                "round": round_rounds if fixed else int(rounds.best_iteration),
            })

        pred = {head: _mean_over_members(members, head) for head in ("winner", "method", "round")}
        info = {
            "best_iteration": {
                # one member -> the scalar the single-fit path always wrote, so a
                # one-seed run reproduces a committed report's fit_info exactly.
                head: (iterations[0][head] if len(iterations) == 1
                       else [it[head] for it in iterations])
                for head in ("winner", "method", "round")
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


_LOGIT_EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(q / (1.0 - q))


def _mask_round_45(probs: np.ndarray, three_round: np.ndarray) -> np.ndarray:
    """Zero the '45' column for three-round fights and renormalise those rows.

    `MultiTaskNet.round_probs` does this inside the torch member (by masking
    the logit before the softmax) and the XGB member does not do it at all, so
    an average of the two puts mass on a round that cannot happen. Rows whose
    45 column is already exactly zero are left completely alone -- renormalising
    a row that already sums to one still perturbs its last bits, and a blend at
    weight 0.0 has to be bit-identical to the torch member.
    """
    out = np.array(probs, dtype=float, copy=True)
    changed = np.asarray(three_round, dtype=bool) & (out[:, 3] != 0.0)
    out[np.asarray(three_round, dtype=bool), 3] = 0.0
    if changed.any():
        out[changed] = out[changed] / out[changed].sum(axis=1, keepdims=True)
    return out


@dataclass
class BlendCandidate:
    """Fixed-weight average of an XGBoost and a torch ensemble, calibrated after.

    SP2.2's pre-registered candidate (docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md).
    Two things distinguish it from `scripts/blend_check.py`, SP2.1's post-hoc
    composer of two finished reports, and both are the point of the experiment:

    * **Both members are seed ensembles.** SP2.1 measured a single XGBoost
      fit's seed sd at 0.00087-0.00125, three times the torch ensemble's
      0.000346, so a blend containing one XGB fit inherits half of that.
    * **The winner head is temperature-scaled AFTER averaging**, on the fold's
      inner-validation year, because averaging two differently-calibrated
      probability streams is not itself calibrated (SP2.1's blend measured ECE
      0.0178 against the deployed model's 0.0088). The temperature is fitted on
      the inner-val rows only and never sees an evaluation row -- the members
      are fitted once on a widened eval mask (inner_val | eval) so both row
      sets are scored by the same fit, which changes nothing about training:
      every member trains on `fold.train` and early-stops on `fold.inner_val`
      exactly as it does on its own.

    `weight` is the weight on the XGB member; the torch member gets 1 - weight.
    It is FIXED by the pre-registration at 0.5 and is never fitted -- the
    model-v2 session found fitted stacking weights lose to a plain average, and
    fitting them on these same folds is the selection failure this project has
    been burned by before. The 0.3/0.7 cells exist as a flatness diagnostic.
    """
    name: str = "blend"
    seeds: tuple = (0, 1, 2, 3, 4)
    weight: float = 0.5  # on the XGB member
    calibrate: bool = True
    params: dict = field(default_factory=dict)  # XGB member's param override
    config: dict = field(default_factory=dict)  # torch member's config
    max_epochs: int = 200
    patience: int = 20
    # As XGBCandidate.drop_columns: both members hold these out of their
    # matrices and the columns stay in the table, so slices still report.
    drop_columns: tuple = ()

    def members(self):
        """The two members, each with the blend's seed list and drop-columns."""
        return (
            XGBCandidate(name="xgb", params=self.params, drop_columns=self.drop_columns,
                         seeds=tuple(self.seeds)),
            TorchCandidate(name="torch", seeds=tuple(self.seeds), config=self.config,
                           max_epochs=self.max_epochs, patience=self.patience,
                           drop_columns=self.drop_columns),
        )

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        w = float(self.weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"blend weight must be in [0, 1] (got {self.weight!r})")
        if (fold.inner_val & fold.eval).any():
            raise ValueError(
                f"fold {fold.year}: inner_val and eval overlap, so the post-average "
                "temperature would be fitted on evaluation rows"
            )
        scored = fold.inner_val | fold.eval
        wide = Fold(year=fold.year, train=fold.train, inner_val=fold.inner_val, eval=scored)

        xgb_member, torch_member = self.members()
        xgb_pred, xgb_info = xgb_member.fit_predict(features, wide, sample_weight)
        torch_pred, torch_info = torch_member.fit_predict(features, wide, sample_weight)

        blended = {
            head: w * np.asarray(xgb_pred[head], dtype=float)
            + (1.0 - w) * np.asarray(torch_pred[head], dtype=float)
            for head in ("winner", "method", "round")
        }
        three_round = (features.loc[scored, "scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        blended["round"] = _mask_round_45(blended["round"], three_round)

        is_val = fold.inner_val[scored]  # positions within `scored`
        is_eval = fold.eval[scored]
        temperature = 1.0
        if self.calibrate:
            temperature = fit_temperature(
                _logit(blended["winner"][is_val]),
                features.loc[fold.inner_val, "y_winner"].to_numpy(dtype=float),
            )
        winner = blended["winner"][is_eval]
        if self.calibrate:
            winner = 1.0 / (1.0 + np.exp(-_logit(winner) / temperature))
        pred = {
            "winner": winner,
            "method": blended["method"][is_eval],
            "round": blended["round"][is_eval],
        }
        info = {
            "blend_weight": w,
            "temperature": float(temperature),
            "calibrated": bool(self.calibrate),
            "n_train": int(fold.train.sum()),
            "xgb": xgb_info,
            "torch": torch_info,
        }
        return pred, info
