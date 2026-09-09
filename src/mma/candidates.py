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
from sklearn.isotonic import IsotonicRegression

from mma.blend import LOGIT_EPS, apply_temperature, blend_heads, logit, mask_round_45
from mma.hazard import (
    HAZARD_CLASSES, build_decision_rows, build_hazard_rows, mirror_corners,
)
from mma.models.net import MultiTaskNet
from mma.models.train_loop import (
    METHOD_CLASSES, ROUND_CLASSES, encode_targets, fit_temperature, predict, train_one,
)
from mma.models.xgb import feature_frame, train_binary, train_multiclass
from mma.simulator import simulate
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


# The blend's arithmetic -- the weighted average, the round-45 mask and the
# post-average temperature -- lives in `mma.blend`, so the SERVED blend
# (`mma.inference.BlendedPredictor`) computes the same number this candidate
# measured rather than a second implementation of the same three operations.
CALIBRATORS = ("temperature", "isotonic")


def fit_isotonic(p_val: np.ndarray, y_val: np.ndarray):
    """Isotonic post-average calibrator, fitted on inner-validation rows.

    SP2.2 Task 3's single bounded remediation of the ECE gate, and a
    POST-HOC variant: the pre-registration fixes the candidate's calibration
    step as "temperature-scaled after averaging", and this replaces that step
    rather than adding a candidate. The mechanism it tests is that a single
    scalar cannot fix a *shape* mismatch between two differently-calibrated
    probability streams, which a monotone piecewise-constant map can.

    It sees exactly the data `fit_temperature` sees -- the fold's
    inner-validation year -- and never an evaluation row. Outputs are clipped
    to `LOGIT_EPS` because isotonic regression happily predicts exactly 0 or
    1 on a pure bin, which log-loss reads as infinity; `out_of_bounds="clip"`
    holds evaluation probabilities outside the inner-val range at the end
    values rather than raising.
    """
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.asarray(p_val, dtype=float), np.asarray(y_val, dtype=float))

    def apply(p: np.ndarray) -> np.ndarray:
        return np.clip(model.predict(np.asarray(p, dtype=float)), LOGIT_EPS, 1.0 - LOGIT_EPS)

    return apply


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
    # "temperature" is the pre-registered calibration step and the default, so
    # nothing about B0/B1 changes. "isotonic" is SP2.2 Task 3's post-hoc
    # remediation of the ECE gate; it is a diagnostic, has no fresh-seed
    # confirmation of its own, and must not be read as a shipping form.
    calibrator: str = "temperature"
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
        if self.calibrator not in CALIBRATORS:
            raise ValueError(
                f"blend calibrator must be one of {CALIBRATORS} (got {self.calibrator!r})"
            )
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

        blended = blend_heads(xgb_pred, torch_pred, w)
        three_round = (features.loc[scored, "scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        blended["round"] = mask_round_45(blended["round"], three_round)

        is_val = fold.inner_val[scored]  # positions within `scored`
        is_eval = fold.eval[scored]
        temperature = 1.0
        winner = blended["winner"][is_eval]
        if self.calibrate:
            y_val = features.loc[fold.inner_val, "y_winner"].to_numpy(dtype=float)
            if self.calibrator == "temperature":
                temperature = fit_temperature(logit(blended["winner"][is_val]), y_val)
                winner = apply_temperature(winner, temperature)
            else:
                winner = fit_isotonic(blended["winner"][is_val], y_val)(winner)
        pred = {
            "winner": winner,
            "method": blended["method"][is_eval],
            "round": blended["round"][is_eval],
        }
        info = {
            "blend_weight": w,
            "temperature": float(temperature),  # 1.0 when the calibrator is not a temperature
            "calibrated": bool(self.calibrate),
            "n_train": int(fold.train.sum()),
            # Recorded only when it is NOT the pre-registered temperature, so a
            # report produced by the default path keeps the fit_info shape every
            # committed blend report already has.
            **({} if self.calibrator == "temperature" else {"calibrator": str(self.calibrator)}),
            "xgb": xgb_info,
            "torch": torch_info,
        }
        return pred, info


# --------------------------------------------------------------------------
# The round-by-round simulator as a harness candidate (SP3)
# --------------------------------------------------------------------------

#: Fixed by the SP3 plan before any run: 10,000 simulated fights per member
#: per orientation, and Laplace alpha = 1 over the cells a fight can reach.
#: Both are part of what a prediction IS, so both are reported in `info`.
HAZARD_N_RUNS = 10_000
HAZARD_ALPHA = 1.0
#: Rounds simulated for the 45 fights with no recorded `scheduled_rounds`.
#: `mma.hazard` keeps them (their rounds fought are known from the finish) and
#: leaves the column NA so the model sees it as missing; the simulator still
#: needs a bound, and 3 is the harness's standing default for a missing
#: scheduled-round value (`slice_masks`, `TorchCandidate`, `BlendCandidate`).
HAZARD_DEFAULT_ROUNDS = 3


def _round_frame(features: pd.DataFrame, n_rounds: np.ndarray) -> pd.DataFrame:
    """`features` repeated once per round, with a 1-based `round_no`.

    The prediction-time twin of `hazard.build_hazard_rows`' expansion, and
    deliberately column-for-column identical to it once the label is dropped:
    the two frames are concatenated into one model matrix, so a different
    column order here would be a silently mis-fed model.
    """
    counts = np.asarray(n_rounds, dtype=int)
    out = features.iloc[np.repeat(np.arange(len(features)), counts)].reset_index(drop=True)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    out["round_no"] = np.arange(len(out), dtype=int) - starts + 1
    return out


def _bucket_rounds(array: np.ndarray, n_classes: int = len(ROUND_CLASSES)) -> np.ndarray:
    """`(2, 2, R)` over rounds 1..R -> `(2, 2, n_classes)` over the harness's
    round classes, folding rounds 4 and up into the trailing '45' class."""
    head = n_classes - 1
    out = np.zeros(array.shape[:2] + (n_classes,), dtype=array.dtype)
    out[:, :, :min(array.shape[2], head)] = array[:, :, :head]
    if array.shape[2] > head:
        out[:, :, head] = array[:, :, head:].sum(axis=2)
    return out


@dataclass
class HazardCandidate:
    """The Monte Carlo fight simulator behind the standard `fit_predict`.

    Two XGBoost members are fitted on the fold's TRAINING fights only -- a
    5-class model over `mma.hazard.HAZARD_CLASSES` on the per-round hazard
    rows, and a binary model on the decision rows -- and every evaluation
    fight is then played out `n_runs` times by `mma.simulator.simulate`. The
    result is one joint distribution over outcome cells per fight, from which
    the winner, method and finish-round marginals are read off. They are
    returned in the existing `pred` shape so every metric the harness already
    computes keeps working, with the joint carried alongside as
    `joint_cells` so `walkforward.score_rows` can score the realised cell
    directly instead of composing three marginals it does not need to.

    **Seed ensembling** follows `XGBCandidate`: the members' `predict_proba`
    outputs are averaged across seeds and the simulation is run once on the
    averaged hazard, rather than simulating each seed and averaging outcome
    distributions. That is the same "average the probabilities" convention
    every other ensemble in this project uses.

    **Symmetrisation** follows `mma.inference.predict_symmetrized`: each
    evaluation fight is predicted from both corner orderings (the mirrored
    orientation via `hazard.mirror_corners`) and the two joint distributions
    are averaged after the mirrored one is mapped back -- which for a joint
    means exchanging its A and B blocks, not just flipping a scalar. Both
    orderings share one RNG stream, so a matchup with no corner asymmetry
    comes back at exactly 0.5 rather than 0.5 plus Monte Carlo noise
    (`tests/test_candidates.py` pins that).

    `fights` supplies `finish_round`, which the feature table only carries
    bucketed as '45'; the exact round is what makes the hazard rows'
    censoring correct, so it is required rather than approximated.
    """
    name: str = "hazard"
    fights: pd.DataFrame | None = None
    seeds: tuple = (0, 1, 2, 3, 4)
    params: dict = field(default_factory=dict)
    drop_columns: tuple = ()
    n_runs: int = HAZARD_N_RUNS
    alpha: float = HAZARD_ALPHA
    sim_seed: int = 0

    def _seed_params(self, seed: int) -> dict:
        if "random_state" in self.params:
            raise ValueError(
                "HazardCandidate(seeds=...) sets random_state per member, but params "
                f"already fixes random_state={self.params['random_state']!r}; pass one of them"
            )
        return {**self.params, "random_state": int(seed)}

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        if self.fights is None:
            raise ValueError(
                "HazardCandidate needs the fights table (it carries finish_round, which "
                "the feature table only has bucketed as '45'); pass fights=..."
            )
        train_feats = features.loc[fold.train].reset_index(drop=True)
        val_feats = features.loc[fold.inner_val].reset_index(drop=True)
        eval_feats = features.loc[fold.eval].reset_index(drop=True)
        eval_mirror = mirror_corners(eval_feats)

        haz_train = build_hazard_rows(train_feats, self.fights)
        haz_val = build_hazard_rows(val_feats, self.fights)
        dec_train = build_decision_rows(train_feats, self.fights)
        dec_val = build_decision_rows(val_feats, self.fights)
        _require_all_classes(haz_train["hazard_label"], HAZARD_CLASSES, "hazard", fold)

        n_rounds = (
            eval_feats["scheduled_rounds"].fillna(HAZARD_DEFAULT_ROUNDS).to_numpy(dtype=int)
        )
        n_eval = len(eval_feats)

        # One model matrix per member, built from the training, inner-val and
        # both evaluation orientations at once so the `weight_class` category
        # set and the column order are identical across all four.
        haz_frames = [haz_train.drop(columns=["hazard_label"]),
                      haz_val.drop(columns=["hazard_label"]),
                      _round_frame(eval_feats, n_rounds),
                      _round_frame(eval_mirror, n_rounds)]
        dec_frames = [dec_train.drop(columns=["decision_label"]),
                      dec_val.drop(columns=["decision_label"]),
                      eval_feats, eval_mirror]
        xh = _split_frames(feature_frame(pd.concat(haz_frames, ignore_index=True),
                                         self.drop_columns), haz_frames)
        xd = _split_frames(feature_frame(pd.concat(dec_frames, ignore_index=True),
                                         self.drop_columns), dec_frames)

        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        by_fight = None if w is None else dict(zip(features["fight_id"], w))

        def weights(rows):
            return None if by_fight is None else rows["fight_id"].map(by_fight).to_numpy(dtype=float)

        hazard_probs, mirror_probs, decision_probs, mirror_decision = [], [], [], []
        iterations = {"hazard": [], "decision": []}
        for seed in self.seeds:
            params = self._seed_params(seed)
            hz = train_multiclass(xh[0], haz_train["hazard_label"], xh[1], haz_val["hazard_label"],
                                  HAZARD_CLASSES, params=params, sample_weight=weights(haz_train))
            dc = train_binary(xd[0], dec_train["decision_label"], xd[1], dec_val["decision_label"],
                              params=params, sample_weight=weights(dec_train))
            hazard_probs.append(hz.predict_proba(xh[2]))
            mirror_probs.append(hz.predict_proba(xh[3]))
            decision_probs.append(dc.predict_proba(xd[2])[:, 1])
            mirror_decision.append(dc.predict_proba(xd[3])[:, 1])
            iterations["hazard"].append(int(hz.best_iteration))
            iterations["decision"].append(int(dc.best_iteration))

        hazard = _normalise_rows(_mean_over_seeds(hazard_probs))
        mirrored = _normalise_rows(_mean_over_seeds(mirror_probs))
        decision = _mean_over_seeds(decision_probs)
        mirror_dec = _mean_over_seeds(mirror_decision)

        n_classes = len(ROUND_CLASSES)
        finish = np.zeros((n_eval, 2, 2, n_classes))
        cards = np.zeros((n_eval, 2))
        finish_zero = np.zeros((n_eval, 2, 2, n_classes), dtype=bool)
        cards_zero = np.zeros((n_eval, 2), dtype=bool)
        starts = np.cumsum(n_rounds) - n_rounds
        errors, empty, valid = [], 0, 0
        for i in range(n_eval):
            rounds = int(n_rounds[i])
            window = slice(int(starts[i]), int(starts[i]) + rounds)
            runs = []
            for probs, p_cards in ((hazard[window], decision[i]), (mirrored[window], mirror_dec[i])):
                # A fresh generator per orientation from the SAME seed: the
                # mirrored run must consume an identical random stream, or a
                # self-mirroring matchup would not average to exactly 0.5.
                runs.append(simulate(probs, float(p_cards), rounds, self.n_runs,
                                     np.random.default_rng([self.sim_seed, i]), alpha=self.alpha))
            straight, flipped = runs
            # `flipped` is in the mirrored frame, where corner A is this
            # table's corner B, so its winner axis is reversed before it is
            # averaged in. Getting this backwards would invert half of every
            # method and round prediction.
            finish[i] = 0.5 * (_bucket_rounds(straight.finish_probs)
                               + _bucket_rounds(flipped.finish_probs)[::-1])
            cards[i] = 0.5 * (straight.decision_probs + flipped.decision_probs[::-1])
            raw = (_bucket_rounds(straight.finish_counts) == 0) & (
                _bucket_rounds(flipped.finish_counts)[::-1] == 0)
            finish_zero[i] = raw
            cards_zero[i] = (straight.decision_counts == 0) & (flipped.decision_counts[::-1] == 0)
            reachable = min(rounds, n_classes)
            empty += int(raw[:, :, :reachable].sum()) + int(cards_zero[i].sum())
            valid += 4 * reachable + 2
            errors.extend((straight.p_a_wins_standard_error, flipped.p_a_wins_standard_error))

        cells = np.concatenate([finish.reshape(n_eval, -1), cards], axis=1)
        p_a = finish[:, 0].sum(axis=(1, 2)) + cards[:, 0]
        p_b = finish[:, 1].sum(axis=(1, 2)) + cards[:, 1]
        total = p_a + p_b
        finish_mass = finish.sum(axis=(1, 2, 3))
        pred = {
            "winner": p_a / total,
            "method": np.stack([finish[:, :, 0, :].sum(axis=(1, 2)),
                                finish[:, :, 1, :].sum(axis=(1, 2)),
                                cards.sum(axis=1)], axis=1) / total[:, None],
            "round": finish.sum(axis=(1, 2)) / np.where(finish_mass > 0, finish_mass, 1.0)[:, None],
            "joint_cells": cells,
            "joint_zero_mass": np.concatenate(
                [finish_zero.reshape(n_eval, -1), cards_zero], axis=1),
        }
        info = {
            "hazard": {"best_iteration": _one_or_all(iterations["hazard"]),
                       "n_train": int(len(haz_train))},
            "decision": {"best_iteration": _one_or_all(iterations["decision"]),
                         "n_train": int(len(dec_train))},
            "n_train": int(fold.train.sum()),
            "n_runs": int(self.n_runs),
            "alpha": float(self.alpha),
            "mc_standard_error": round(float(np.mean(errors)), 6),
            "mc_standard_error_max": round(float(np.max(errors)), 6),
            "zero_mass_cell_fraction": round(empty / valid, 6),
        }
        return pred, info


def _split_frames(matrix: pd.DataFrame, frames: list) -> list:
    """Slice one concatenated model matrix back into its parts, in order."""
    bounds = np.cumsum([0] + [len(f) for f in frames])
    return [matrix.iloc[bounds[i]:bounds[i + 1]] for i in range(len(frames))]


def _mean_over_seeds(members: list) -> np.ndarray:
    """Row-wise mean across seeds; one member is returned untouched."""
    return np.asarray(members[0]) if len(members) == 1 else np.mean(members, axis=0)


def _normalise_rows(probs: np.ndarray) -> np.ndarray:
    """Rescale each row to sum to 1 -- averaging softmax rows across seeds
    leaves them a few ulps off, which `simulate` rejects outright."""
    return probs / probs.sum(axis=1, keepdims=True)


def _one_or_all(values: list):
    """The scalar for a one-member ensemble, the list otherwise -- the shape
    convention `XGBCandidate` established for `best_iteration`."""
    return values[0] if len(values) == 1 else list(values)
