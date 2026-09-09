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
    round_frame,
)
from mma.joint import (
    compose_joint_cells, impose_winner_marginal, marginals_from_cells,
)
from mma.models.net import MultiTaskNet
from mma.models.train_loop import (
    METHOD_CLASSES, ROUND_CLASSES, encode_targets, fit_temperature, predict, train_one,
)
from mma.models.xgb import feature_frame, train_binary, train_multiclass
from mma.simulator import (
    DEFAULT_ALPHA, DEFAULT_N_RUNS, DEFAULT_ROUNDS, DEFAULT_SIM_SEED,
    mean_over_seeds, normalise_rows, simulate_fights,
)
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
    # D3 control (SP3 fallback branch): also emit this blend's own prediction
    # as a joint over outcome cells, composed as P(w) x P(m) x P(r | finish).
    # It changes NOTHING about the prediction -- the three heads are the same
    # arrays either way -- and only routes the joint metric through
    # `evaluate.joint_cell_log_loss` instead of `joint_outcome_log_loss`. That
    # is the point: it shows the cell machinery and the re-weighting layout
    # contribute nothing of their own, so the hybrid's joint gain has to come
    # from the simulator's conditional structure. Off by default, so every
    # committed blend report keeps its shape.
    emit_joint_cells: bool = False

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
        if self.emit_joint_cells:
            pred["joint_cells"] = compose_joint_cells(
                pred["winner"], pred["method"], pred["round"], METHOD_CLASSES, ROUND_CLASSES)
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

#: The simulation parameters the SP3 plan fixed before any run, re-exported
#: from `mma.simulator` where the served path reads them too. They are the
#: HARNESS defaults; the deployed values live in `models/simulator.json` and
#: are hashed, because both are part of what a prediction is.
HAZARD_N_RUNS = DEFAULT_N_RUNS
HAZARD_ALPHA = DEFAULT_ALPHA
HAZARD_DEFAULT_ROUNDS = DEFAULT_ROUNDS


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

    **Calibration (`calibrate=True`, SP3's fallback branch).** E2 measured the
    simulator's winner marginal at ECE 0.0286 against the blend's 0.0124 --
    the simulator has no calibration step at all, where every other candidate
    in this project has one. Turning `calibrate` on adds the one the blend
    uses: a temperature fitted on the fold's inner-validation year (never on
    an evaluation row) and applied to the winner marginal, then imposed back
    on the joint by `mma.joint.impose_winner_marginal` so the cells stay
    coherent with it. It is a DIAGNOSTIC that separates calibration from
    paradigm, and it is off by default.

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
    # D1 (SP3 fallback branch): temperature-scale the simulator's WINNER
    # marginal on the fold's inner-validation year, exactly as
    # `BlendCandidate` scales its post-average winner, and impose the result
    # back on the joint. Off by default, so `hazard_e1`/`hazard_e2` and every
    # other committed simulator report reproduce unchanged.
    calibrate: bool = False

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
        if self.calibrate and (fold.inner_val & fold.eval).any():
            raise ValueError(
                f"fold {fold.year}: inner_val and eval overlap, so the post-simulation "
                "temperature would be fitted on evaluation rows"
            )
        train_feats = features.loc[fold.train].reset_index(drop=True)
        val_feats = features.loc[fold.inner_val].reset_index(drop=True)
        eval_feats = features.loc[fold.eval].reset_index(drop=True)
        n_eval = len(eval_feats)
        # Calibrating needs the simulator's winner marginal on the
        # inner-validation year too, so those rows are simulated as well --
        # appended AFTER the evaluation rows, never before, so that every
        # evaluation fight keeps the per-row RNG stream (`sim_seed`, row
        # position) it had in the uncalibrated run. The temperature is then
        # the ONLY difference between this candidate and the plain simulator.
        sim_feats = (
            pd.concat([eval_feats, val_feats], ignore_index=True)
            if self.calibrate else eval_feats
        )
        sim_mirror = mirror_corners(sim_feats)

        haz_train = build_hazard_rows(train_feats, self.fights)
        haz_val = build_hazard_rows(val_feats, self.fights)
        dec_train = build_decision_rows(train_feats, self.fights)
        dec_val = build_decision_rows(val_feats, self.fights)
        _require_all_classes(haz_train["hazard_label"], HAZARD_CLASSES, "hazard", fold)

        n_rounds = (
            sim_feats["scheduled_rounds"].fillna(HAZARD_DEFAULT_ROUNDS).to_numpy(dtype=int)
        )
        n_sim = len(sim_feats)

        # One model matrix per member, built from the training, inner-val and
        # both evaluation orientations at once so the `weight_class` category
        # set and the column order are identical across all four.
        haz_frames = [haz_train.drop(columns=["hazard_label"]),
                      haz_val.drop(columns=["hazard_label"]),
                      round_frame(sim_feats, n_rounds),
                      round_frame(sim_mirror, n_rounds)]
        dec_frames = [dec_train.drop(columns=["decision_label"]),
                      dec_val.drop(columns=["decision_label"]),
                      sim_feats, sim_mirror]
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

        hazard = normalise_rows(mean_over_seeds(hazard_probs))
        mirrored = normalise_rows(mean_over_seeds(mirror_probs))
        decision = mean_over_seeds(decision_probs)
        mirror_dec = mean_over_seeds(mirror_decision)

        # The simulation itself, and the corner-averaging that composes the
        # two orientations into one joint, live in `mma.simulator` -- the
        # served predictor calls the SAME function, so the deployed hybrid
        # cannot drift from the one the harness measured.
        simulated = simulate_fights(
            hazard, mirrored, decision, mirror_dec, n_rounds,
            n_runs=self.n_runs, alpha=self.alpha, sim_seed=self.sim_seed,
            n_round_classes=len(ROUND_CLASSES),
        )
        cells, zero_mass = simulated["cells"], simulated["zero_mass"]
        # The diagnostics describe the SCORED rows only; with calibration on,
        # the inner-validation rows are simulated too and appended after them.
        errors = simulated["standard_errors"][:n_eval]
        empty = int(simulated["n_zero_mass_reachable"][:n_eval].sum())
        valid = int(simulated["n_reachable_cells"][:n_eval].sum())

        temperature = None
        if self.calibrate:
            # `BlendCandidate`'s calibration step, moved downstream of the
            # simulation: fit on the inner-validation year's winner marginal,
            # then impose the scaled probability back on the joint. Imposing
            # rather than merely reporting it is what keeps the cells coherent
            # -- each corner's block is scaled by one scalar, so
            # P(method, round | winner) is exactly the simulator's.
            simulated = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)["winner"]
            y_val = features.loc[fold.inner_val, "y_winner"].to_numpy(dtype=float)
            temperature = float(fit_temperature(logit(simulated[n_eval:]), y_val))
            cells, zero_mass = cells[:n_eval], zero_mass[:n_eval]
            cells = impose_winner_marginal(
                cells, apply_temperature(simulated[:n_eval], temperature),
                METHOD_CLASSES, ROUND_CLASSES,
            )

        marginals = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)
        pred = {
            "winner": marginals["winner"],
            "method": marginals["method"],
            "round": marginals["round"],
            "joint_cells": cells,
            "joint_zero_mass": zero_mass,
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
            # Recorded only when the fallback branch's calibration is on, so a
            # report from the default path keeps the fit_info shape the
            # committed simulator reports already have.
            **({} if temperature is None
               else {"temperature": round(temperature, 4), "calibrated": True}),
        }
        return pred, info


def _split_frames(matrix: pd.DataFrame, frames: list) -> list:
    """Slice one concatenated model matrix back into its parts, in order."""
    bounds = np.cumsum([0] + [len(f) for f in frames])
    return [matrix.iloc[bounds[i]:bounds[i + 1]] for i in range(len(frames))]


def _one_or_all(values: list):
    """The scalar for a one-member ensemble, the list otherwise -- the shape
    convention `XGBCandidate` established for `best_iteration`."""
    return values[0] if len(values) == 1 else list(values)


@dataclass
class HybridCandidate:
    """The v3 spec's defined fallback: the blend's winner, the simulator's shape.

    > "a documented hybrid where the direct winner model's probability is
    > imposed and the simulator supplies P(method, round | winner) by
    > re-weighting simulated runs."
    > -- docs/superpowers/specs/2026-09-06-predictor-v3-simulator-design.md, SP3

    This is a pre-registered branch, taken because E2 landed exactly on the
    case it was written for: the simulator's joint beat the incumbent's by
    0.0537 (five times the 0.01 bar) while its winner marginal lost by 0.0042
    (twelve times sigma_seed), and D1 showed that calibrating the simulator
    recovers only about half of that.

    The composition is arithmetic on two existing candidates' outputs, so
    both are reused whole rather than reimplemented:

        P_hybrid(winner, method, round) = P_blend(winner)
                                          x P_sim(method, round | winner)

    `mma.joint.impose_winner_marginal` does it by scaling each corner's block
    of simulated cells by a single scalar -- which is literally re-weighting
    the simulated runs by who won them, and leaves every conditional the
    simulation produced untouched.

    **The winner clause is satisfied by construction, not by measurement.**
    `pred["winner"]` is the blend member's array itself, so the hybrid's
    winner log-loss, accuracy, Brier and ECE are the incumbent's to the last
    bit. Only the joint is a new number.

    The hazard member deliberately runs UNCALIBRATED: imposing a winner
    marginal overwrites whatever winner the simulator had, and a scalar
    rescale cannot change a conditional, so calibrating it first would
    produce identical cells at twice the cost.
    """
    name: str = "hybrid"
    fights: pd.DataFrame | None = None
    seeds: tuple = (0, 1, 2, 3, 4)
    weight: float = 0.5  # the blend member's weight on ITS xgb member
    calibrate: bool = True  # the blend member's post-average temperature
    calibrator: str = "temperature"
    params: dict = field(default_factory=dict)  # XGB params, shared by both members
    config: dict = field(default_factory=dict)  # the blend's torch member config
    max_epochs: int = 200
    patience: int = 20
    drop_columns: tuple = ()
    n_runs: int = HAZARD_N_RUNS
    alpha: float = HAZARD_ALPHA
    sim_seed: int = 0

    def members(self):
        """The incumbent blend and the simulator, sharing seeds and columns."""
        return (
            BlendCandidate(name="blend", seeds=tuple(self.seeds), weight=self.weight,
                           calibrate=self.calibrate, calibrator=self.calibrator,
                           params=self.params, config=self.config,
                           max_epochs=self.max_epochs, patience=self.patience,
                           drop_columns=self.drop_columns),
            HazardCandidate(name="hazard", fights=self.fights, seeds=tuple(self.seeds),
                            params=self.params, drop_columns=self.drop_columns,
                            n_runs=self.n_runs, alpha=self.alpha, sim_seed=self.sim_seed,
                            calibrate=False),
        )

    def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight=None):
        blend_member, hazard_member = self.members()
        blend_pred, blend_info = blend_member.fit_predict(features, fold, sample_weight)
        hazard_pred, hazard_info = hazard_member.fit_predict(features, fold, sample_weight)

        cells = impose_winner_marginal(
            hazard_pred["joint_cells"], blend_pred["winner"], METHOD_CLASSES, ROUND_CLASSES)
        marginals = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)
        pred = {
            # The blend's own array, not a copy read back off the cells: the
            # winner clause is met by construction and every winner metric in
            # the report must be the incumbent's number exactly.
            "winner": blend_pred["winner"],
            "method": marginals["method"],
            "round": marginals["round"],
            "joint_cells": cells,
            # Re-weighting scales cells; it cannot make an empty one non-empty,
            # so the simulator's raw zero-mass flags carry over unchanged.
            "joint_zero_mass": hazard_pred["joint_zero_mass"],
        }
        info = {
            "n_train": int(fold.train.sum()),
            "blend": blend_info,
            "hazard": hazard_info,
        }
        return pred, info
