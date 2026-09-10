"""Load the committed scorer and predict hypothetical matchups.

Since SP3 the deployed scorer is the HYBRID (`SimulatorPredictor`): the blend
supplies P(A wins) and the Monte Carlo fight simulator supplies
P(method, round | winner), composed into one joint distribution over outcome
cells. `BlendedPredictor` is still here and is still where the winner
probability comes from -- `SimulatorPredictor` holds one and returns its
`winner_prob` untouched -- but the method and finish-round heads it carries
are no longer what the app or the prediction records show.

Since SP2.2 that blend is itself the equal-weight average of the five-seed
XGBoost ensemble and the five-seed torch ensemble, temperature-scaled after
averaging; `Ensemble` is the torch member alone.

The three predictors nest, and each one's `predict` returns a superset of the
one below it, so every caller written against `Ensemble.predict`'s contract
keeps working:

    Ensemble  ->  BlendedPredictor  ->  SimulatorPredictor
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

from mma import context as fight_context_module
from mma import external, glicko, notice, serving
from mma.blend import apply_temperature, blend_heads, mask_round_45
from mma.feature_blocks import (
    CONTEXT_BLOCK, EXTERNAL_BLOCK, NOTICE_BLOCK, TRAJECTORY_BLOCK,
    resolve_blocks, state_key_blocks, state_keys, table_blocks,
)
from mma.hazard import mirror_corners, round_frame
from mma.joint import impose_winner_marginal, marginals_from_cells, swap_corners
from mma.models.net import MultiTaskNet
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
from mma.models.xgb import align_to_booster, feature_frame
from mma.simulator import mean_over_seeds, normalise_rows, simulate_fights
from mma.tensors import Preprocessor

ROOT = Path(__file__).resolve().parents[2]

# The original split-protocol training cutoff (scripts/train_xgb.py and
# scripts/train_torch.py in split mode): train on date < TRAIN_END.
TRAIN_END = "2021-01-01"
# The deployed torch ensemble's metrics file, which records how it was trained.
TORCH_METRICS = ROOT / "models" / "torch" / "metrics_val.json"

# --- the deployed blend (SP2.2) --------------------------------------------
# The walk-forward report the deployed scorer's evidence comes from: B1, the
# candidate the pre-registration's rules shipped. Its `config` records the
# weight and its `fit_info` the per-fold temperatures that
# `scripts/build_blend_config.py` derives BLEND_CONFIG from.
BLEND_REPORT = ROOT / "models" / "walkforward" / "blend_b1.json"
# The committed weight and post-average temperature `BlendedPredictor.load`
# combines the two members with (`scripts/build_blend_config.py` writes it
# from BLEND_REPORT). Both numbers move every recorded probability exactly as
# a retrained booster does, so they are hashed by
# `mma.versioning.MODEL_ARTIFACT_GLOBS` alongside the model weights -- there
# used to be module constants here instead (BLEND_WEIGHT / BLEND_TEMPERATURE),
# which the hash never covered: editing one silently changed every recorded
# probability while leaving `model_version` byte-identical. There is
# deliberately no in-code fallback value for a missing artifact -- silently
# serving a guessed weight or temperature is a milder version of the same
# failure the artifact-hash design exists to prevent.
BLEND_CONFIG = ROOT / "models" / "blend.json"


def load_blend_config(path: Path = BLEND_CONFIG) -> dict:
    """The committed {"weight": ..., "temperature": ...} the deployed blend
    combines its two members with. Raises if the artifact is missing rather
    than falling back to a guessed value -- see BLEND_CONFIG above."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/build_blend_config.py to derive it "
            "from the walk-forward report (models/walkforward/blend_b1.json)"
        )
    return json.loads(path.read_text())


# --- the deployed simulator (SP3) ------------------------------------------
# The four simulation parameters `SimulatorPredictor` plays a fight out with.
# They are hashed by `mma.versioning.MODEL_ARTIFACT_GLOBS` for exactly the
# reason `models/blend.json` is: every one of them moves a probability, and a
# number that moves a probability while living outside the hash silently
# changes what a recorded prediction means. `mma.simulator`'s DEFAULT_*
# constants are the HARNESS's values -- what `HazardCandidate` ran with when
# the hybrid was measured; this file is the DEPLOYED values, and
# `tests/test_versioning.py` pins the two together so a deployment cannot
# quietly serve a simulation the harness never scored.
SIMULATOR_CONFIG = ROOT / "models" / "simulator.json"
#: The keys `load_simulator_config` requires. Anything that changes a
#: simulated probability belongs here, or the track record's premise breaks.
SIMULATOR_PARAMETERS = ("n_runs", "alpha", "sim_seed", "default_rounds")


def load_simulator_config(path: Path = SIMULATOR_CONFIG) -> dict:
    """The committed simulation parameters, validated.

    Raises if the artifact is missing or incomplete rather than falling back
    to the module defaults: serving a guessed `n_runs` or `alpha` would
    produce a prediction no report describes, which is the same failure the
    artifact hash exists to prevent one level up (see SIMULATOR_CONFIG).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; it carries the simulation parameters the hybrid "
            "was measured with (n_runs, alpha, sim_seed, default_rounds) and "
            "there is deliberately no in-code fallback"
        )
    config = json.loads(path.read_text())
    missing = [key for key in SIMULATOR_PARAMETERS if key not in config]
    if missing:
        raise ValueError(f"{path} is missing simulation parameter(s) {missing}")
    return config


def load_deployed_metrics(path: Path = TORCH_METRICS) -> dict:
    """models/torch/metrics_val.json as a dict, or {} when it does not exist."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def deployed_training_mask(features: pd.DataFrame, metrics: dict | None = None) -> pd.Series:
    """Boolean mask of the rows the deployed ensemble was trained on.

    `metrics` is models/torch/metrics_val.json (read from disk when None).
    Under the refit_through recipe (the default deployment since SP1) the
    ensemble trained on every decisive fight dated <= `train_through`, so
    that is the mask -- in practice every row of features.parquet. A
    split-mode metrics file (no `mode` key) does not record its cutoff, so
    the original split, date < TRAIN_END, is assumed.
    """
    if metrics is None:
        metrics = load_deployed_metrics()
    if metrics.get("mode") == "refit_through":
        return features["date"] <= pd.Timestamp(metrics["train_through"])
    return features["date"] < TRAIN_END


def compute_display_priors(features: pd.DataFrame, metrics: dict | None = None) -> dict:
    """Empirical class frequencies over the deployed model's training rows.

    Phase 5 final review finding: the method/round heads are trained with
    class-weighted loss (`class_weights` in train_loop.py) so the model
    doesn't collapse onto the majority class. That makes the raw softmax
    outputs class-weight-biased, not calibrated probabilities -- on
    validation finishes, 5-round fights show predicted P(rounds 4-5) = 0.696
    vs an empirical rate of 0.182 (~3.8x overstated), and 3-round fights'
    P(round 3) is roughly doubled.

    These empirical priors are one side of the comparison
    scripts/check_display_calibration.py makes (see
    `compute_correction_factors` for the other). The row set is
    `deployed_training_mask`
    -- the rows the committed ensemble actually trained on, read from
    models/torch/metrics_val.json (pass `metrics` to override): all rows
    through `train_through` for a refit_through model, date < TRAIN_END
    for the original split. Restricting to the model's own training rows
    keeps the correction a property of the deployed model rather than
    leaking a held-out period's class balance into the displayed numbers;
    with the refit recipe there is no held-out period, so the base rates
    are simply the full history the model saw.
    """
    train = features[deployed_training_mask(features, metrics)]

    method_counts = train["y_method"].value_counts()
    method_total = method_counts.sum()
    method_prior = {
        cls: (float(method_counts.get(cls, 0)) / method_total if method_total else 0.0)
        for cls in METHOD_CLASSES
    }

    def _round_prior(subset: pd.DataFrame) -> dict:
        finishes = subset[subset["y_finish_round"].notna()]
        counts = finishes["y_finish_round"].value_counts()
        total = counts.sum()
        return {
            cls: (float(counts.get(cls, 0)) / total if total else 0.0)
            for cls in ROUND_CLASSES
        }

    # scheduled_rounds is nullable (Int64); fillna(3) matches the same
    # three-round default used for the "45" logit mask in MultiTaskNet.round_probs.
    sched = train["scheduled_rounds"].fillna(3)
    round_3 = _round_prior(train[sched <= 3])
    round_3["45"] = 0.0  # 3-round fights cannot reach rounds 4-5 by construction
    total_3 = sum(round_3.values())
    if total_3 > 0:
        round_3 = {cls: v / total_3 for cls, v in round_3.items()}

    round_5 = _round_prior(train[sched == 5])

    return {"method": method_prior, "round_3": round_3, "round_5": round_5}


def compute_correction_factors(empirical: dict, mean_predicted: dict) -> dict:
    """Mean-matching factors: factor(c) = empirical_prior(c) / mean_model_predicted(c).

    **No longer applied to anything.** Until SP3 these multiplied every
    displayed method and round probability, because the class-weighted heads
    below were miscalibrated in aggregate. The deployed scorer's method and
    round splits are now marginals of the simulator's joint distribution, and
    `scripts/check_display_calibration.py` measured them as landing within a
    few points of the base rates unaided -- so the correction is retired, and
    this function survives as the measurement's own arithmetic: a factor near
    1 IS the statement that no correction is needed. Applying it now would move
    each marginal off the joint the app's outcome table is read from.

    A plain multiply-by-prior (Saerens) correction was too weak here because
    the class-weighted heads' likelihood *ratios* are themselves miscalibrated
    (e.g. mean predicted P(rounds 4-5) on train 5-round finishes is ~0.7 vs an
    empirical 0.18, so even after multiplying by the prior the displayed P(45)
    stayed ~0.68). Dividing by the model's own mean predicted probability on
    its training rows makes the *aggregate* corrected distribution match the
    empirical base rates exactly (before per-row renormalization) while
    preserving each fight's relative signal.

    `mean_predicted` is the scorer's mean predicted distribution over the
    matching deployed-training rows (`deployed_training_mask`; computed by
    scripts/check_display_calibration.py).
    Guard: if mean_predicted(c) < 1e-6 (e.g. the "45" class for 3-round
    fights, which the model masks to ~0), the factor is set to 0.0 rather
    than exploding.
    """
    return {
        cls: (
            float(empirical[cls]) / float(mean_predicted[cls])
            if mean_predicted.get(cls, 0.0) >= 1e-6
            else 0.0
        )
        for cls in empirical
    }


class Ensemble:
    def __init__(self, nets, temperatures, preprocessor):
        self.nets = nets
        self.temperatures = temperatures
        self.preprocessor = preprocessor

    @classmethod
    def load(cls, directory=ROOT / "models" / "torch") -> "Ensemble":
        directory = Path(directory)
        preprocessor = Preprocessor.load(directory / "preprocess.json")
        nets, temperatures = [], []
        for path in sorted(directory.glob("net_seed*.pt")):
            payload = torch.load(path, weights_only=False)
            net = MultiTaskNet(
                n_features=payload["n_features"],
                n_weight_classes=payload["n_weight_classes"],
            )
            net.load_state_dict(payload["state_dict"])
            net.eval()
            nets.append(net)
            temperatures.append(float(payload["temperature"]))
        if not nets:
            raise FileNotFoundError(f"no checkpoints in {directory}")
        return cls(nets, temperatures, preprocessor)

    @torch.no_grad()
    def predict_members(self, features: pd.DataFrame) -> list[dict]:
        """Each seed's calibrated head probabilities, in seed order.

        `predict` is the mean of these. `BlendedPredictor` needs them
        individually so its reported spread describes the models that actually
        serve rather than the torch half alone.
        """
        x, wc = self.preprocessor.transform(features)
        x_t, wc_t = torch.tensor(x), torch.tensor(wc)
        three_round = torch.tensor(
            (features["scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        )
        members = []
        for net, temperature in zip(self.nets, self.temperatures):
            winner_logits, method_logits, round_logits = net(x_t, wc_t)
            members.append({
                "winner": torch.sigmoid(winner_logits / temperature).numpy(),
                "method": torch.softmax(method_logits, dim=1).numpy(),
                "round": MultiTaskNet.round_probs(round_logits, three_round).numpy(),
            })
        return members

    def predict(self, features: pd.DataFrame) -> dict:
        members = self.predict_members(features)
        winner = np.stack([m["winner"] for m in members])
        return {
            "winner_prob": winner.mean(axis=0),
            "winner_spread": winner.max(axis=0) - winner.min(axis=0),
            "method_probs": np.mean([m["method"] for m in members], axis=0),
            "round_probs": np.mean([m["round"] for m in members], axis=0),
            "method_classes": METHOD_CLASSES,
            "round_classes": ROUND_CLASSES,
        }

    @torch.no_grad()
    def mc_dropout(self, features: pd.DataFrame, passes: int = 100, seed: int = 0):
        """Stochastic winner probabilities from seed-0 net, dropout-only train mode.

        Phase 4 final review requirement: MC dropout must NOT call plain
        `net.train()` on a shared model instance, because that would also
        flip BatchNorm into train mode and mutate its running statistics
        (which are used for point predictions elsewhere). Instead we flip
        ONLY the `nn.Dropout` submodules to train mode and leave BatchNorm
        (and everything else) in eval mode, so running stats never change.
        This is verified by test_mc_dropout_preserves_batchnorm, which
        compares every buffer (including BatchNorm running_mean/var)
        before and after an MC dropout call.
        """
        net, temperature = self.nets[0], self.temperatures[0]
        x, wc = self.preprocessor.transform(features)
        x_t, wc_t = torch.tensor(x), torch.tensor(wc)
        for module in net.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        torch.manual_seed(seed)
        samples = []
        for _ in range(passes):
            winner_logits, _, _ = net(x_t, wc_t)
            samples.append(torch.sigmoid(winner_logits / temperature).numpy())
        net.eval()
        return np.stack(samples)


class BlendedPredictor:
    """The deployed scorer: XGBoost seed ensemble + torch ensemble, calibrated.

    SP2.2's shipped candidate B1 (docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md,
    models/walkforward/sp2_2_decision.json). Its `predict` returns exactly the
    dict `Ensemble.predict` returns, so `predict_symmetrized`, the app, the
    prospective run and the display-priors build all keep working against one
    contract -- and, in particular, **the corner-averaging in
    `predict_symmetrized` is applied to the BLEND**, not to one member and then
    blended, which would not be the same number.

    The three operations that combine the members come from `mma.blend`, the
    same module `mma.candidates.BlendCandidate` uses, so the served prediction
    is the construction the harness measured rather than a re-implementation.
    Two things necessarily differ from the harness, both because deployment has
    no held-out year:

    * the members are the REFIT-through-latest fits (fixed budgets, all data)
      rather than per-fold early-stopped ones, exactly as the torch member
      alone was before this;
    * the post-average temperature is the fixed value committed to
      `models/blend.json` (`BLEND_CONFIG`) rather than one fitted per fold,
      derived from the harness report's per-fold temperatures by the median
      rule `run_walkforward.fixed_budget_from` uses.

    `winner_spread` is the spread of the five PER-SEED blends -- seed i's XGB
    booster blended with seed i's net, temperature-scaled. Their mean is the
    headline probability exactly (the blend is linear in each member's mean),
    so this is a real "how much do the members disagree" number for the ten
    models that serve, and it is what the app's +- band reports.
    """

    def __init__(self, ensemble: "Ensemble", boosters: dict, weight: float,
                 temperature: float):
        self.ensemble = ensemble
        self.boosters = boosters  # {"winner"/"method"/"round": [XGBClassifier, ...]}
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.preprocessor = ensemble.preprocessor  # callers introspect the contract

    @classmethod
    def load(cls, root: Path = ROOT, weight: float | None = None,
             temperature: float | None = None) -> "BlendedPredictor":
        """Load both members from the committed artifacts.

        The XGBoost heads are `models/xgb_<head>_seed<seed>.json`, sorted by
        seed so the per-seed pairing with the torch nets is stable. A head with
        no artifacts is a loud error: silently serving a four-model blend, or a
        torch-only one, is exactly the failure the model hash exists to make
        impossible.

        `weight`/`temperature` default to the committed `models/blend.json`
        (`root / "models" / "blend.json"`, via `load_blend_config`) -- the
        deployed configuration the version hash covers. Pass them explicitly
        to override for a test or an experiment against a candidate that isn't
        deployed; that override is never read from disk and is exactly what it
        says, nothing more.
        """
        root = Path(root)
        if weight is None or temperature is None:
            config = load_blend_config(root / "models" / "blend.json")
            weight = config["weight"] if weight is None else weight
            temperature = config["temperature"] if temperature is None else temperature
        ensemble = Ensemble.load(root / "models" / "torch")
        boosters = {}
        for head in ("winner", "method", "round"):
            paths = sorted((root / "models").glob(f"xgb_{head}_seed*.json"),
                           key=lambda q: int(q.stem.rsplit("seed", 1)[1]))
            if not paths:
                raise FileNotFoundError(
                    f"no XGBoost {head} boosters under {root / 'models'} "
                    f"(expected xgb_{head}_seed*.json); run scripts/train_xgb.py"
                )
            models = []
            for path in paths:
                model = xgb.XGBClassifier(enable_categorical=True)
                model.load_model(path)
                models.append(model)
            boosters[head] = models
        return cls(ensemble, boosters, weight, temperature)

    def _xgb_predict(self, features: pd.DataFrame) -> list[dict]:
        """One dict of head probabilities per XGB seed, row-aligned.

        `align_to_booster` reshapes the served frame to each head's own trained
        columns and re-categorises `weight_class` with that model's category
        list, so a division the model never saw becomes a missing value rather
        than an XGBoostError -- the same graceful degradation the torch
        member's preprocessor already applies to an unknown weight class.
        """
        x = feature_frame(features)
        per_head = {}
        for head, models in self.boosters.items():
            per_head[head] = [
                model.predict_proba(align_to_booster(x, model.get_booster()))
                for model in models
            ]
        n_seeds = len(per_head["winner"])
        return [
            {
                "winner": per_head["winner"][i][:, 1],
                "method": per_head["method"][i],
                "round": per_head["round"][i],
            }
            for i in range(n_seeds)
        ]

    def predict(self, features: pd.DataFrame) -> dict:
        three_round = (features["scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        torch_members = self.ensemble.predict_members(features)
        xgb_members = self._xgb_predict(features)

        torch_mean = {head: np.mean([m[head] for m in torch_members], axis=0)
                      for head in ("winner", "method", "round")}
        xgb_mean = {head: np.mean([m[head] for m in xgb_members], axis=0)
                    for head in ("winner", "method", "round")}

        blended = blend_heads(xgb_mean, torch_mean, self.weight)
        winner = apply_temperature(blended["winner"], self.temperature)
        rounds = mask_round_45(blended["round"], three_round)

        # Per-seed blends, for the disagreement band only. Pairing seed i with
        # seed i is arbitrary but harmless: their mean is the headline
        # probability regardless of the pairing, because the blend is linear.
        paired = min(len(torch_members), len(xgb_members))
        per_seed = np.stack([
            apply_temperature(
                self.weight * np.asarray(xgb_members[i]["winner"], dtype=float)
                + (1.0 - self.weight) * np.asarray(torch_members[i]["winner"], dtype=float),
                self.temperature,
            )
            for i in range(paired)
        ])
        return {
            "winner_prob": winner,
            "winner_spread": per_seed.max(axis=0) - per_seed.min(axis=0),
            "method_probs": blended["method"],
            "round_probs": rounds,
            "method_classes": METHOD_CLASSES,
            "round_classes": ROUND_CLASSES,
        }

    def mc_dropout(self, features: pd.DataFrame, passes: int = 100, seed: int = 0):
        """MC dropout samples from the TORCH member's seed-0 net.

        The XGBoost member has no dropout and no cheap stochastic analogue, so
        this describes the torch half's parameter uncertainty only. The app
        re-centres the samples on the blend's headline probability via
        `predict_symmetrized`'s `mc_dropout_shift`; treat the resulting spread
        as the torch member's, drawn around the blend's mean, rather than as
        the blend's own posterior. It is a display aid, not a reported metric.
        """
        return self.ensemble.mc_dropout(features, passes=passes, seed=seed)


class SimulatorPredictor:
    """The deployed scorer since SP3: the blend's winner, the simulator's shape.

    SP3's shipped candidate, the pre-registered hybrid
    (`mma.candidates.HybridCandidate`, `models/walkforward/sp3_decision.json`):

        P(winner, method, round) = P_blend(winner) x P_sim(method, round | winner)

    A `BlendedPredictor` supplies the first factor and the five-seed hazard and
    decision models supply the second, played out `n_runs` times per fight per
    corner orientation by `mma.simulator.simulate`. What comes back is one
    joint distribution over the outcome cells `mma.evaluate` defines, from
    which the method and finish-round marginals and P(goes the distance) are
    read off -- so they cannot contradict each other or the winner the way
    three independent heads could.

    **The winner is the blend's, untouched.** `predict` returns the blend's
    `winner_prob` array itself, and `mma.joint.impose_winner_marginal` scales
    each corner's block of simulated cells so the joint's winner marginal is
    exactly that number. Deploying the hybrid therefore cannot move a single
    winner probability -- that is the whole point of the spec's fallback
    design, and `tests/test_inference.py` pins it rather than assuming it.

    **The arithmetic is the harness's, not a re-implementation.**
    `mma.simulator.simulate_fights` (the per-fight simulation and the
    corner-averaging that composes two orientations into one joint),
    `mma.simulator.mean_over_seeds` / `normalise_rows` (the seed ensemble),
    `mma.hazard.round_frame` / `mirror_corners` (the two frames the hazard
    member is fed) and `mma.joint.impose_winner_marginal` /
    `marginals_from_cells` are the same functions `HazardCandidate` and
    `HybridCandidate` call. `tests/test_serving_parity.py` serves the very
    models a harness fold fitted and asserts the cells come back identical,
    which is the prediction-level form of the feature-row parity this project
    already enforces.

    Two things necessarily differ from the harness, both because deployment
    has no held-out year -- the same two that already differ for the blend:
    the members are the refit-through-latest fits (fixed budgets, all data)
    rather than per-fold early-stopped ones, and the simulation parameters are
    the committed `models/simulator.json` rather than the module defaults
    (they are equal today, and a test keeps them so).
    """

    def __init__(self, blend: "BlendedPredictor", hazard_models: list,
                 decision_models: list, config: dict):
        self.blend = blend
        self.hazard_models = hazard_models
        self.decision_models = decision_models
        self.n_runs = int(config["n_runs"])
        self.alpha = float(config["alpha"])
        self.sim_seed = int(config["sim_seed"])
        self.default_rounds = int(config["default_rounds"])
        # Introspection the app and the display-priors build already do
        # against the blend; the hybrid IS the blend plus the simulator, so it
        # forwards them rather than making callers reach through `.blend`.
        self.ensemble = blend.ensemble
        self.preprocessor = blend.preprocessor
        self.weight = blend.weight
        self.temperature = blend.temperature

    @classmethod
    def load(cls, root: Path = ROOT, blend: "BlendedPredictor | None" = None,
             config: dict | None = None) -> "SimulatorPredictor":
        """Load the blend plus the two simulator members from the committed
        artifacts (`models/xgb_{hazard,decision}_seed*.json`).

        A member with no artifacts is a loud error for the same reason a
        missing booster is in `BlendedPredictor.load`: serving a
        three-seed simulator, or falling back to the blend's own method and
        round heads, would be a scorer no report describes.

        A *torn* pair is the same error wearing a disguise, so `_check_seeds`
        rejects it too -- see there for what that is and what it can and
        cannot catch.
        """
        root = Path(root)
        blend = BlendedPredictor.load(root) if blend is None else blend
        config = load_simulator_config(root / "models" / "simulator.json") if config is None else config
        members, seeds = {}, {}
        for member in ("hazard", "decision"):
            paths = sorted((root / "models").glob(f"xgb_{member}_seed*.json"),
                           key=lambda q: int(q.stem.rsplit("seed", 1)[1]))
            if not paths:
                raise FileNotFoundError(
                    f"no XGBoost {member} models under {root / 'models'} "
                    f"(expected xgb_{member}_seed*.json); run scripts/train_hazard.py"
                )
            seeds[member] = [int(path.stem.rsplit("seed", 1)[1]) for path in paths]
            models = []
            for path in paths:
                model = xgb.XGBClassifier(enable_categorical=True)
                model.load_model(path)
                models.append(model)
            members[member] = models
        cls._check_seeds(root, seeds["hazard"], seeds["decision"])
        return cls(blend, members["hazard"], members["decision"], config)

    @staticmethod
    def _check_seeds(root: Path, hazard_seeds: list[int], decision_seeds: list[int]) -> None:
        """Refuse a simulator whose two members are not the same fit.

        `scripts/train_hazard.py` writes interleaved -- hazard seed *n*, then
        decision seed *n*, then the next seed -- in place and non-atomically,
        and writes `models/hazard_metrics.json` only after the whole loop. An
        interrupted local run therefore leaves per-seed artifacts on disk that
        load perfectly well and are not a single fit. Until this check that
        was invisible: the filenames are the same, the glob is non-empty, and
        the ensemble serves.

        Two shapes are rejected:

        * the members disagree about which seeds exist, which is what an
          interruption between a hazard save and the decision save beside it
          leaves;
        * the seeds on disk are not the seeds `hazard_metrics.json` says were
          fitted, which is what a run at a different ``SEEDS`` leaves --
          the previous fit's extra seed files keep their filenames and load
          beside the new ones, so the count alone proves nothing.

        What it does NOT catch, and no load-time check could: an interruption
        at a loop boundary, which leaves both members with the same seed set
        and the same count, some seeds from the new fit and some from the old.
        Detecting that needs the writes to be atomic (fit into a temp dir,
        rename over) rather than a reader that can only see filenames.

        The metrics comparison runs only when the file is present, because a
        root holding a fold's models and nothing else is a legitimate caller
        (`tests/test_serving_parity.py`) and that file is not part of
        `mma.versioning.MODEL_ARTIFACT_GLOBS`. The deployed root always has
        one, and a test pins it against the committed artifacts.
        """
        if sorted(hazard_seeds) != sorted(decision_seeds):
            raise ValueError(
                f"the simulator's two members are not the same fit: hazard seeds "
                f"{sorted(hazard_seeds)} but decision seeds {sorted(decision_seeds)} "
                f"under {root / 'models'}; re-run scripts/train_hazard.py"
            )
        metrics_path = root / "models" / "hazard_metrics.json"
        if not metrics_path.exists():
            return
        expected = json.loads(metrics_path.read_text()).get("seeds")
        if expected is not None and sorted(hazard_seeds) != sorted(int(s) for s in expected):
            raise ValueError(
                f"the simulator's seeds on disk ({sorted(hazard_seeds)}) are not the "
                f"seeds {metrics_path.name} says were fitted ({sorted(expected)}); "
                f"the artifacts and hazard_metrics.json describe different fits -- "
                f"re-run scripts/train_hazard.py"
            )

    def _rounds(self, features: pd.DataFrame) -> np.ndarray:
        """Rounds simulated per fight: `scheduled_rounds`, or the committed
        default for the fights that have none (45 of them in the training
        table), which is what `HazardCandidate` bounds them by too."""
        return (features["scheduled_rounds"].fillna(self.default_rounds)
                .to_numpy(dtype=int))

    def _mean_proba(self, models: list, x: pd.DataFrame) -> np.ndarray:
        """The seed ensemble's prediction, each member aligned to its own
        trained columns and categories (`align_to_booster`, as the blend's
        XGB member does, so an unseen weight class degrades to a missing
        value rather than raising)."""
        return mean_over_seeds(
            [model.predict_proba(align_to_booster(x, model.get_booster()))
             for model in models]
        )

    def predict(self, features: pd.DataFrame) -> dict:
        """`BlendedPredictor.predict`'s dict, with the method and round heads
        replaced by the simulator's and the joint distribution added.

        The extra keys are `joint_cells` (the `mma.evaluate` cell layout),
        `joint_zero_mass` (which of those cells no simulated fight landed in,
        before smoothing) and `p_distance`.

        A fight's RNG stream is `[sim_seed, its row position]`, so the same
        fight predicted at a different position in a batch comes back with the
        same distribution plus a different Monte Carlo draw -- of order the
        `p_a_wins` standard error, ~0.005 at the deployed `n_runs`. Serving
        goes through `predict_symmetrized` one matchup at a time, where the
        position is always 0, so a served prediction is reproducible exactly;
        the variation only shows up when the same row is scored inside
        different batches (`scripts/check_display_calibration.py`).
        """
        blended = self.blend.predict(features)
        n_rounds = self._rounds(features)
        mirror = mirror_corners(features)

        hazard = normalise_rows(self._mean_proba(
            self.hazard_models, feature_frame(round_frame(features, n_rounds))))
        mirrored = normalise_rows(self._mean_proba(
            self.hazard_models, feature_frame(round_frame(mirror, n_rounds))))
        decision = self._mean_proba(self.decision_models, feature_frame(features))[:, 1]
        mirror_decision = self._mean_proba(
            self.decision_models, feature_frame(mirror))[:, 1]

        simulated = simulate_fights(
            hazard, mirrored, decision, mirror_decision, n_rounds,
            n_runs=self.n_runs, alpha=self.alpha, sim_seed=self.sim_seed,
            n_round_classes=len(ROUND_CLASSES),
        )
        # The hybrid composition: the blend's winner imposed on the simulated
        # cells, which rescales each corner's block by one scalar and leaves
        # P(method, round | that corner wins) exactly as simulated.
        cells = impose_winner_marginal(
            simulated["cells"], blended["winner_prob"], METHOD_CLASSES, ROUND_CLASSES)
        marginals = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)
        return {
            **blended,
            "method_probs": marginals["method"],
            "round_probs": marginals["round"],
            "joint_cells": cells,
            "joint_zero_mass": simulated["zero_mass"],
            "p_distance": marginals["method"][:, METHOD_CLASSES.index("decision")],
        }

    def mc_dropout(self, features: pd.DataFrame, passes: int = 100, seed: int = 0):
        """MC dropout from the blend's torch member -- see
        `BlendedPredictor.mc_dropout`. The simulator has no dropout and this
        describes the winner probability, which is the blend's."""
        return self.blend.mc_dropout(features, passes=passes, seed=seed)


def predict_symmetrized(
    ensemble, matchup_ab: pd.DataFrame, matchup_ba: pd.DataFrame
) -> dict:
    """Predict a matchup from both orientations and average them.

    `ensemble` is anything with `Ensemble`'s `predict` contract -- since SP2.2
    the deployed caller passes a `BlendedPredictor`, so **the thing being
    symmetrized is the blend**: each orientation is blended and calibrated
    first, and the two blended probabilities are then corner-averaged. Blending
    two already-symmetrized members would be a different number, and averaging
    a member's corners after the blend would be a third.

    Phase 5 finding: the model is not perfectly symmetric under fighter
    order -- P(A beats B) + P(B beats A) can be off from 1.0 by ~15
    percentage points in a smoke test (see test_stronger_fighter_favored_and_symmetric,
    which only asserts within 0.08). This averages both orientations so the
    reported probability is exactly self-consistent: p + (1-p) == 1.

    `matchup_ab` and `matchup_ba` must be `build_matchup(...)` outputs for the
    same pair with fighters swapped (A-vs-B and B-vs-A respectively).
    Method and finish-round distributions describe the fight, not a corner,
    so they are averaged elementwise across orientations rather than flipped.
    A `SimulatorPredictor` additionally returns a joint distribution, which DOES
    name a corner: `_symmetrize_joint` maps the mirrored orientation back with
    `mma.joint.swap_corners` before averaging, and re-reads the method and round
    marginals off the averaged joint so every number reported comes from the one
    distribution reported.
    Ensemble spread is reported as the max of the two orientations' spreads
    (a conservative uncertainty estimate). The `mc_dropout_shift` field is
    the correction needed to re-center A-orientation-only MC dropout samples
    on the symmetrized headline probability, so callers can avoid a second,
    more expensive MC dropout pass on the B orientation.
    """
    result_ab = ensemble.predict(matchup_ab)
    result_ba = ensemble.predict(matchup_ba)
    p_ab = float(result_ab["winner_prob"][0])
    p_ba = float(result_ba["winner_prob"][0])
    p = 0.5 * (p_ab + (1.0 - p_ba))
    spread = max(float(result_ab["winner_spread"][0]), float(result_ba["winner_spread"][0]))
    method = 0.5 * (result_ab["method_probs"][0] + result_ba["method_probs"][0])
    rounds = 0.5 * (result_ab["round_probs"][0] + result_ba["round_probs"][0])
    out = {
        "winner_prob": p,
        "winner_spread": spread,
        "method_probs": method,
        "round_probs": rounds,
        "method_classes": result_ab["method_classes"],
        "round_classes": result_ab["round_classes"],
        "orientation_ab_prob": p_ab,
        "orientation_ba_prob": p_ba,
        "mc_dropout_shift": p - p_ab,
    }
    if "joint_cells" in result_ab:
        out.update(_symmetrize_joint(result_ab, result_ba, p))
    return out


def _symmetrize_joint(result_ab: dict, result_ba: dict, p: float) -> dict:
    """The joint half of `predict_symmetrized`, for a `SimulatorPredictor`.

    A joint names a corner on its winner axis, so the mirrored orientation is
    mapped back with `mma.joint.swap_corners` before averaging -- averaging
    the two elementwise would add corner A's KO distribution to corner B's.

    The winner marginal survives this exactly. Each orientation's joint
    already carries its own blend probability as its winner marginal
    (`impose_winner_marginal` makes it exact), so the average of `p_ab` and
    the swapped `1 - p_ba` is `p`, the symmetrized probability the rest of
    this function reports -- which is why the method and round marginals are
    RE-READ from the averaged joint rather than averaged separately: everything
    reported then comes from the one distribution that is also reported.

    `impose_winner_marginal` is applied once more to the average, which is a
    no-op to floating point (the drift it removes is ~1e-16, the rounding of
    an average of two numbers that were each already exact) and makes "the
    hybrid does not move a winner probability" true BY CONSTRUCTION rather
    than by an arithmetic argument this function would otherwise have to
    check at serving time. It replaces a runtime `raise`: a serving-path
    assertion whose only failure mode was a floating-point tail would have
    surfaced as a raw Streamlit traceback in the app and, worse, would have
    killed `mma.prospective.predict_event` for one matchup and every event
    after it in the weekly run. The invariant is asserted in
    `tests/test_inference.py` instead, where a real regression is caught in
    CI rather than in production.
    """
    method_classes = list(result_ab["method_classes"])
    round_classes = list(result_ab["round_classes"])
    cells = 0.5 * (
        np.asarray(result_ab["joint_cells"])[:1]
        + swap_corners(result_ba["joint_cells"], method_classes, round_classes)[:1]
    )
    cells = impose_winner_marginal(cells, np.array([p]), method_classes, round_classes)
    marginals = marginals_from_cells(cells, method_classes, round_classes)
    # A cell is only empty for the matchup if it was empty in BOTH orientations.
    zero_mass = (
        np.asarray(result_ab["joint_zero_mass"], dtype=bool)[0]
        & swap_corners(np.asarray(result_ba["joint_zero_mass"], dtype=float),
                       method_classes, round_classes)[0].astype(bool)
    )
    return {
        "method_probs": marginals["method"][0],
        "round_probs": marginals["round"][0],
        "joint_cells": cells[0],
        "joint_zero_mass": zero_mass,
        "p_distance": float(marginals["method"][0][method_classes.index("decision")]),
    }


# A ufcstats fighter id: 16 lowercase hex characters. Every id in
# data/processed/fighters.parquet has this shape (4,581 of 4,581), which is
# what makes it safe to use as a validation rule rather than a guess.
UFCSTATS_ID = re.compile(r"^[0-9a-f]{16}$")


def _fighter_id(bio: pd.Series, corner: str) -> str:
    """The ufcstats id of a bio row: its index label, as every caller passes it.

    The `external` block joins by id and never by name, so a bio row whose
    label is not an id has to be an error. Checking `isinstance(str)` was not
    enough, because the obvious wrong call --
    `fighters.set_index("name").loc["Jon Jones"]` -- produces a bio row whose
    label IS a string, just the wrong one. That passed the check and then
    matched nothing in the external table, so a mapped fighter was served
    all-NaN external values with `external_missing` quietly set: a silent
    wrong answer, and precisely the name-matching failure the id-only join
    exists to make impossible. Validating the SHAPE closes it.
    """
    fighter_id = getattr(bio, "name", None)
    if not isinstance(fighter_id, str) or not UFCSTATS_ID.match(fighter_id):
        raise KeyError(
            f"the {EXTERNAL_BLOCK!r} block joins by ufcstats fighter id (16 "
            f"lowercase hex characters), but the corner-{corner} bio row is not "
            f"labelled with one (got {fighter_id!r}); pass "
            "fighters.set_index('fighter_id').loc[id]"
        )
    return fighter_id


def build_matchup(
    snapshot_a: pd.Series, snapshot_b: pd.Series,
    bio_a: pd.Series, bio_b: pd.Series,
    weight_class: str, title_fight: bool, scheduled_rounds: int,
    as_of: pd.Timestamp,
    blocks=None,
    referee=None, referee_rates=None, event_country=None,
    notice_a=None, notice_b=None,
) -> pd.DataFrame:
    """One feature row matching the training feature contract (A vs B, no swap).

    The row itself is built by `mma.serving.feature_row`, the same function
    `mma.features.build_features` uses for the training table -- this
    function's only job is to turn a current-state snapshot plus a bio row
    into the state dict that builder expects. `tests/test_serving_parity.py`
    pins the two paths to identical values.

    *Which* state keys that dict carries comes from
    `feature_blocks.state_keys(blocks)`, not a tuple maintained here: a key a
    block declares that neither the derived extras below nor the snapshot nor
    the bio row can supply raises, rather than silently becoming NaN. `blocks`
    defaults to `feature_blocks.table_blocks()` -- the blocks the deployed
    model's own feature table was built from -- so the served row matches the
    trained one without every caller having to name them; pass an explicit
    list only to serve a different contract than the one on disk.

    The `external` block is the one place this function needs a fighter's
    IDENTITY rather than just their state: its table is joined by ufcstats
    `fighter_id`. That id is taken from the bio row's index label, which is
    what every caller already passes (`fighters.set_index("fighter_id").loc[id]`);
    a bio row without one raises rather than silently serving an unmatched,
    all-NaN external corner.

    The blocks beyond `base`/`external` need card-level facts a snapshot
    cannot carry, and each is an argument with a default that matches what a
    real future card actually supplies:

      * `context` -- `referee` (a Wikipedia card names none, so the default
        None is the honest case and produces the same two NaNs plus flag that
        an unrefereed training row gets), `referee_rates`
        (`mma.context.referee_rates` over the fights known as of the
        prediction) and `event_country` (`mma.context.event_country` of the
        venue). `home_country` is then the corner's nationality -- which only
        the `external` snapshot carries -- against that country.
      * `notice` -- `notice_a` / `notice_b`, the per-corner state from
        `mma.notice.from_observation` when the event page states a late
        replacement or a missed weight, and `mma.notice.unknown_state()`
        (the default) otherwise, which is what every future bout gets.
      * `trajectory` -- nothing extra: the Glicko triple is grown from the
        fighter's post-fight state over the days since their last bout with
        `mma.glicko.decay_days`, which is the same rule the training pass
        applies, and `years_since_ufc_debut` is measured from the snapshot's
        `first_date`.

    NOTE: elo_fights (via "pre_fights") and career_fights both read
    snapshot["career_fights"] here. In training these come from two separate
    counters -- the ratings table's fight count and the fight-history table's
    count -- which can differ slightly for a fighter if a bout was recorded
    in one table but not the other. At inference time there is only the
    single current-state snapshot, so both coincide exactly; the difference
    is negligible after standardization (Preprocessor.transform).
    """
    blocks = table_blocks() if blocks is None else blocks
    resolved = resolve_blocks(blocks)
    keys = state_keys(blocks)
    owners = state_key_blocks(blocks)

    ext_a = ext_b = None
    if EXTERNAL_BLOCK in resolved:
        table = external.load_table()
        ext_a = external.state_for(_fighter_id(bio_a, "a"), as_of, table)
        ext_b = external.state_for(_fighter_id(bio_b, "b"), as_of, table)

    if NOTICE_BLOCK in resolved:
        notice_a = notice.unknown_state() if notice_a is None else dict(notice_a)
        notice_b = notice.unknown_state() if notice_b is None else dict(notice_b)

    def side(snapshot, bio, ext, camp=None):
        age = (
            (as_of - bio["dob"]).days / 365.25 if pd.notna(bio["dob"]) else np.nan
        )
        days = (
            (as_of - snapshot["last_date"]).days
            if pd.notna(snapshot.get("last_date"))
            else np.nan
        )
        career_fights = snapshot.get("career_fights")
        # Keys the snapshot does not carry under the name the spec uses, or
        # does not carry at all (bio fields, the as-of derivations, the flags).
        extras = {
            "age": age,
            "height_cm": bio["height_cm"],
            "reach_cm": bio["reach_cm"],
            "reach_missing": pd.isna(bio["reach_cm"]),
            "dob_missing": pd.isna(bio["dob"]),
            "southpaw": bio["stance"] == "Southpaw",
            "debut": serving.debut_flag(career_fights),
            "days_since_last": days,
            "pre_overall": snapshot["elo_overall"],
            "pre_striking": snapshot["elo_striking"],
            "pre_grappling": snapshot["elo_grappling"],
            "pre_fights": career_fights,
        }
        if ext is not None:
            extras.update(ext)
        if TRAJECTORY_BLOCK in resolved:
            # The trained value is the fighter's post-fight deviation grown
            # over the lay-off, so serving grows it the same way rather than
            # serving the stale post-fight number.
            grown = glicko.decay_days(
                glicko.Rating(
                    float(snapshot["glicko_mu"]),
                    float(snapshot["glicko_phi"]),
                    float(snapshot["glicko_sigma"]),
                ),
                None if pd.isna(days) else days,
            )
            first_date = snapshot.get("first_date")
            extras.update({
                "pre_glicko_mu": grown.rating,
                "pre_glicko_phi": grown.rd,
                "pre_glicko_sigma": grown.volatility,
                "years_since_ufc_debut": (
                    (as_of - first_date).days / 365.25
                    if first_date is not None and pd.notna(first_date)
                    else np.nan
                ),
                "age_squared": age ** 2,
                "age_x_fights": (
                    age * float(career_fights)
                    if career_fights is not None and pd.notna(career_fights)
                    else np.nan
                ),
            })
        if camp is not None:
            extras.update(camp)
        if CONTEXT_BLOCK in resolved:
            nationality = (ext or {}).get(external.NATIONALITY)
            extras["home_country"] = fight_context_module.home_country(
                nationality, event_country
            )
        state = {}
        for key in keys:
            if key in extras:
                state[key] = extras[key]
            elif key in snapshot:
                state[key] = snapshot[key]
            elif key in bio:
                state[key] = bio[key]
            else:
                raise KeyError(
                    f"snapshot/bio cannot supply {key!r}, declared by block "
                    f"{owners[key]!r}; add it to mma.snapshots.build_snapshots "
                    "or to the derived extras in mma.inference.build_matchup"
                )
        return state

    fight_context = {
        "weight_class": weight_class,
        "title_fight": title_fight,
        "scheduled_rounds": scheduled_rounds,
    }
    if ext_a is not None:
        fight_context.update(external.fight_context(ext_a, ext_b))
    if NOTICE_BLOCK in resolved:
        fight_context.update(notice.fight_context(notice_a, notice_b))
    if CONTEXT_BLOCK in resolved:
        fight_context.update(
            fight_context_module.referee_context(referee, referee_rates)
        )
        fight_context.update(fight_context_module.home_context(
            (ext_a or {}).get(external.NATIONALITY),
            (ext_b or {}).get(external.NATIONALITY),
            event_country,
        ))
    row = serving.feature_row(
        side(snapshot_a, bio_a, ext_a, notice_a),
        side(snapshot_b, bio_b, ext_b, notice_b),
        fight_context,
        blocks=blocks,
    )
    frame = pd.DataFrame([row])
    frame["weight_class"] = frame["weight_class"].astype("string")
    return frame
