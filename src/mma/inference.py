"""Load the committed ensemble and predict hypothetical matchups."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mma import context as fight_context_module
from mma import external, glicko, notice, serving
from mma.feature_blocks import (
    CONTEXT_BLOCK, EXTERNAL_BLOCK, NOTICE_BLOCK, TRAJECTORY_BLOCK,
    resolve_blocks, state_key_blocks, state_keys, table_blocks,
)
from mma.models.net import MultiTaskNet
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
from mma.tensors import Preprocessor

ROOT = Path(__file__).resolve().parents[2]

# The original split-protocol training cutoff (scripts/train_xgb.py and
# scripts/train_torch.py in split mode): train on date < TRAIN_END.
TRAIN_END = "2021-01-01"
# The deployed torch ensemble's metrics file, which records how it was trained.
TORCH_METRICS = ROOT / "models" / "torch" / "metrics_val.json"


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

    These empirical priors are the numerators of the mean-matching
    correction factors built by scripts/build_display_priors.py (see
    `compute_correction_factors`). The row set is `deployed_training_mask`
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

    A plain multiply-by-prior (Saerens) correction was too weak here because
    the class-weighted heads' likelihood *ratios* are themselves miscalibrated
    (e.g. mean predicted P(rounds 4-5) on train 5-round finishes is ~0.7 vs an
    empirical 0.18, so even after multiplying by the prior the displayed P(45)
    stayed ~0.68). Dividing by the model's own mean predicted probability on
    its training rows makes the *aggregate* corrected distribution match the
    empirical base rates exactly (before per-row renormalization) while
    preserving each fight's relative signal.

    `mean_predicted` is the ensemble's mean predicted distribution over the
    matching deployed-training rows (`deployed_training_mask`; built by
    scripts/build_display_priors.py).
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


def apply_prior_correction(probs: dict, factors: dict) -> dict:
    """Elementwise correction: p_display(c) ∝ p_model(c) * factor(c).

    `probs` and `factors` are both {class_label: value} dicts over the same
    class set. `factors` are the mean-matching correction factors from
    `compute_correction_factors` (models/torch/display_priors.json).
    Renormalizes so the output sums to 1. If the weighted sum is zero
    (e.g. all overlapping factors are zero), returns `probs` unchanged
    rather than dividing by zero.
    """
    corrected = {cls: p * factors.get(cls, 0.0) for cls, p in probs.items()}
    total = sum(corrected.values())
    if total <= 0:
        return dict(probs)
    return {cls: v / total for cls, v in corrected.items()}


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
    def predict(self, features: pd.DataFrame) -> dict:
        x, wc = self.preprocessor.transform(features)
        x_t, wc_t = torch.tensor(x), torch.tensor(wc)
        three_round = torch.tensor(
            (features["scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        )
        winner_probs, method_probs, round_probs = [], [], []
        for net, temperature in zip(self.nets, self.temperatures):
            winner_logits, method_logits, round_logits = net(x_t, wc_t)
            winner_probs.append(torch.sigmoid(winner_logits / temperature).numpy())
            method_probs.append(torch.softmax(method_logits, dim=1).numpy())
            round_probs.append(
                MultiTaskNet.round_probs(round_logits, three_round).numpy()
            )
        winner = np.stack(winner_probs)
        return {
            "winner_prob": winner.mean(axis=0),
            "winner_spread": winner.max(axis=0) - winner.min(axis=0),
            "method_probs": np.mean(method_probs, axis=0),
            "round_probs": np.mean(round_probs, axis=0),
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


def predict_symmetrized(
    ensemble: "Ensemble", matchup_ab: pd.DataFrame, matchup_ba: pd.DataFrame
) -> dict:
    """Predict a matchup from both orientations and average them.

    Phase 5 finding: the model is not perfectly symmetric under fighter
    order -- P(A beats B) + P(B beats A) can be off from 1.0 by ~15
    percentage points in a smoke test (see test_stronger_fighter_favored_and_symmetric,
    which only asserts within 0.08). This averages both orientations so the
    reported probability is exactly self-consistent: p + (1-p) == 1.

    `matchup_ab` and `matchup_ba` must be `build_matchup(...)` outputs for the
    same pair with fighters swapped (A-vs-B and B-vs-A respectively).
    Method and finish-round distributions describe the fight, not a corner,
    so they are averaged elementwise across orientations rather than flipped.
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
    return {
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
