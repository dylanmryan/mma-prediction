"""Load the committed ensemble and predict hypothetical matchups."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mma.models.net import MultiTaskNet
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
from mma.tensors import Preprocessor

ROOT = Path(__file__).resolve().parents[2]

# Same train/tuning cutoff used by scripts/train_xgb.py and scripts/train_torch.py.
TRAIN_END = "2021-01-01"


def compute_display_priors(features: pd.DataFrame) -> dict:
    """Empirical class frequencies from the training split (date < TRAIN_END).

    Phase 5 final review finding: the method/round heads are trained with
    class-weighted loss (`class_weights` in train_loop.py) so the model
    doesn't collapse onto the majority class. That makes the raw softmax
    outputs class-weight-biased, not calibrated probabilities -- on
    validation finishes, 5-round fights show predicted P(rounds 4-5) = 0.696
    vs an empirical rate of 0.182 (~3.8x overstated), and 3-round fights'
    P(round 3) is roughly doubled. We correct this Saerens-style at display
    time (see `apply_prior_correction`): multiply model probabilities by
    empirical training-set priors and renormalize, which converts the
    class-weighted output back into a frequency-respecting one. Priors come
    from the training split only, to avoid leaking validation/test-period
    class balance into the displayed numbers.
    """
    train = features[features["date"] < TRAIN_END]

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


def apply_prior_correction(probs: dict, priors: dict) -> dict:
    """Saerens-style posterior correction: p_display(c) ∝ p_model(c) * prior(c).

    `probs` and `priors` are both {class_label: value} dicts over the same
    class set. Renormalizes so the output sums to 1. If the weighted sum is
    zero (e.g. all overlapping priors are zero), returns `probs` unchanged
    rather than dividing by zero.
    """
    corrected = {cls: p * priors.get(cls, 0.0) for cls, p in probs.items()}
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


def build_matchup(
    snapshot_a: pd.Series, snapshot_b: pd.Series,
    bio_a: pd.Series, bio_b: pd.Series,
    weight_class: str, title_fight: bool, scheduled_rounds: int,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """One feature row matching the training feature contract (A vs B, no swap)."""
    def side(snapshot, bio):
        age = (
            (as_of - bio["dob"]).days / 365.25 if pd.notna(bio["dob"]) else np.nan
        )
        days = (
            (as_of - snapshot["last_date"]).days
            if pd.notna(snapshot.get("last_date"))
            else np.nan
        )
        return {
            "age": age,
            "height_cm": bio["height_cm"],
            "reach_cm": bio["reach_cm"],
            "southpaw": bio["stance"] == "Southpaw",
            "days_since_last": days,
            "pre_overall": snapshot["elo_overall"],
            "pre_striking": snapshot["elo_striking"],
            "pre_grappling": snapshot["elo_grappling"],
            "pre_fights": snapshot["career_fights"],
            **{
                name: snapshot.get(name)
                for name in (
                    "career_fights", "career_wins", "career_win_rate",
                    "career_finish_rate", "kd_pf", "sub_att_pf", "td_landed_pf",
                    "td_acc", "td_def", "sig_pm", "sig_absorbed_pm", "ctrl_share",
                    "streak", "last5_win_rate", "last5_avg_opp_elo",
                )
            },
        }

    first, second = side(snapshot_a, bio_a), side(snapshot_b, bio_b)
    row: dict = {
        "weight_class": weight_class,
        "title_fight": title_fight,
        "scheduled_rounds": scheduled_rounds,
    }
    # NOTE: elo_fights_diff (via "pre_fights") and career_fights_diff (via
    # history_names below) both read side(...)["career_fights"], i.e. the
    # same snapshot["career_fights"] value. In training these came from two
    # separate counters -- the ratings-table fight count and the fight-history
    # table's count -- which could differ slightly for a fighter if a bout
    # was recorded in one table but not the other. At inference time we only
    # have the single current-state snapshot, so both diffs coincide exactly;
    # this is negligible after standardization (Preprocessor.transform).
    diff_names = {
        "pre_overall": "elo", "pre_striking": "striking_elo",
        "pre_grappling": "grappling_elo", "pre_fights": "elo_fights",
        "height_cm": "height", "reach_cm": "reach", "age": "age",
    }
    history_names = (
        "career_fights", "career_wins", "career_win_rate", "career_finish_rate",
        "kd_pf", "sub_att_pf", "td_landed_pf", "td_acc", "td_def",
        "sig_pm", "sig_absorbed_pm", "ctrl_share", "streak", "days_since_last",
        "last5_win_rate", "last5_avg_opp_elo",
    )
    for name in history_names:
        row[f"{name}_diff"] = _minus(first.get(name), second.get(name))
    for source, out in diff_names.items():
        row[f"{out}_diff"] = _minus(first.get(source), second.get(source))
    for label, data in (("a", first), ("b", second)):
        row[f"age_{label}"] = data["age"]
        row[f"career_fights_{label}"] = data["career_fights"]
        row[f"southpaw_{label}"] = bool(data["southpaw"])
        row[f"debut_{label}"] = (data["career_fights"] or 0) == 0
    row["debut_matchup"] = row["debut_a"] ^ row["debut_b"]
    row["stance_mismatch"] = row["southpaw_a"] ^ row["southpaw_b"]
    frame = pd.DataFrame([row])
    frame["weight_class"] = frame["weight_class"].astype("string")
    return frame


def _minus(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)
