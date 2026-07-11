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
