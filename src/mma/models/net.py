"""Multi-task net: shared trunk, winner/method/finish-round heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MASK_VALUE = -1e9


class MultiTaskNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_weight_classes: int,
        embedding_dim: int = 4,
        hidden: tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.weight_class_embedding = nn.Embedding(n_weight_classes, embedding_dim)
        layers: list[nn.Module] = []
        width = n_features + embedding_dim
        for size in hidden:
            layers += [
                nn.Linear(width, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.winner_head = nn.Linear(width, 1)
        self.method_head = nn.Linear(width, 3)
        self.round_head = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor, weight_class: torch.Tensor):
        combined = torch.cat([x, self.weight_class_embedding(weight_class)], dim=1)
        hidden = self.trunk(combined)
        return (
            self.winner_head(hidden).squeeze(-1),
            self.method_head(hidden),
            self.round_head(hidden),
        )

    @staticmethod
    def round_probs(round_logits: torch.Tensor, three_round: torch.Tensor):
        logits = round_logits.clone()
        logits[three_round, 3] = _MASK_VALUE
        return F.softmax(logits, dim=1)


def _weighted_mean(losses, sample_weight, class_weight=None):
    """sum(s * l) / sum(s * c); plain mean if the denominator is zero.

    ``losses`` come from a ``reduction="none"`` call, so for the class-weighted
    terms each row is already multiplied by its class weight ``c``; dividing by
    ``sum(s * c)`` mirrors PyTorch's ``reduction="mean"`` normalisation.
    """
    denom = sample_weight if class_weight is None else sample_weight * class_weight
    total = denom.sum()
    if total <= 0:
        return losses.mean()
    return (sample_weight * losses).sum() / total


def multitask_loss(
    winner_logits, method_logits, round_logits,
    y_winner, y_method, y_round, three_round,
    method_weights, round_weights,
    method_scale: float = 0.5, round_scale: float = 0.25,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Winner BCE + method_scale * method CE + round_scale * round CE.

    Method/round terms only cover rows whose label is known (>= 0); the R4-5
    logit is masked for three-round fights. ``method_weights``/``round_weights``
    are per-class weights (PyTorch ``cross_entropy(weight=...)`` semantics).

    ``sample_weight`` (1-D float tensor aligned with the batch, e.g. recency
    weights) turns every mean into a weighted mean: for the winner term
    ``sum(s_i * l_i) / sum(s_i)``; for the method/round terms the class weight
    and the sample weight multiply, ``sum(s_i * w[y_i] * l_i) / sum(s_i * w[y_i])``
    over the known rows -- so all-ones sample weights reproduce the unweighted
    loss exactly, and a zero weight sum falls back to a plain mean. When
    ``sample_weight`` is None the original unweighted expressions are used
    verbatim so existing checkpoints stay bit-reproducible.
    """
    if sample_weight is None:
        loss = F.binary_cross_entropy_with_logits(winner_logits, y_winner)
    else:
        loss = _weighted_mean(
            F.binary_cross_entropy_with_logits(winner_logits, y_winner, reduction="none"),
            sample_weight,
        )
    method_known = y_method >= 0
    if method_known.any():
        if sample_weight is None:
            method_loss = F.cross_entropy(
                method_logits[method_known], y_method[method_known],
                weight=method_weights,
            )
        else:
            targets = y_method[method_known]
            method_loss = _weighted_mean(
                F.cross_entropy(method_logits[method_known], targets,
                                weight=method_weights, reduction="none"),
                sample_weight[method_known], method_weights[targets],
            )
        loss = loss + method_scale * method_loss
    round_known = y_round >= 0
    if round_known.any():
        logits = round_logits.clone()
        logits[three_round, 3] = _MASK_VALUE
        if sample_weight is None:
            round_loss = F.cross_entropy(
                logits[round_known], y_round[round_known], weight=round_weights,
            )
        else:
            targets = y_round[round_known]
            round_loss = _weighted_mean(
                F.cross_entropy(logits[round_known], targets,
                                weight=round_weights, reduction="none"),
                sample_weight[round_known], round_weights[targets],
            )
        loss = loss + round_scale * round_loss
    return loss
