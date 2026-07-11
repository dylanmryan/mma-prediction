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


def multitask_loss(
    winner_logits, method_logits, round_logits,
    y_winner, y_method, y_round, three_round,
    method_weights, round_weights,
    method_scale: float = 0.5, round_scale: float = 0.25,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(winner_logits, y_winner)
    method_known = y_method >= 0
    if method_known.any():
        loss = loss + method_scale * F.cross_entropy(
            method_logits[method_known], y_method[method_known],
            weight=method_weights,
        )
    round_known = y_round >= 0
    if round_known.any():
        logits = round_logits.clone()
        logits[three_round, 3] = _MASK_VALUE
        loss = loss + round_scale * F.cross_entropy(
            logits[round_known], y_round[round_known], weight=round_weights,
        )
    return loss
