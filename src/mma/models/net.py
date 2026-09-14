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
        margin_head: bool = False,
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
        # SP5's auxiliary scorecard-margin head. Constructed LAST and only when
        # asked, so a net without it draws exactly the parameters the
        # incumbent draws and committed checkpoints keep loading: the
        # `margin_scale = 0` arm is the incumbent, not a re-run of it.
        self.margin_head = nn.Linear(width, 1) if margin_head else None

    def forward(self, x: torch.Tensor, weight_class: torch.Tensor,
                return_margin: bool = False):
        """The three shipped heads; the margin only when asked for.

        Opt-in rather than a fourth element of every return, so `mma.inference`
        -- which unpacks exactly three -- needs no change and the serving path
        is untouched by SP5.
        """
        combined = torch.cat([x, self.weight_class_embedding(weight_class)], dim=1)
        hidden = self.trunk(combined)
        heads = (
            self.winner_head(hidden).squeeze(-1),
            self.method_head(hidden),
            self.round_head(hidden),
        )
        if not return_margin:
            return heads
        margin = (self.margin_head(hidden).squeeze(-1)
                  if self.margin_head is not None else None)
        return heads + (margin,)

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
    margin_logits: torch.Tensor | None = None,
    y_margin: torch.Tensor | None = None,
    margin_scale: float = 0.0,
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

    # SP5: the scorecard margin, masked to the rows that HAVE a scorecard.
    # Every finish and every decision whose cards did not parse carries NaN,
    # which is 61% of the table, so the mask is the common case rather than
    # the edge one -- and NaN must never reach the arithmetic, since a single
    # NaN gradient silently destroys the whole run.
    if margin_scale > 0 and margin_logits is not None and y_margin is not None:
        margin_known = torch.isfinite(y_margin)
        if margin_known.any():
            # Huber: a 5-round sweep scores |margin| near 7 while most
            # decisions sit near 1, and squared error would let the handful of
            # blowouts set the gradient for the whole head.
            margin_losses = F.smooth_l1_loss(
                margin_logits[margin_known], y_margin[margin_known],
                reduction="none",
            )
            if sample_weight is None:
                margin_loss = margin_losses.mean()
            else:
                margin_loss = _weighted_mean(
                    margin_losses, sample_weight[margin_known]
                )
            loss = loss + margin_scale * margin_loss
    return loss
