"""Training losses: cross-entropy, class-weighted cross-entropy and focal loss, each with label smoothing."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .data import class_loss_weights


class FocalLoss(nn.Module):
    """Cross-entropy scaled by (1 - p_true)^gamma, so confidently right images count less.
    gamma = 0 is plain cross-entropy."""

    def __init__(self, gamma: float, label_smoothing: float):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=self.label_smoothing)
        p_true = logits.softmax(dim=1).gather(1, target[:, None])[:, 0]
        return ((1 - p_true) ** self.gamma * ce).mean()


def build_loss(name: str, train_labels: torch.Tensor, n_classes: int, label_smoothing: float,
               focal_gamma: float) -> nn.Module:
    """name: ce | weighted_ce | focal. weighted_ce weighs classes inversely to their training counts."""
    if name == "focal":
        return FocalLoss(focal_gamma, label_smoothing)
    weight = class_loss_weights(train_labels.numpy(), n_classes) if name == "weighted_ce" else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
