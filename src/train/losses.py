# src/train/losses.py

from __future__ import annotations

import torch
import torch.nn as nn


def soft_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    batch_dice: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = probs.contiguous()
    targets = targets.contiguous()

    if batch_dice:
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum()
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice

    b = probs.shape[0]
    probs = probs.view(b, -1)
    targets = targets.view(b, -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class CompoundBCEDiceLoss(nn.Module):
    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        batch_dice: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.batch_dice = bool(batch_dice)
        self.eps = float(eps)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)

        bce_val = self.bce(logits, targets)
        dice_val = soft_dice_loss(
            probs,
            targets,
            batch_dice=self.batch_dice,
            eps=self.eps,
        )

        total = self.w_bce * bce_val + self.w_dice * dice_val

        parts = {
            "bce": bce_val,
            "dice_loss": dice_val,
        }
        return total, parts


@torch.no_grad()
def hard_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    batch_dice: bool = False,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.contiguous()
    targets = targets.contiguous()

    if batch_dice:
        inter = (preds * targets).sum()
        denom = preds.sum() + targets.sum()
        return (2.0 * inter + eps) / (denom + eps)

    b = preds.shape[0]
    preds = preds.view(b, -1)
    targets = targets.view(b, -1)

    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()
