# src/train/trainer_3d.py

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.train.losses import hard_dice_score, soft_dice_score


def train_one_epoch_3d(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    steps_per_epoch: int,
    log_every: int = 50,
    pin_memory: bool = False,
    scheduler=None,
    scaler=None,
    amp: bool = True,
) -> Dict[str, float]:
    model.train()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_score_sum = 0.0

    running_total = 0.0
    running_steps = 0

    it = iter(train_loader)

    for step in range(1, steps_per_epoch + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        imgs = batch["image"].to(device, non_blocking=pin_memory)   # (B, 1, Z, H, W)
        lbls = batch["mask"].to(device, non_blocking=pin_memory)    # (B, 1, H, W)

        use_amp = bool(amp) and (scaler is not None) and torch.cuda.is_available()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)  # (B, 1, Z, H, W)

            center_index = imgs.shape[2] // 2
            logits_center = logits[:, :, center_index, :, :]  # (B, 1, H, W)

            total_loss, parts = criterion(logits_center, lbls)

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            dice_score = hard_dice_score(logits_center, lbls)

        total_val = float(total_loss.item())
        bce_val = float(parts["bce"].detach().item())
        dice_loss_val = float(parts["dice_loss"].detach().item())
        dice_score_val = float(dice_score.detach().item())

        total_sum += total_val
        bce_sum += bce_val
        dice_loss_sum += dice_loss_val
        dice_score_sum += dice_score_val

        running_total += total_val
        running_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg_total = running_total / max(running_steps, 1)
            print(
                f"[Epoch {epoch}] step {step}/{steps_per_epoch} | train_total_loss={avg_total:.4f}",
                flush=True,
            )
            running_total = 0.0
            running_steps = 0

    return {
        "train_total_loss": total_sum / max(steps_per_epoch, 1),
        "train_bce": bce_sum / max(steps_per_epoch, 1),
        "train_dice_loss": dice_loss_sum / max(steps_per_epoch, 1),
        "train_dice_score": dice_score_sum / max(steps_per_epoch, 1),
    }


@torch.no_grad()
def validate_one_epoch_3d(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    epoch: int,
    log_every: int = 0,
    pin_memory: bool = False,
) -> Dict[str, float]:
    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_score_sum = 0.0
    soft_dice_score_sum = 0.0
    n_steps = 0

    for step, batch in enumerate(val_loader, start=1):
        imgs = batch["image"].to(device, non_blocking=pin_memory)   # (B, 1, Z, H, W)
        lbls = batch["mask"].to(device, non_blocking=pin_memory)    # (B, 1, H, W)

        logits = model(imgs)  # (B, 1, Z, H, W)

        center_index = imgs.shape[2] // 2
        logits_center = logits[:, :, center_index, :, :]  # (B, 1, H, W)

        total_loss, parts = criterion(logits_center, lbls)
        dice_score = hard_dice_score(logits_center, lbls)
        soft_dice = soft_dice_score(logits_center, lbls)

        total_sum += float(total_loss.item())
        bce_sum += float(parts["bce"].detach().item())
        dice_loss_sum += float(parts["dice_loss"].detach().item())
        dice_score_sum += float(dice_score.detach().item())
        soft_dice_score_sum += float(soft_dice.detach().item())
        n_steps += 1

        if log_every > 0 and step % log_every == 0:
            print(
                f"[Epoch {epoch}] val step {step}/{len(val_loader)} | val_total_loss={total_sum / n_steps:.4f}",
                flush=True,
            )

    return {
        "val_total_loss": total_sum / max(n_steps, 1),
        "val_bce": bce_sum / max(n_steps, 1),
        "val_dice_loss": dice_loss_sum / max(n_steps, 1),
        "val_hard_dice_score": dice_score_sum / max(n_steps, 1),
        "val_soft_dice_score": soft_dice_score_sum / max(n_steps, 1),
    }