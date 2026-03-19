# src/utils/visualization.py

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset


def _hard_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((pred_bin * gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    return (2.0 * inter + eps) / (denom + eps)


def _to_hw(arr: np.ndarray) -> np.ndarray:
    """Strip batch/channel dims to get (H, W)."""
    arr = arr.squeeze()
    assert arr.ndim == 2, f"Expected 2-D array, got shape {arr.shape}"
    return arr


def _norm_for_display(img: np.ndarray) -> np.ndarray:
    """Linearly scale to [0, 1] for display."""
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


def _make_overlay(
    image_hw: np.ndarray,
    mask_hw: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Return an (H, W, 3) float32 RGB image with the mask region tinted yellow.
    """
    rgb = np.stack([image_hw, image_hw, image_hw], axis=-1).astype(np.float32)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    mask3 = mask_hw[..., None].astype(np.float32)
    rgb = rgb * (1.0 - alpha * mask3) + yellow * (alpha * mask3)
    return np.clip(rgb, 0.0, 1.0)


@torch.no_grad()
def save_val_panels(
    model: nn.Module,
    val_ds: Dataset,
    device: torch.device,
    save_dir: str | Path,
    n_samples: int = 3,
    indices: Optional[Sequence[int]] = None,
    overlay_alpha: float = 0.45,
    threshold: float = 0.5,
    prefix: str = "val_panel",
) -> None:
    """
    Save one PNG per sampled validation frame.

    Each PNG has 4 columns:
      1. ultrasound image
      2. ground-truth mask
      3. predicted mask
      4. image with predicted mask overlaid in yellow

    Args:
        model: trained model in eval mode (outputs logits).
        val_ds: validation dataset with keys "image" and "mask".
        device: device for inference.
        save_dir: directory where PNGs will be saved.
        n_samples: number of examples to save if indices not provided.
        indices: fixed dataset indices to use instead of random sampling.
        overlay_alpha: yellow overlay strength.
        threshold: sigmoid threshold for binary prediction.
        prefix: filename prefix.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(val_ds)
    if n == 0:
        print("save_val_panels: validation dataset is empty, skipping.", flush=True)
        return

    if indices is None:
        k = min(n_samples, n)
        indices = random.sample(range(n), k)
    else:
        indices = list(indices)

    model.eval()

    col_titles = ["Image", "GT Mask", "Predicted Mask", "Overlay (pred, yellow)"]

    for i, idx in enumerate(indices, start=1):
        sample = val_ds[idx]

        image_t = sample["image"]   # (1, H, W)
        mask_t = sample["mask"]     # (1, H, W)

        inp = image_t.unsqueeze(0).to(device)   # (1, 1, H, W)
        logits = model(inp)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_bin = (prob >= threshold).astype(np.float32)

        image_hw = _to_hw(image_t.numpy())
        gt_hw = _to_hw(mask_t.numpy())
        image_disp = _norm_for_display(image_hw)

        dice = _hard_dice(pred_bin, gt_hw)
        cine_id = sample.get("cine_id", f"idx={idx}")

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), squeeze=False)
        axes = axes[0]

        for col, title in enumerate(col_titles):
            axes[col].set_title(title, fontsize=10, fontweight="bold")

        axes[0].imshow(image_disp, cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(gt_hw, cmap="gray", vmin=0, vmax=1)
        axes[2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)

        overlay = _make_overlay(image_disp, pred_bin, alpha=overlay_alpha)
        axes[3].imshow(overlay)
        patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label="prediction")
        axes[3].legend(handles=[patch], fontsize=7, loc="lower right", framealpha=0.6)

        for ax in axes:
            ax.axis("off")

        fig.suptitle(f"{cine_id} | idx={idx} | Dice={dice:.3f}", fontsize=11)

        out_path = save_dir / f"{prefix}_{i:02d}_idx{idx}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved validation panel -> {out_path}", flush=True)