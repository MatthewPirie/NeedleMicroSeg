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


def _norm_for_display(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


def _make_overlay(
    image_hw: np.ndarray,
    mask_hw: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    rgb = np.stack([image_hw, image_hw, image_hw], axis=-1).astype(np.float32)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    mask3 = mask_hw[..., None].astype(np.float32)
    rgb = rgb * (1.0 - alpha * mask3) + yellow * (alpha * mask3)
    return np.clip(rgb, 0.0, 1.0)


@torch.no_grad()
def save_val_panels_3d(
    model: nn.Module,
    val_ds: Dataset,
    device: torch.device,
    save_dir: str | Path,
    n_samples: int = 3,
    indices: Optional[Sequence[int]] = None,
    overlay_alpha: float = 0.45,
    threshold: float = 0.5,
    context_radius: int = 2,
    prefix: str = "val_panel_3d",
) -> None:
    """
    For each sampled validation case, save 2 PNGs:

    1) Center-frame panel:
       image | GT mask | predicted mask | overlay

    2) Context panel:
       predicted masks/overlays for frames around center, e.g. t-2..t+2
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(val_ds)
    if n == 0:
        print("save_val_panels_3d: validation dataset is empty, skipping.", flush=True)
        return

    if indices is None:
        k = min(n_samples, n)
        indices = random.sample(range(n), k)
    else:
        indices = list(indices)

    model.eval()

    for i, idx in enumerate(indices, start=1):
        sample = val_ds[idx]

        image_t = sample["image"]   # (1, Z, H, W)
        mask_t = sample["mask"]     # (1, H, W)

        inp = image_t.unsqueeze(0).to(device)   # (1, 1, Z, H, W)
        logits = model(inp)                     # (1, 1, Z, H, W)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()   # (Z, H, W)
        pred_bin_all = (probs >= threshold).astype(np.float32)

        image_np = image_t.numpy()[0]   # (Z, H, W)
        gt_hw = mask_t.numpy()[0]       # (H, W)

        Z = image_np.shape[0]
        center_index = sample.get("center_index", Z // 2)

        cine_id = sample.get("cine_id", f"idx={idx}")

        # ------------------------------------------------------------------
        # PNG 1: center-frame panel
        # ------------------------------------------------------------------
        center_img = image_np[center_index]
        center_pred = pred_bin_all[center_index]

        center_img_disp = _norm_for_display(center_img)
        dice = _hard_dice(center_pred, gt_hw)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), squeeze=False)
        axes = axes[0]

        col_titles = ["Image", "GT Mask", "Predicted Mask", "Overlay (pred, yellow)"]
        for col, title in enumerate(col_titles):
            axes[col].set_title(title, fontsize=10, fontweight="bold")

        axes[0].imshow(center_img_disp, cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(gt_hw, cmap="gray", vmin=0, vmax=1)
        axes[2].imshow(center_pred, cmap="gray", vmin=0, vmax=1)

        overlay = _make_overlay(center_img_disp, center_pred, alpha=overlay_alpha)
        axes[3].imshow(overlay)
        patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label="prediction")
        axes[3].legend(handles=[patch], fontsize=7, loc="lower right", framealpha=0.6)

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{cine_id} | idx={idx} | center={center_index} | Dice={dice:.3f}",
            fontsize=11,
        )

        out_path_main = save_dir / f"{prefix}_{i:02d}_idx{idx}_center.png"
        fig.tight_layout()
        fig.savefig(out_path_main, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved center validation panel -> {out_path_main}", flush=True)

        # ------------------------------------------------------------------
        # PNG 2: context panel (neighbor frames)
        # ------------------------------------------------------------------
        z_indices = list(
            range(
                max(0, center_index - context_radius),
                min(Z, center_index + context_radius + 1),
            )
        )

        ncols = len(z_indices)
        fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6), squeeze=False)

        for col, z in enumerate(z_indices):
            img_hw = image_np[z]
            pred_hw = pred_bin_all[z]
            img_disp = _norm_for_display(img_hw)
            overlay_hw = _make_overlay(img_disp, pred_hw, alpha=overlay_alpha)

            rel = z - center_index
            title = f"t{rel:+d}" if rel != 0 else "t (center)"

            axes[0, col].imshow(img_disp, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(title, fontsize=10, fontweight="bold")
            axes[0, col].axis("off")

            axes[1, col].imshow(overlay_hw)
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Image", fontsize=10)
        axes[1, 0].set_ylabel("Pred overlay", fontsize=10)

        fig.suptitle(
            f"{cine_id} | idx={idx} | context predictions around center={center_index}",
            fontsize=11,
        )

        out_path_context = save_dir / f"{prefix}_{i:02d}_idx{idx}_context.png"
        fig.tight_layout()
        fig.savefig(out_path_context, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved context validation panel -> {out_path_context}", flush=True)