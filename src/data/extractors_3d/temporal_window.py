# src/data/extractors_3d/temporal_window.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def extract(
    cine: np.ndarray,
    mask: np.ndarray,
    annotation_index: int,
    metadata: Dict[str, Any] | None = None,
    z_window: int = 9,
    out_hw: Tuple[int, int] = (256, 256),
) -> Dict[str, np.ndarray]:
    """
    Extract a temporal window centered on the annotated frame, then resize
    spatially.

    Args:
        cine: Full cine, shape (T, H, W)
        mask: 2D needle mask for annotated frame, shape (H, W)
        annotation_index: Index of annotated frame in full cine
        metadata: Optional metadata, unused here but kept for interface consistency
        z_window: Number of frames in temporal window, should be odd
        out_hw: Output spatial size (out_h, out_w)

    Returns:
        dict with:
            image: (Z, out_h, out_w)
            mask: (out_h, out_w)
            center_index: local index of annotated frame within extracted window
    """
    if cine.ndim != 3:
        raise ValueError(f"Expected cine shape (T, H, W), got {cine.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (H, W), got {mask.shape}")

    T, _, _ = cine.shape
    out_h, out_w = int(out_hw[0]), int(out_hw[1])

    # Put the annotated frame at local index `left`
    left = z_window // 2
    right = z_window - left - 1
    center_index = left

    # Build temporal window with edge replication
    frame_indices = []
    for offset in range(-left, right + 1):
        idx = annotation_index + offset
        idx = min(max(idx, 0), T - 1)
        frame_indices.append(idx)

    image_window = cine[frame_indices]  # (Z, H, W)

    # Resize image window: (Z, H, W) -> (1, Z, H, W)
    image_t = torch.from_numpy(image_window).float()[None, ...]
    image_out = F.interpolate(
        image_t,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )

    # Resize mask: (H, W) -> (1, 1, H, W)
    mask_t = torch.from_numpy(mask).float()[None, None, ...]
    mask_out = F.interpolate(
        mask_t,
        size=(out_h, out_w),
        mode="nearest",
    )

    return {
        "image": image_out[0].cpu().numpy().astype(np.float32, copy=False),   # (Z, out_h, out_w)
        "mask": mask_out[0, 0].cpu().numpy().astype(np.float32, copy=False),  # (out_h, out_w)
        "center_index": center_index,
    }