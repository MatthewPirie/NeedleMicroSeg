from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def extract(
    image: np.ndarray,
    mask: np.ndarray,
    metadata: Dict[str, Any] | None = None,
    out_hw: Tuple[int, int] = (256, 256),
) -> Dict[str, np.ndarray]:
    out_h, out_w = int(out_hw[0]), int(out_hw[1])

    # Convert to torch tensors with shape (N, C, H, W)
    image_t = torch.from_numpy(image).float()[None, None, ...]
    mask_t = torch.from_numpy(mask).float()[None, None, ...]

    image_out = F.interpolate(
        image_t,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )

    mask_out = F.interpolate(
        mask_t,
        size=(out_h, out_w),
        mode="nearest",
    )

    return {
        "image": image_out[0, 0].cpu().numpy().astype(np.float32, copy=False),
        "mask": mask_out[0, 0].cpu().numpy().astype(np.float32, copy=False),
    }