# src/utils/normalization.py

from __future__ import annotations

import numpy as np


def zscore_per_image(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    mean = float(image.mean())
    std = float(image.std())
    return (image - mean) / (std + eps)


def minmax_per_image(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo = float(image.min())
    hi = float(image.max())
    return (image - lo) / (hi - lo + eps)


def get_normalizer(name: str | None):
    if name is None or name == "none":
        return None
    if name == "zscore_per_image":
        return zscore_per_image
    if name == "minmax_per_image":
        return minmax_per_image
    raise ValueError(f"Unknown normalization name: {name}")