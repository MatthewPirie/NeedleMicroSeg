# src/data/augmentations_2d.py

from __future__ import annotations

from typing import List, Optional, Tuple

from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    Rand2DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
)

# Maps name -> (transform_class, default_kwargs)
_SPATIAL_AUGS = {"flip", "affine", "elastic"}
_INTENSITY_AUGS = {"noise", "blur", "shift", "scale", "contrast"}

ALL_AUGS = sorted(_SPATIAL_AUGS | _INTENSITY_AUGS)


def build_train_transforms_2d(
    spatial_hw: Tuple[int, int] = (256, 256),
    enabled_augs: Optional[List[str]] = None,
    seed: Optional[int] = None,
    # per-aug probability overrides (passed via aug_kwargs in config)
    flip_prob: float = 0.5,
    affine_prob: float = 0.15,
    elastic_prob: float = 0.10,
    noise_prob: float = 0.10,
    blur_prob: float = 0.20,
    shift_prob: float = 0.15,
    scale_prob: float = 0.15,
    contrast_prob: float = 0.20,
) -> Optional[Compose]:
    """
    Build a MONAI Compose for training augmentation.

    Expects sample dicts with keys ``"image"`` and ``"mask"`` (numpy arrays
    or tensors of shape (1, H, W)).  Spatial transforms are applied to both;
    intensity transforms to ``"image"`` only.

    Args:
        spatial_hw: (H, W) output spatial size for RandAffined.
        enabled_augs: subset of ALL_AUGS to include.  Pass ``None`` or an
            empty list to get an identity (no augmentation) transform.
        seed: optional random seed for reproducibility.
        flip_prob … contrast_prob: per-transform application probabilities.
    """
    if not enabled_augs:
        return None

    enabled = set(enabled_augs)
    tfms = []

    # ── Spatial ──────────────────────────────────────────────────────────────
    if "flip" in enabled:
        tfms += [
            RandFlipd(keys=("image", "mask"), prob=flip_prob, spatial_axis=1),
            RandFlipd(keys=("image", "mask"), prob=flip_prob, spatial_axis=0),
        ]

    if "affine" in enabled:
        tfms.append(
            RandAffined(
                keys=("image", "mask"),
                prob=affine_prob,
                rotate_range=(0.350,),
                scale_range=(0.2, 0.2),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                spatial_size=spatial_hw,
            )
        )

    if "elastic" in enabled:
        tfms.append(
            Rand2DElasticd(
                keys=("image", "mask"),
                spacing=(32, 32),
                magnitude_range=(1, 3),
                prob=elastic_prob,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    # ── Intensity (image only) ────────────────────────────────────────────────
    if "noise" in enabled:
        tfms.append(RandGaussianNoised(keys="image", prob=noise_prob, mean=0.0, std=0.03))

    if "blur" in enabled:
        tfms.append(
            RandGaussianSmoothd(
                keys="image",
                prob=blur_prob,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
            )
        )

    if "shift" in enabled:
        tfms.append(RandShiftIntensityd(keys="image", prob=shift_prob, offsets=(-0.05, 0.05)))

    if "scale" in enabled:
        tfms.append(RandScaleIntensityd(keys="image", prob=scale_prob, factors=(0.75, 1.25)))

    if "contrast" in enabled:
        tfms.append(RandAdjustContrastd(keys="image", prob=contrast_prob, gamma=(0.7, 1.5)))

    compose = Compose(tfms)
    if seed is not None:
        compose.set_random_state(seed=seed)

    return compose


def build_val_transforms_2d() -> None:
    """No augmentation at validation time."""
    return None
