from __future__ import annotations

from typing import List, Optional, Tuple

from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
)

# Spatial dims are (Z, H, W) because tensors are (C, Z, H, W)
_SPATIAL_AUGS = {"flip", "translate", "rotate_scale"}
_INTENSITY_AUGS = {"noise", "blur", "shift", "scale", "contrast"}

ALL_AUGS = sorted(_SPATIAL_AUGS | _INTENSITY_AUGS)


def build_train_transforms_3d(
    spatial_hw: Tuple[int, int] = (256, 256),
    enabled_augs: Optional[List[str]] = None,
    seed: Optional[int] = None,
    flip_prob: float = 0.5,
    translate_prob: float = 0.15,
    rotate_scale_prob: float = 0.15,
    noise_prob: float = 0.10,
    blur_prob: float = 0.20,
    shift_prob: float = 0.15,
    scale_prob: float = 0.15,
    contrast_prob: float = 0.20,
    translate_range: Tuple[float, float] = (20.0, 20.0),
    rotate_range: float = 0.350,
    scale_range: Tuple[float, float] = (0.2, 0.2),
) -> Optional[Compose]:
    """
    Expects sample dicts with:
      image: (1, Z, H, W)
      mask:  (1, Z, H, W)

    Applies the same spatial transform to image and mask.
    We keep z augmentation off by setting z-components to 0.
    """
    if not enabled_augs:
        return None

    enabled = set(enabled_augs)
    tfms = []

    # Flip only in H/W, not Z
    if "flip" in enabled:
        tfms += [
            RandFlipd(keys=("image", "mask"), prob=flip_prob, spatial_axis=1),  # H
            RandFlipd(keys=("image", "mask"), prob=flip_prob, spatial_axis=2),  # W
        ]

    # Translate only in H/W
    if "translate" in enabled:
        tfms.append(
            RandAffined(
                keys=("image", "mask"),
                prob=translate_prob,
                translate_range=(0.0, translate_range[0], translate_range[1]),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                spatial_size=(None, spatial_hw[0], spatial_hw[1]),
            )
        )

    # Rotate/scale only in-plane
    # Rotation here is around the Z axis, so H/W plane rotates together
    if "rotate_scale" in enabled:
        tfms.append(
            RandAffined(
                keys=("image", "mask"),
                prob=rotate_scale_prob,
                rotate_range=(rotate_range, 0.0, 0.0),
                scale_range=(0.0, scale_range[0], scale_range[1]),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                spatial_size=(None, spatial_hw[0], spatial_hw[1]),
            )
        )

    # Intensity transforms on image only
    if "noise" in enabled:
        tfms.append(
            RandGaussianNoised(keys="image", prob=noise_prob, mean=0.0, std=0.03)
        )

    if "blur" in enabled:
        tfms.append(
            RandGaussianSmoothd(
                keys="image",
                prob=blur_prob,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.0, 0.0),  # no blur across Z
            )
        )

    if "shift" in enabled:
        tfms.append(
            RandShiftIntensityd(keys="image", prob=shift_prob, offsets=(-0.05, 0.05))
        )

    if "scale" in enabled:
        tfms.append(
            RandScaleIntensityd(keys="image", prob=scale_prob, factors=(0.75, 1.25))
        )

    if "contrast" in enabled:
        tfms.append(
            RandAdjustContrastd(keys="image", prob=contrast_prob, gamma=(0.7, 1.5))
        )

    compose = Compose(tfms)
    if seed is not None:
        compose.set_random_state(seed=seed)

    return compose


def build_val_transforms_3d() -> None:
    return None