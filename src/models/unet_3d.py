# src/models/unet_3d.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

try:
    from monai.networks.nets import DynUNet
    _HAS_DYNUNET = True
except Exception:
    DynUNet = None
    _HAS_DYNUNET = False

from monai.networks.nets import UNet


_UNET_3D_VARIANTS: Dict[str, Dict[str, Any]] = {
    "base": {
        "filters": (32, 64, 128, 256, 320, 320, 320),
        "kernel_sizes": (
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        "strides": (
            (1, 1, 1),
            (1, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
        "norm": "INSTANCE",
        "num_res_units": 0,
        "deep_supervision": False,
    },
    "small": {
        "filters": (32, 64, 128, 256, 320),
        "kernel_sizes": (
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        "strides": (
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
        "norm": "INSTANCE",
        "num_res_units": 0,
        "deep_supervision": False,
    },
        "base_no_z_downsample": {
        "filters": (32, 64, 128, 256, 320, 320, 320),
        "kernel_sizes": (
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        "strides": (
            (1, 1, 1),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
        "norm": "INSTANCE",
        "num_res_units": 0,
        "deep_supervision": False,
    },
}


def available_unet_3d_variants():
    return sorted(_UNET_3D_VARIANTS.keys())


def build_unet_3d(
    in_channels: int = 1,
    out_channels: int = 1,
    variant: str = "base",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if variant not in _UNET_3D_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Options: {available_unet_3d_variants()}"
        )

    cfg = _UNET_3D_VARIANTS[variant]

    if _HAS_DYNUNET:
        model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=list(cfg["kernel_sizes"]),
            strides=list(cfg["strides"]),
            upsample_kernel_size=list(cfg["strides"][1:]),
            filters=list(cfg["filters"]),
            norm_name=cfg["norm"],
            deep_supervision=bool(cfg.get("deep_supervision", False)),
            res_block=False,
        )
        model_name = "monai_dynunet_3d"
    else:
        try:
            model = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=cfg["filters"],
                strides=cfg["strides"][1:],
                num_res_units=cfg["num_res_units"],
                norm=cfg["norm"],
            )
        except Exception:
            isotropic_strides = (2,) * (len(cfg["filters"]) - 1)
            model = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=cfg["filters"],
                strides=isotropic_strides,
                num_res_units=cfg["num_res_units"],
                norm=cfg["norm"],
            )
        model_name = "monai_unet_3d"

    meta: Dict[str, Any] = {
        "model_name": model_name,
        "model_variant": variant,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "filters": tuple(cfg["filters"]),
        "kernel_sizes": tuple(cfg["kernel_sizes"]),
        "strides": tuple(cfg["strides"]),
        "norm": cfg["norm"],
        "num_res_units": cfg["num_res_units"],
        "deep_supervision": bool(cfg.get("deep_supervision", False)),
        "uses_dynunet": _HAS_DYNUNET,
    }
    return model, meta
