# src/data/extractors_3d/__init__.py

from __future__ import annotations

from .temporal_window import extract as temporal_window


_EXTRACTORS = {
    "temporal_window": temporal_window,
}


def get_extractor(name: str):
    if name not in _EXTRACTORS:
        available = ", ".join(sorted(_EXTRACTORS))
        raise KeyError(f"Unknown 3D extractor: {name}. Available: {available}")
    return _EXTRACTORS[name]


def available_extractors():
    return sorted(_EXTRACTORS.keys())