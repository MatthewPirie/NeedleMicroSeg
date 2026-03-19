from __future__ import annotations

from typing import Callable, Dict, List

ExtractorFn = Callable[..., dict]


def _registry() -> Dict[str, ExtractorFn]:
    # Lazy imports help avoid circular imports
    from .full_frame_resize import extract as full_frame_resize

    return {
        "full_frame_resize": full_frame_resize,
    }


def available_extractors() -> List[str]:
    return sorted(_registry().keys())


def get_extractor(name: str) -> ExtractorFn:
    reg = _registry()
    if name not in reg:
        raise ValueError(
            f"Unknown extractor '{name}'. Options: {available_extractors()}"
        )
    return reg[name]