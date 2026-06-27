"""Compatibility API for the professional non-destructive engine."""

from __future__ import annotations

import numpy as np
from PIL import Image

from core.artwork_mask import build_artwork_mask
from core.background_confirm import confirm_background_mask
from core.non_destructive_clean import restore_artwork_pixels, safe_despeckle as _engine_safe_despeckle


def build_protected_art_mask(image: Image.Image) -> np.ndarray:
    """Return a conservative protected-art mask."""
    return build_artwork_mask(image).artwork_mask


def safe_background_removal(image: Image.Image, tolerance: int = 42) -> tuple[Image.Image, np.ndarray]:
    """Remove only confirmed background outside protected art."""
    rgba = image.convert("RGBA")
    artwork = build_artwork_mask(rgba)
    background = confirm_background_mask(rgba, artwork.artwork_mask, artwork.main_bbox, tolerance=tolerance)
    arr = np.array(rgba)
    arr[background, 3] = 0
    return Image.fromarray(arr, "RGBA"), background


def safe_despeckle(alpha: np.ndarray, artwork_mask: np.ndarray, min_area: int = 8, rgb: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Remove isolated noise outside protected art."""
    if rgb is None:
        rgb = np.zeros((alpha.shape[0], alpha.shape[1], 3), dtype=np.uint8)
    return _engine_safe_despeckle(alpha, artwork_mask, rgb, min_area=min_area)


def restore_protected_pixels(original: Image.Image, result: Image.Image, protected_mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Restore pixels removed inside the protected mask."""
    return restore_artwork_pixels(original, result, protected_mask)


__all__ = [
    "build_protected_art_mask",
    "safe_background_removal",
    "safe_despeckle",
    "restore_protected_pixels",
]
