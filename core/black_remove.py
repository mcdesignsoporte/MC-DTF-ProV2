from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from core.detail_protect import detail_protection_mask, restore_protected_alpha


def _black_candidates(rgb: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    maxc = rgb.max(axis=2).astype(np.float32)
    minc = rgb.min(axis=2).astype(np.float32)
    chroma = maxc - minc
    pure = maxc <= threshold
    near = (maxc <= threshold + 18) & (chroma <= 16)
    return pure, near


def _background_reachable(mask: np.ndarray) -> np.ndarray:
    flood = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    work = mask.astype(np.uint8) * 255
    h, w = mask.shape
    for point in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if work[point[1], point[0]]:
            cv2.floodFill(work, flood, point, 128)
    return work == 128


def _soft_alpha(alpha: np.ndarray, rgb: np.ndarray, remove_mask: np.ndarray, threshold: int, softness: int) -> np.ndarray:
    maxc = rgb.max(axis=2).astype(np.float32)
    fade = np.clip((maxc - threshold) / max(1, softness), 0, 1)
    out = alpha.astype(np.float32).copy()
    out[remove_mask] = 0
    halo = (~remove_mask) & (maxc < threshold + softness)
    out[halo] = out[halo] * fade[halo]
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_black_background(img: Image.Image, threshold: int = 24, softness: int = 14, protect_details: bool = True) -> Image.Image:
    """Remove pure background black while preserving letters, outlines, shadows, and artwork detail."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    pure_black, near_black = _black_candidates(rgb, threshold)
    reachable = _background_reachable(near_black)
    remove_mask = pure_black & reachable

    if protect_details:
        protected = detail_protection_mask(rgba, threshold=threshold, radius=5)
        remove_mask = remove_mask & ~protected
    else:
        protected = np.zeros(remove_mask.shape, dtype=bool)

    new_alpha = _soft_alpha(alpha, rgb, remove_mask, threshold, softness)
    if protect_details:
        new_alpha = restore_protected_alpha(new_alpha, alpha, protected & ~remove_mask, strength=0.75)

    arr[:, :, 3] = new_alpha
    return Image.fromarray(arr, "RGBA")
