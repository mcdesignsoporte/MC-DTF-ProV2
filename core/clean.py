from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def clean_alpha(img: Image.Image, alpha_cut: int = 70, despeckle_area: int = 2, edge_contract: int = 0) -> Image.Image:
    """Clean alpha using vectorized NumPy and OpenCV operations."""
    arr = np.array(img.convert("RGBA"))
    alpha = threshold_alpha(arr[:, :, 3], alpha_cut)
    alpha = despeckle(alpha, min_area=despeckle_area)
    alpha = contract_edge(alpha, iterations=edge_contract)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def threshold_alpha(alpha: np.ndarray, alpha_cut: int) -> np.ndarray:
    """Convert semi-transparent pixels to solid or transparent."""
    return np.where(alpha >= alpha_cut, 255, 0).astype(np.uint8)


def despeckle(alpha: np.ndarray, min_area: int = 2) -> np.ndarray:
    """Remove connected alpha islands smaller than min_area."""
    if min_area <= 1:
        return alpha
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def contract_edge(alpha: np.ndarray, iterations: int = 0) -> np.ndarray:
    """Contract alpha edges when the user explicitly asks for it."""
    if iterations <= 0:
        return alpha
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(alpha, kernel, iterations=iterations)


def trim_transparent(img: Image.Image, padding: int = 20) -> Image.Image:
    """Trim empty transparent space while preserving requested padding."""
    rgba = img.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        return rgba
    left, top, right, bottom = bbox
    crop = (
        max(0, left - padding),
        max(0, top - padding),
        min(rgba.width, right + padding),
        min(rgba.height, bottom + padding),
    )
    return rgba.crop(crop)
