from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from core.artwork_mask import _border_color


def confirm_background_mask(
    image: Image.Image,
    artwork_mask: np.ndarray,
    main_bbox: tuple[int, int, int, int],
    tolerance: int = 42,
) -> np.ndarray:
    """Confirm only uniform low-detail background connected to canvas edges."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bg = _border_color(rgb)
    distance = np.linalg.norm(rgb.astype(np.int16) - bg.astype(np.int16), axis=2)
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) < 9
    edges = cv2.Canny(gray, 40, 135) == 0
    low_saturation = hsv[:, :, 1] < 80
    candidate = (distance <= tolerance) & texture & edges & low_saturation & (alpha > 0) & ~artwork_mask
    bbox_guard = _expanded_bbox_guard(candidate.shape, main_bbox)
    candidate &= ~bbox_guard
    reachable = _reachable_from_edges(candidate)
    return reachable & ~artwork_mask


def _expanded_bbox_guard(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = np.zeros(shape, dtype=bool)
    out[y0:y1, x0:x1] = True
    return out


def _reachable_from_edges(mask: np.ndarray) -> np.ndarray:
    work = mask.astype(np.uint8) * 255
    flood = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    h, w = mask.shape
    for point in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if work[point[1], point[0]]:
            cv2.floodFill(work, flood, point, 128)
    return work == 128
