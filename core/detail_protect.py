from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def detail_protection_mask(img: Image.Image, threshold: int = 24, radius: int = 5) -> np.ndarray:
    """Find letters, white splashes, smoke, logos, and thin contours near dark pixels."""
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = gray > max(threshold + 35, 62)
    edges = cv2.Canny(gray, 45, 130) > 0
    low_saturation = (rgb.max(axis=2) - rgb.min(axis=2)) < 38
    smoke = (gray > threshold + 12) & (gray < 170) & low_saturation
    detail = bright | edges | smoke
    kernel = np.ones((max(3, radius), max(3, radius)), np.uint8)
    return cv2.dilate(detail.astype(np.uint8), kernel, iterations=1) > 0


def restore_protected_alpha(base_alpha: np.ndarray, original_alpha: np.ndarray, mask: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Restore alpha in protected regions without making the whole black background opaque."""
    out = base_alpha.astype(np.float32).copy()
    original = original_alpha.astype(np.float32)
    out[mask] = np.maximum(out[mask], original[mask] * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def keep_connected_dark_details(remove_mask: np.ndarray, protect_mask: np.ndarray) -> np.ndarray:
    """Avoid removing dark components that touch detected details."""
    num_labels, labels = cv2.connectedComponents(remove_mask.astype(np.uint8), 8)
    safe_remove = remove_mask.copy()
    for label in range(1, num_labels):
        component = labels == label
        if np.any(component & protect_mask):
            safe_remove[component] = False
    return safe_remove
