from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ArtworkMaskResult:
    """Main artwork protection data."""

    artwork_mask: np.ndarray
    main_bbox: tuple[int, int, int, int]
    confidence_map: np.ndarray


def build_artwork_mask(image: Image.Image) -> ArtworkMaskResult:
    """Build a conservative artwork mask that favors preserving design pixels."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bg = _border_color(rgb)
    color_distance = np.linalg.norm(rgb.astype(np.int16) - bg.astype(np.int16), axis=2)
    edges = cv2.Canny(gray, 35, 130) > 0
    saturation = hsv[:, :, 1] > 26
    contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) > 10
    alpha_signal = alpha > 20
    non_background = color_distance > 34
    seed = alpha_signal & (non_background | saturation | edges | contrast)
    if np.count_nonzero(seed) < max(12, seed.size // 5000):
        seed = alpha_signal & (non_background | edges)
    main = _main_component(seed)
    bbox = _expanded_bbox(main, seed.shape)
    bbox_mask = _bbox_mask(seed.shape, bbox)
    near_main = _dilate(main, max(5, min(seed.shape) // 24))
    detail_signal = alpha_signal & (edges | saturation | contrast | non_background)
    detail_near = detail_signal & (bbox_mask | near_main)
    artwork = _dilate(main | detail_near, 3) | detail_near
    confidence = np.zeros(seed.shape, dtype=np.float32)
    confidence[artwork] = 0.85
    confidence[detail_near] = 1.0
    return ArtworkMaskResult(artwork.astype(bool), bbox, confidence)


def mask_to_preview(mask: np.ndarray, color: tuple[int, int, int, int]) -> Image.Image:
    """Convert a boolean mask to an RGBA overlay."""
    arr = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    arr[mask] = color
    return Image.fromarray(arr, "RGBA")


def _main_component(seed: np.ndarray) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    if count <= 1:
        return seed
    h, w = seed.shape
    center = np.array([w / 2, h / 2], dtype=np.float32)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    distances = np.linalg.norm(centroids - center, axis=1)
    scores = areas - distances * max(1.0, areas.max()) / max(h, w)
    scores[0] = -1
    return labels == int(np.argmax(scores))


def _expanded_bbox(mask: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, shape[1], shape[0])
    pad = max(8, min(shape) // 10)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(shape[1], int(xs.max()) + pad + 1)
    y1 = min(shape[0], int(ys.max()) + pad + 1)
    return (x0, y0, x1, y1)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = np.zeros(shape, dtype=bool)
    out[y0:y1, x0:x1] = True
    return out


def _border_color(rgb: np.ndarray, border: int = 16) -> np.ndarray:
    h, w, _ = rgb.shape
    b = max(2, min(border, h // 4, w // 4))
    samples = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3), rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    quantized = (samples // 16) * 16
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    return colors[int(np.argmax(counts))]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
