from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from core.detail_protect import detail_protection_mask, restore_protected_alpha


def dominant_background_color(img: Image.Image, border: int = 16) -> tuple[int, int, int]:
    """Estimate the background color from image borders."""
    rgb = np.array(img.convert("RGB"))
    h, w, _ = rgb.shape
    b = max(2, min(border, h // 4, w // 4))
    samples = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3), rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    quantized = (samples // 16) * 16
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    return tuple(int(v) for v in colors[int(np.argmax(counts))])


def background_uniformity(img: Image.Image, color: tuple[int, int, int], tolerance: int = 34) -> float:
    """Return how much of the border matches the estimated background."""
    rgb = np.array(img.convert("RGB"))
    h, w, _ = rgb.shape
    b = max(2, min(16, h // 4, w // 4))
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3), rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    distance = np.linalg.norm(border.astype(np.int16) - np.array(color, dtype=np.int16), axis=1)
    return float((distance <= tolerance).mean() * 100)


def remove_dominant_background(
    img: Image.Image,
    tolerance: int = 38,
    softness: int = 18,
    protect_details: bool = True,
    color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Remove solid, dominant, chroma, mildly noisy, or gradient-like backgrounds."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    bg = np.array(color or dominant_background_color(rgba), dtype=np.int16)
    distance = np.linalg.norm(rgb - bg, axis=2)
    candidate = distance <= tolerance
    reachable = _reachable_from_edges(candidate)
    if protect_details:
        protected = detail_protection_mask(rgba, threshold=24, radius=5) & ~candidate
        reachable = reachable & ~protected
    else:
        protected = np.zeros(reachable.shape, dtype=bool)
    new_alpha = _fade_alpha(alpha, distance, reachable, tolerance, softness)
    if protect_details:
        new_alpha = restore_protected_alpha(new_alpha, alpha, protected & ~reachable, strength=0.8)
    arr[:, :, 3] = new_alpha
    return Image.fromarray(arr, "RGBA")


def cleanup_light_background_residue(
    img: Image.Image,
    background_color: tuple[int, int, int] | None = None,
    tolerance: int = 62,
    alpha_threshold: int = 20,
) -> Image.Image:
    """Remove exterior light residue without touching enclosed white artwork."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    bg = np.array(background_color or dominant_background_color(rgba), dtype=np.int16)
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
    distance = np.linalg.norm(rgb - bg, axis=2)
    light_low_detail = (gray >= 210) & (hsv[:, :, 1] <= 42)
    near_background = distance <= tolerance
    candidate = (alpha > alpha_threshold) & light_low_detail & near_background
    traversable = candidate | (alpha <= alpha_threshold)
    exterior = _reachable_from_edges(traversable)
    remove_mask = candidate & exterior
    if not np.any(remove_mask):
        return rgba
    out = arr.copy()
    out[remove_mask, 3] = 0
    return Image.fromarray(out, "RGBA")


def light_residue_percent(
    img: Image.Image,
    background_color: tuple[int, int, int] | None = None,
    tolerance: int = 62,
) -> float:
    """Return the visible exterior light residue percentage for QA tests."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    bg = np.array(background_color or dominant_background_color(rgba), dtype=np.int16)
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
    distance = np.linalg.norm(rgb - bg, axis=2)
    residue = (alpha > 20) & (gray >= 210) & (hsv[:, :, 1] <= 42) & (distance <= tolerance)
    exterior = _reachable_from_edges(residue | (alpha <= 20))
    return float((residue & exterior).mean() * 100)


def cleanup_light_edge_matte(
    img: Image.Image,
    background_color: tuple[int, int, int] | None = None,
    tolerance: int = 70,
    band_px: int = 3,
) -> Image.Image:
    """Reduce white matte contamination only along the exterior alpha edge."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    band = _alpha_edge_band(alpha, band_px)
    if not np.any(band):
        return rgba

    rgb = arr[:, :, :3].astype(np.int16)
    bg = np.array(background_color or dominant_background_color(rgba), dtype=np.int16)
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
    distance = np.linalg.norm(rgb - bg, axis=2)
    matte = band & (alpha > 0) & (gray >= 180) & (hsv[:, :, 1] <= 64)
    matte &= (distance <= tolerance) | ((alpha < 235) & (gray >= 170))
    if not np.any(matte):
        return rgba

    out = arr.copy()
    strong = matte & ((distance <= tolerance * 0.72) | (gray >= 220))
    weak = matte & ~strong
    out[strong, 3] = 0
    out[weak, 3] = np.minimum(out[weak, 3], (out[weak, 3].astype(np.float32) * 0.35).astype(np.uint8))
    return Image.fromarray(out, "RGBA")


def edge_light_residue_score(img: Image.Image, band_px: int = 3) -> float:
    """Measure visible light matte in the alpha edge when composited on black."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    band = _alpha_edge_band(alpha, band_px)
    total = int(np.count_nonzero(band))
    if total == 0:
        return 0.0

    alpha_f = alpha.astype(np.float32) / 255.0
    comp = (arr[:, :, :3].astype(np.float32) * alpha_f[:, :, None]).astype(np.uint8)
    gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
    residue = band & (alpha > 0) & (gray >= 84) & (hsv[:, :, 1] <= 72)
    return round(float(np.count_nonzero(residue) / total * 100), 3)


def remove_background_opencv(img: Image.Image, protect_details: bool = True) -> Image.Image:
    """Segment non-uniform backgrounds using GrabCut seeded by image borders."""
    rgba = img.convert("RGBA")
    rgb = np.array(rgba.convert("RGB"))
    h, w, _ = rgb.shape
    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    pad = max(6, min(h, w) // 30)
    mask[:pad, :] = cv2.GC_BGD
    mask[-pad:, :] = cv2.GC_BGD
    mask[:, :pad] = cv2.GC_BGD
    mask[:, -pad:] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if protect_details:
        foreground = np.maximum(foreground, detail_protection_mask(rgba).astype(np.uint8) * 255)
    out = rgba.copy()
    out.putalpha(Image.fromarray(foreground, "L"))
    return out


def _reachable_from_edges(mask: np.ndarray) -> np.ndarray:
    work = mask.astype(np.uint8) * 255
    flood = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    h, w = mask.shape
    for point in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if work[point[1], point[0]]:
            cv2.floodFill(work, flood, point, 128)
    return work == 128


def _alpha_edge_band(alpha: np.ndarray, radius: int) -> np.ndarray:
    opaque = alpha > 20
    transparent = alpha <= 20
    if not np.any(opaque) or not np.any(transparent):
        return np.zeros_like(alpha, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    near_transparent = cv2.dilate(transparent.astype(np.uint8), kernel, iterations=1).astype(bool)
    return opaque & near_transparent


def _fade_alpha(alpha: np.ndarray, distance: np.ndarray, remove_mask: np.ndarray, tolerance: int, softness: int) -> np.ndarray:
    out = alpha.astype(np.float32).copy()
    out[remove_mask] = 0
    halo = (~remove_mask) & (distance < tolerance + softness)
    fade = np.clip((distance - tolerance) / max(1, softness), 0, 1)
    out[halo] *= fade[halo]
    return np.clip(out, 0, 255).astype(np.uint8)
