from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from core.artwork_mask import ArtworkMaskResult, build_artwork_mask
from core.background_confirm import confirm_background_mask


@dataclass(frozen=True)
class NonDestructiveResult:
    """Processed image, QA masks, and conservative cleanup metrics."""

    image: Image.Image
    artwork_mask: np.ndarray
    background_mask: np.ndarray
    doubtful_mask: np.ndarray
    restored_mask: np.ndarray
    risk: dict[str, object]
    stats: dict[str, object]


def non_destructive_clean(
    image: Image.Image,
    min_area: int = 8,
    background_tolerance: int = 42,
    safe_mode: bool = True,
) -> NonDestructiveResult:
    """Remove only confirmed background and isolated noise outside protected artwork."""
    original = image.convert("RGBA")
    artwork = build_artwork_mask(original)
    expanded_art = _expand(artwork.artwork_mask, max(3, min(original.size) // 80))
    background = confirm_background_mask(original, expanded_art, artwork.main_bbox, tolerance=background_tolerance)
    arr = np.array(original)
    out = arr.copy()
    out[background, 3] = 0
    alpha, noise = safe_despeckle(out[:, :, 3], expanded_art, arr[:, :, :3], min_area=min_area)
    out[:, :, 3] = alpha
    result = Image.fromarray(out, "RGBA")
    risk = estimate_art_loss_risk(original, result, expanded_art, artwork.main_bbox)
    restored_mask = np.zeros(expanded_art.shape, dtype=bool)
    if safe_mode and bool(risk["risk_detected"]):
        result, restored_mask = restore_artwork_pixels(original, result, expanded_art)
        risk = estimate_art_loss_risk(original, result, expanded_art, artwork.main_bbox)
    doubtful = expanded_art & ~background
    stats = {
        "restored_pixels": int(np.count_nonzero(restored_mask)),
        "background_removed": int(np.count_nonzero(background)),
        "artwork_protected": int(np.count_nonzero(expanded_art)),
        "noise_removed": int(np.count_nonzero(noise)),
    }
    return NonDestructiveResult(result, expanded_art, background, doubtful, restored_mask, risk, stats)


def safe_despeckle(
    alpha: np.ndarray,
    artwork_mask: np.ndarray,
    rgb: np.ndarray,
    min_area: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove isolated tiny alpha components only outside artwork and visual detail."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    if count <= 1 or min_area <= 1:
        return alpha, np.zeros_like(alpha, dtype=bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 35, 130) > 0
    saturated = hsv[:, :, 1] > 35
    near_art = _expand(artwork_mask, 9)
    areas = stats[:, cv2.CC_STAT_AREA]
    edge_counts = np.bincount(labels.ravel(), weights=edges.ravel(), minlength=count)
    sat_counts = np.bincount(labels.ravel(), weights=saturated.ravel(), minlength=count)
    art_counts = np.bincount(labels.ravel(), weights=near_art.ravel(), minlength=count)
    tiny_unsaturated = (areas <= min_area + 1) & (sat_counts == 0)
    tiny_edgeless = (areas <= min_area) & (edge_counts == 0)
    remove_labels = (tiny_unsaturated | tiny_edgeless) & (art_counts == 0)
    remove_labels[0] = False
    removed = remove_labels[labels]
    cleaned = alpha.copy()
    cleaned[removed] = 0
    return cleaned, removed


def estimate_art_loss_risk(
    original: Image.Image,
    result: Image.Image,
    artwork_mask: np.ndarray,
    main_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    """Estimate whether protected artwork pixels were removed."""
    src = np.array(original.convert("RGBA"))
    dst = np.array(result.convert("RGBA"))
    lost = artwork_mask & (src[:, :, 3] > 20) & (dst[:, :, 3] <= 20)
    lost_pixels = int(np.count_nonzero(lost))
    art_pixels = max(1, int(np.count_nonzero(artwork_mask)))
    gray = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2HSV)
    edge_loss = int(np.count_nonzero(lost & (cv2.Canny(gray, 35, 130) > 0)))
    sat_loss = int(np.count_nonzero(lost & (hsv[:, :, 1] > 35)))
    bbox_loss = int(np.count_nonzero(lost & _bbox_mask(lost.shape, main_bbox))) if main_bbox else 0
    percent = round(lost_pixels / art_pixels * 100, 3)
    risk = percent > 1.0 or edge_loss > 0 or sat_loss > 0 or bbox_loss > 0
    return {
        "risk_detected": bool(risk),
        "lost_pixels": lost_pixels,
        "lost_percent": percent,
        "edge_loss": edge_loss,
        "saturated_loss": sat_loss,
        "bbox_loss": bbox_loss,
    }


def restore_artwork_pixels(original: Image.Image, result: Image.Image, artwork_mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Restore removed artwork pixels from the original image with solid alpha."""
    src = np.array(original.convert("RGBA"))
    out = np.array(result.convert("RGBA"))
    restore = artwork_mask & (src[:, :, 3] > 20) & (out[:, :, 3] <= 20)
    out[restore] = src[restore]
    out[restore, 3] = 255
    return Image.fromarray(out, "RGBA"), restore


def _expand(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if bbox is None:
        return out
    x0, y0, x1, y1 = bbox
    out[y0:y1, x0:x1] = True
    return out
