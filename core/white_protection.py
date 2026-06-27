from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class WhiteProtectionStats:
    """Summary of protected and removed white regions."""

    white_detected: int
    white_protected: int
    white_removed: int
    protected_percent: float
    possible_detail_loss: bool

    def to_dict(self) -> dict[str, object]:
        """Return serializable protection statistics."""
        return asdict(self)


def detect_white_details(img: Image.Image, level: str = "normal") -> dict[str, object]:
    """Detect white pixels that behave like artwork details."""
    rgba = img.convert("RGBA")
    mask = build_protection_mask(rgba, level=level)
    white = _white_candidate(rgba, level)
    stats = _stats(white, mask, np.zeros_like(mask))
    return {"mask": mask, "stats": stats.to_dict()}


def build_protection_mask(img: Image.Image, level: str = "normal") -> np.ndarray:
    """Build a boolean mask for internal white details."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    white = _white_candidate(rgba, level)
    if not np.any(white):
        return white

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 45, 135) > 0
    texture = cv2.Laplacian(gray, cv2.CV_32F)
    textured = np.abs(texture) > _texture_threshold(level)
    colored = (hsv[:, :, 1] > _saturation_threshold(level)) & (alpha > 10) & ~white
    dark_lines = (gray < 92) & (alpha > 10)
    context = _dilate(colored | dark_lines | edges | textured, _context_radius(level))

    count, labels = cv2.connectedComponents(white.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(white, dtype=bool)

    areas = np.bincount(labels.ravel(), minlength=count).astype(np.float32)
    touches = _edge_touch_labels(labels, count)
    context_counts = np.bincount(labels.ravel(), weights=context.ravel(), minlength=count)
    edge_counts = np.bincount(labels.ravel(), weights=edges.ravel(), minlength=count)
    context_ratio = context_counts / np.maximum(areas, 1)
    edge_ratio = edge_counts / np.maximum(areas, 1)

    small_contextual = (areas <= _max_context_area(level)) & (context_ratio > _context_ratio(level))
    thin_detail = (areas <= _max_context_area(level)) & (edge_ratio > 0.004)
    protected_labels = (~touches) & (small_contextual | thin_detail | (areas > 0))
    protected_labels[0] = False
    return white & protected_labels[labels]


def merge_protection(original: Image.Image, processed: Image.Image, mask: np.ndarray) -> Image.Image:
    """Restore protected white detail pixels from the original image."""
    base = processed.convert("RGBA")
    src = np.array(original.convert("RGBA"))
    out = np.array(base)
    valid = mask & (src[:, :, 3] > 0)
    out[valid] = src[valid]
    return Image.fromarray(out, "RGBA")


def protect_white_regions(
    original: Image.Image,
    processed: Image.Image,
    level: str = "normal",
) -> tuple[Image.Image, np.ndarray, WhiteProtectionStats]:
    """Restore protected white details after background removal."""
    mask = build_protection_mask(original, level=level)
    original_white = _white_candidate(original, level)
    before_alpha = np.array(original.convert("RGBA").getchannel("A"))
    after_alpha = np.array(processed.convert("RGBA").getchannel("A"))
    removed = original_white & (before_alpha > 20) & (after_alpha <= 20)
    protected_removed = removed & mask
    restored = merge_protection(original, processed, protected_removed)
    stats = _stats(original_white, mask, removed & ~mask)
    return restored, mask, stats


def mask_preview(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Create a green RGBA preview for protected white details."""
    preview = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    preview[mask] = (0, 220, 120, 210)
    img = Image.fromarray(preview, "RGBA")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return img


def _white_candidate(img: Image.Image, level: str) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    threshold = {"suave": 246, "normal": 235, "maxima": 220, "maxima_auto": 220}.get(level, 235)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return (rgb.min(axis=2) >= threshold) & (chroma <= 42) & (alpha > 20)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _edge_touch_labels(labels: np.ndarray, count: int) -> np.ndarray:
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    touches = np.zeros(count, dtype=bool)
    touches[np.unique(border)] = True
    return touches


def _stats(white: np.ndarray, protected: np.ndarray, removed: np.ndarray, possible_loss: bool = False) -> WhiteProtectionStats:
    total = int(np.count_nonzero(white))
    kept = int(np.count_nonzero(protected))
    gone = int(np.count_nonzero(removed))
    percent = round((kept / total * 100) if total else 0.0, 2)
    return WhiteProtectionStats(total, kept, gone, percent, possible_loss)


def _context_radius(level: str) -> int:
    return {"suave": 2, "normal": 4, "maxima": 7, "maxima_auto": 7}.get(level, 4)


def _context_ratio(level: str) -> float:
    return {"suave": 0.025, "normal": 0.014, "maxima": 0.004, "maxima_auto": 0.004}.get(level, 0.014)


def _texture_threshold(level: str) -> float:
    return {"suave": 20.0, "normal": 14.0, "maxima": 8.0, "maxima_auto": 8.0}.get(level, 14.0)


def _saturation_threshold(level: str) -> int:
    return {"suave": 40, "normal": 28, "maxima": 16, "maxima_auto": 16}.get(level, 28)


def _max_context_area(level: str) -> int:
    return {"suave": 1200, "normal": 2400, "maxima": 4800, "maxima_auto": 4800}.get(level, 2400)
