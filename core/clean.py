from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FineDetailStats:
    """Statistics for safe fine-detail cleanup."""

    fine_details_detected: int
    details_protected: int
    noise_removed: int
    components_removed: int

    def to_dict(self) -> dict[str, int]:
        """Return serializable cleanup statistics."""
        return asdict(self)


def clean_alpha(
    img: Image.Image,
    alpha_cut: int = 70,
    despeckle_area: int = 2,
    edge_contract: int = 0,
    protect_details: bool = True,
    fine_detail_level: str = "maxima",
) -> Image.Image:
    """Clean alpha using vectorized NumPy and OpenCV operations."""
    result, _, _ = clean_alpha_with_stats(
        img,
        alpha_cut=alpha_cut,
        despeckle_area=despeckle_area,
        edge_contract=edge_contract,
        protect_details=protect_details,
        fine_detail_level=fine_detail_level,
    )
    return result


def clean_alpha_with_stats(
    img: Image.Image,
    alpha_cut: int = 70,
    despeckle_area: int = 2,
    edge_contract: int = 0,
    protect_details: bool = True,
    fine_detail_level: str = "maxima",
) -> tuple[Image.Image, FineDetailStats, np.ndarray]:
    """Clean alpha while protecting valid fine details."""
    arr = np.array(img.convert("RGBA"))
    alpha = threshold_alpha(arr[:, :, 3], alpha_cut)
    if protect_details:
        alpha, detail_mask, stats = safe_despeckle(arr, alpha, min_area=despeckle_area, level=fine_detail_level)
    else:
        before = np.count_nonzero(alpha)
        alpha = despeckle(alpha, min_area=despeckle_area)
        detail_mask = np.zeros(alpha.shape, dtype=bool)
        stats = FineDetailStats(0, 0, int(before - np.count_nonzero(alpha)), 0)
    alpha = contract_edge(alpha, iterations=edge_contract)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA"), stats, detail_mask


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


def safe_despeckle(
    rgba: np.ndarray,
    alpha: np.ndarray,
    min_area: int = 2,
    level: str = "maxima",
) -> tuple[np.ndarray, np.ndarray, FineDetailStats]:
    """Remove only isolated noise while preserving detail connected to artwork."""
    if min_area <= 1:
        detail_mask = protect_fine_details(rgba, alpha, level=level)
        stats = FineDetailStats(int(np.count_nonzero(detail_mask)), int(np.count_nonzero(detail_mask)), 0, 0)
        return alpha, detail_mask, stats
    detail_mask = protect_fine_details(rgba, alpha, level=level)
    cleaned, removed_mask, components = remove_only_isolated_noise(alpha, detail_mask, min_area=min_area)
    stats = FineDetailStats(
        fine_details_detected=int(np.count_nonzero(detail_mask)),
        details_protected=int(np.count_nonzero(detail_mask & (alpha > 0))),
        noise_removed=int(np.count_nonzero(removed_mask)),
        components_removed=int(components),
    )
    return cleaned, detail_mask, stats


def build_main_artwork_mask(alpha: np.ndarray, level: str = "maxima") -> np.ndarray:
    """Detect the main artwork component and expand its protection zone."""
    solid = alpha > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid.astype(np.uint8), 8)
    if count <= 1:
        return solid
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = int(np.argmax(areas) + 1)
    main = labels == main_label
    if not np.any(main):
        return solid
    x, y, w, h = stats[main_label, :4]
    pad = _main_padding(level, alpha.shape)
    bbox = np.zeros_like(main, dtype=bool)
    y0, y1 = max(0, y - pad), min(alpha.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(alpha.shape[1], x + w + pad)
    bbox[y0:y1, x0:x1] = True
    size = _bounded_dilate_size(level, alpha.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    expanded = cv2.dilate(main.astype(np.uint8), kernel, iterations=1).astype(bool)
    return expanded | bbox


def protect_fine_details(rgba: np.ndarray, alpha: np.ndarray, level: str = "maxima") -> np.ndarray:
    """Build a mask for valid fine details that must survive cleanup."""
    rgb = rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 35, 130) > 0
    saturated = hsv[:, :, 1] > _saturation_threshold(level)
    contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) > _contrast_threshold(level)
    main_zone = build_main_artwork_mask(alpha, level=level)
    candidate = (alpha > 0) & main_zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nearby_edges = cv2.dilate((edges | saturated | contrast).astype(np.uint8), kernel, iterations=1).astype(bool)
    return candidate | ((alpha > 0) & nearby_edges & main_zone)


def remove_only_isolated_noise(
    alpha: np.ndarray,
    protected_mask: np.ndarray,
    min_area: int = 2,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove tiny components only when isolated from protected artwork."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    if count <= 1:
        return alpha, np.zeros_like(alpha, dtype=bool), 0
    areas = stats[:, cv2.CC_STAT_AREA]
    protected_counts = np.bincount(labels.ravel(), weights=protected_mask.ravel(), minlength=count)
    remove_labels = (areas < min_area) & (protected_counts == 0)
    remove_labels[0] = False
    removed = remove_labels[labels]
    out = alpha.copy()
    out[removed] = 0
    return out, removed, int(np.count_nonzero(remove_labels))


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


def _dilate_size(level: str) -> int:
    return {"muy suave": 17, "suave": 23, "normal": 31, "fuerte": 41, "maxima": 45}.get(level, 41)


def _bounded_dilate_size(level: str, shape: tuple[int, int]) -> int:
    limit = max(3, min(shape) // 3)
    if limit % 2 == 0:
        limit -= 1
    size = min(_dilate_size(level), limit)
    return max(3, size if size % 2 else size - 1)


def _main_padding(level: str, shape: tuple[int, int]) -> int:
    base = max(1, min(shape) // 24)
    factor = {"muy suave": 1, "suave": 2, "normal": 3, "fuerte": 4, "maxima": 4}.get(level, 4)
    return min(base * factor, max(1, min(shape) // 6))


def _saturation_threshold(level: str) -> int:
    return {"muy suave": 80, "suave": 58, "normal": 38, "fuerte": 28, "maxima": 22}.get(level, 22)


def _contrast_threshold(level: str) -> float:
    return {"muy suave": 36.0, "suave": 26.0, "normal": 18.0, "fuerte": 12.0, "maxima": 10.0}.get(level, 10.0)
