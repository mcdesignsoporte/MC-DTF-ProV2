from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DTFPrepressSettings:
    """Settings for professional DTF prepress."""

    enable_dtf_prepress: bool = True
    remove_white_halo: bool = True
    remove_black_halo: bool = False
    halo_strength: str = "suave"
    expand_edge_px: int = 1
    bleed_px: int = 0
    create_cutline: bool = False
    min_printable_mm: float = 1.0
    dpi: int = 300


@dataclass(frozen=True)
class DTFPrepressResult:
    """Processed image and QA artifacts for DTF prepress."""

    image: Image.Image
    alpha_quality: dict[str, object]
    white_halo_mask: np.ndarray
    black_halo_mask: np.ndarray
    bleed_mask: np.ndarray
    cutline_mask: np.ndarray
    small_elements_mask: np.ndarray
    small_elements_report: dict[str, object]
    metadata: dict[str, object]


def analyze_alpha_quality(image: Image.Image) -> dict[str, object]:
    """Analyze transparency, halos, alpha noise, and small holes."""
    alpha = np.array(image.convert("RGBA").getchannel("A"))
    total = max(1, alpha.size)
    transparent = np.count_nonzero(alpha <= 5) / total * 100
    solid = np.count_nonzero(alpha >= 250) / total * 100
    semi = 100 - transparent - solid
    binary = alpha > 20
    border = _alpha_border(alpha)
    holes = _small_holes(binary)
    noise = _alpha_noise(alpha, binary)
    white_halo = detect_white_halo(image)
    black_halo = detect_black_halo(image)
    return {
        "transparent_percent": round(float(transparent), 2),
        "solid_percent": round(float(solid), 2),
        "semi_transparent_percent": round(float(semi), 2),
        "alpha_noise_score": round(float(noise), 2),
        "small_holes": int(holes),
        "has_halo_risk": bool(np.any(white_halo) or np.any(black_halo) or semi > 8),
    }


def detect_white_halo(image: Image.Image) -> np.ndarray:
    """Detect bright halo pixels only on the exterior alpha border."""
    arr = np.array(image.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    border = _outer_alpha_border(alpha)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return border & (gray > 210) & (chroma < 42)


def detect_black_halo(image: Image.Image) -> np.ndarray:
    """Detect dark halo pixels only on the exterior alpha border."""
    arr = np.array(image.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    border = _outer_alpha_border(alpha)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return border & (gray < 48)


def remove_white_halo_safe(image: Image.Image, strength: str = "normal") -> Image.Image:
    """Neutralize white edge contamination without touching internal whites."""
    return _replace_halo_with_neighbor_color(image, detect_white_halo(image), strength)


def remove_black_halo_safe(image: Image.Image, strength: str = "normal") -> Image.Image:
    """Neutralize black edge contamination without touching internal outlines."""
    return _replace_halo_with_neighbor_color(image, detect_black_halo(image), strength)


def expand_edge_colors(image: Image.Image, pixels: int = 1) -> Image.Image:
    """Extend artwork edge colors into low-alpha pixels without changing alpha."""
    if pixels <= 0:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    solid = alpha > 20
    target = cv2.dilate(solid.astype(np.uint8), np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8), iterations=1).astype(bool) & ~solid
    if not np.any(target):
        return rgba
    rgb = arr[:, :, :3]
    _, labels = cv2.distanceTransformWithLabels((~solid).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    points = np.column_stack(np.where(solid))
    if len(points) == 0:
        return rgba
    labels_flat = labels[target] - 1
    valid = (labels_flat >= 0) & (labels_flat < len(points))
    ys, xs = np.where(target)
    arr[ys[valid], xs[valid], :3] = rgb[points[labels_flat[valid], 0], points[labels_flat[valid], 1]]
    return Image.fromarray(arr, "RGBA")


def create_bleed(image: Image.Image, pixels: int = 0) -> tuple[Image.Image, np.ndarray]:
    """Create an external color bleed based on artwork colors."""
    rgba = expand_edge_colors(image, max(0, pixels)).convert("RGBA")
    if pixels <= 0:
        return rgba, np.zeros((rgba.height, rgba.width), dtype=bool)
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    solid = alpha > 20
    expanded = cv2.dilate(solid.astype(np.uint8), np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8), iterations=1).astype(bool)
    bleed = expanded & ~solid
    arr[bleed, 3] = 255
    return Image.fromarray(arr, "RGBA"), bleed


def create_cutline_mask(image: Image.Image, offset_px: int = 2) -> np.ndarray:
    """Create a contour mask around artwork for cutline preview/export."""
    alpha = np.array(image.convert("RGBA").getchannel("A"))
    solid = alpha > 20
    radius = max(1, int(offset_px))
    expanded = cv2.dilate(solid.astype(np.uint8), np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8), iterations=1).astype(bool)
    inner = cv2.erode(expanded.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return expanded & ~inner


def detect_too_small_elements(image: Image.Image, dpi: int, min_mm: float = 1.0) -> dict[str, object]:
    """Detect printable alpha components smaller than the requested millimeter threshold."""
    alpha = np.array(image.convert("RGBA").getchannel("A"))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 20).astype(np.uint8), 8)
    min_px = max(1, int(round(float(min_mm) / 25.4 * dpi)))
    small_labels = np.zeros(count, dtype=bool)
    for label in range(1, count):
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        small_labels[label] = min(width, height) < min_px or area < min_px * min_px
    mask = small_labels[labels]
    return {
        "count": int(np.count_nonzero(small_labels)),
        "min_px": int(min_px),
        "min_mm": float(min_mm),
        "warning": bool(np.any(mask)),
        "mask": mask,
    }


def prepare_dtf(image: Image.Image, settings: DTFPrepressSettings | dict[str, object]) -> DTFPrepressResult:
    """Apply safe DTF prepress operations and return QA artifacts."""
    cfg = settings if isinstance(settings, DTFPrepressSettings) else DTFPrepressSettings(**settings)
    work = image.convert("RGBA")
    white_halo = detect_white_halo(work)
    black_halo = detect_black_halo(work)
    if cfg.enable_dtf_prepress:
        if cfg.remove_white_halo:
            work = remove_white_halo_safe(work, cfg.halo_strength)
        if cfg.remove_black_halo:
            work = remove_black_halo_safe(work, cfg.halo_strength)
        work = expand_edge_colors(work, cfg.expand_edge_px)
        work, bleed_mask = create_bleed(work, cfg.bleed_px)
    else:
        bleed_mask = np.zeros((work.height, work.width), dtype=bool)
    cutline = create_cutline_mask(work, max(1, cfg.expand_edge_px if cfg.create_cutline else 1)) if cfg.create_cutline else np.zeros((work.height, work.width), dtype=bool)
    small = detect_too_small_elements(work, cfg.dpi, cfg.min_printable_mm)
    alpha_quality = analyze_alpha_quality(work)
    metadata = {
        "dtf_prepress": bool(cfg.enable_dtf_prepress),
        "cutline_ready": bool(cfg.create_cutline and np.any(cutline)),
        "small_elements_count": int(small["count"]),
        "halo_white_risk": bool(np.any(white_halo)),
        "halo_black_risk": bool(np.any(black_halo)),
    }
    return DTFPrepressResult(work, alpha_quality, white_halo, black_halo, bleed_mask, cutline, small["mask"], _public_small_report(small), metadata)


def mask_png_bytes(mask: np.ndarray, color: tuple[int, int, int, int] = (0, 220, 120, 255)) -> bytes:
    """Encode a boolean mask as transparent PNG bytes."""
    from io import BytesIO

    arr = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    arr[mask] = color
    bio = BytesIO()
    Image.fromarray(arr, "RGBA").save(bio, format="PNG")
    return bio.getvalue()


def _replace_halo_with_neighbor_color(image: Image.Image, halo: np.ndarray, strength: str) -> Image.Image:
    rgba = image.convert("RGBA")
    if not np.any(halo):
        return rgba
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    solid = alpha > 20
    clean_source = solid & ~halo
    if not np.any(clean_source):
        return rgba
    _, labels = cv2.distanceTransformWithLabels((~clean_source).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    points = np.column_stack(np.where(clean_source))
    ys, xs = np.where(halo)
    nearest = labels[ys, xs] - 1
    valid = (nearest >= 0) & (nearest < len(points))
    blend = {"suave": 0.45, "normal": 0.7, "fuerte": 0.9}.get(strength, 0.7)
    source_rgb = arr[points[nearest[valid], 0], points[nearest[valid], 1], :3].astype(np.float32)
    current = arr[ys[valid], xs[valid], :3].astype(np.float32)
    arr[ys[valid], xs[valid], :3] = np.clip(current * (1 - blend) + source_rgb * blend, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _alpha_border(alpha: np.ndarray) -> np.ndarray:
    solid = alpha > 20
    edge = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return edge.astype(bool)


def _outer_alpha_border(alpha: np.ndarray) -> np.ndarray:
    solid = alpha > 20
    expanded = cv2.dilate(solid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return expanded & ~cv2.erode(solid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)


def _small_holes(binary: np.ndarray) -> int:
    holes = ~binary
    count, _, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    if count <= 1:
        return 0
    small = (stats[1:, cv2.CC_STAT_AREA] <= 12).sum()
    return int(small)


def _alpha_noise(alpha: np.ndarray, binary: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return 0.0
    small = stats[1:, cv2.CC_STAT_AREA] < 6
    return float(np.count_nonzero(small))


def _public_small_report(report: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in report.items() if key != "mask"}


def dataclass_to_dict(result: DTFPrepressResult) -> dict[str, object]:
    """Return serializable metadata for a prepress result."""
    data = asdict(result)
    for key in ["white_halo_mask", "black_halo_mask", "bleed_mask", "cutline_mask", "small_elements_mask", "image"]:
        data.pop(key, None)
    return data
