from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.white_complex import compose_on_solid


@dataclass(frozen=True)
class ResidueRefineSettings:
    """Controls for assisted cleanup of light residue components."""

    luminosity_threshold: int = 220
    saturation_threshold: int = 50
    min_area: int = 8
    max_area: int = 5000
    remove_connected: bool = True
    remove_small: bool = False
    preserve_internal_white: bool = True
    manual_remove_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResidueComponent:
    """Measured light component that may be residue or valid artwork."""

    id: int
    area: int
    bbox: tuple[int, int, int, int]
    distance_to_transparency: float
    mean_luminosity: float
    mean_saturation: float
    suspicion_score: int
    touches_transparency: bool
    touches_outer_edge: bool
    classification: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_light_residue_components(
    image: Image.Image,
    settings: ResidueRefineSettings | None = None,
) -> list[ResidueComponent]:
    """Detect opaque white or gray components left after background removal."""
    options = settings or ResidueRefineSettings()
    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    light_mask = _light_opaque_mask(arr, options)
    light_mask &= _size_window_mask(light_mask, options.min_area, options.max_area)
    labels, count = build_residue_component_map(light_mask)
    if count == 0:
        return []

    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)[:, :, 1]
    transparent = alpha <= 20
    near_transparent = cv2.dilate(transparent.astype(np.uint8), _kernel(5), iterations=1).astype(bool)
    distance = cv2.distanceTransform((~transparent).astype(np.uint8), cv2.DIST_L2, 3)
    return _components_from_labels(labels, count, gray, sat, distance, near_transparent, options)


def build_residue_component_map(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return connected component labels for a binary residue candidate mask."""
    labels_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels, max(0, labels_count - 1)


def apply_residue_component_removal(
    image: Image.Image,
    components: list[ResidueComponent],
    settings: ResidueRefineSettings | None = None,
) -> Image.Image:
    """Remove only approved residue components by setting their alpha to zero."""
    options = settings or ResidueRefineSettings()
    arr = np.array(image.convert("RGBA"))
    labels, _ = build_residue_component_map(_light_opaque_mask(arr, options))
    remove_ids = _removable_ids(components, options)
    if not remove_ids:
        return image.convert("RGBA")
    remove_mask = np.isin(labels, list(remove_ids))
    out = arr.copy()
    out[remove_mask, 3] = 0
    return Image.fromarray(out, "RGBA")


def overlay_residue_components(
    image: Image.Image,
    components: list[ResidueComponent],
    settings: ResidueRefineSettings | None = None,
) -> Image.Image:
    """Draw a QA overlay with residue components colored by classification."""
    options = settings or ResidueRefineSettings()
    base = compose_on_solid(image, (0, 0, 0)).convert("RGBA")
    labels, _ = build_residue_component_map(_light_opaque_mask(np.array(image.convert("RGBA")), options))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    colors = {
        "borrar": (240, 40, 40, 150),
        "ambiguo": (255, 210, 30, 140),
        "arte interno": (0, 220, 120, 130),
    }
    for component in components:
        overlay_arr[labels == component.id] = colors.get(component.classification, colors["ambiguo"])
    overlay = Image.fromarray(overlay_arr, "RGBA")
    base.alpha_composite(overlay)
    draw = ImageDraw.Draw(base)
    for component in components[:80]:
        x1, y1, _, _ = component.bbox
        draw.text((x1 + 1, y1 + 1), str(component.id), fill=(255, 255, 255, 255))
    return base


def residue_component_report_json(components: list[ResidueComponent]) -> bytes:
    """Serialize residue component measurements for ZIP debug export."""
    data = [component.to_dict() for component in components]
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def build_residue_debug(
    original_result: Image.Image,
    refined_result: Image.Image,
    components: list[ResidueComponent],
    settings: ResidueRefineSettings,
) -> dict[str, object]:
    """Build previews and stats for assisted residue refinement."""
    overlay = overlay_residue_components(original_result, components, settings)
    component_mask = _component_mask_preview(original_result, components, settings)
    removed = _removed_component_count(components, settings)
    return {
        "components": [component.to_dict() for component in components],
        "stats": {
            "components_detected": len(components),
            "components_removed": removed,
            "ambiguous_components": sum(1 for item in components if item.classification == "ambiguo"),
            "internal_components_preserved": sum(1 for item in components if item.classification == "arte interno"),
        },
        "previews": {
            "residue_components": component_mask,
            "residue_overlay": overlay,
            "refined_preview_black": compose_on_solid(refined_result, (0, 0, 0)),
            "refined_preview_red": compose_on_solid(refined_result, (180, 0, 0)),
        },
        "report_json": residue_component_report_json(components),
    }


def _light_opaque_mask(arr: np.ndarray, settings: ResidueRefineSettings) -> np.ndarray:
    rgb = arr[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    alpha = arr[:, :, 3]
    return (alpha > 220) & (gray >= settings.luminosity_threshold) & (sat <= settings.saturation_threshold)


def _size_window_mask(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    labels, count = build_residue_component_map(mask)
    if count == 0:
        return mask
    stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)[2]
    areas = stats[:, cv2.CC_STAT_AREA]
    valid_ids = np.where((areas >= min_area) & (areas <= max_area))[0]
    valid_ids = valid_ids[valid_ids != 0]
    return np.isin(labels, valid_ids)


def _components_from_labels(
    labels: np.ndarray,
    count: int,
    gray: np.ndarray,
    sat: np.ndarray,
    distance: np.ndarray,
    near_transparent: np.ndarray,
    settings: ResidueRefineSettings,
) -> list[ResidueComponent]:
    stats = cv2.connectedComponentsWithStats((labels > 0).astype(np.uint8), 8)[2]
    components: list[ResidueComponent] = []
    h, w = labels.shape
    for component_id in range(1, count + 1):
        mask = labels == component_id
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        area = int(mask.sum())
        touches_edge = x1 == 0 or y1 == 0 or x2 == w or y2 == h
        touches_transparency = bool(np.any(mask & near_transparent))
        mean_luma = float(np.mean(gray[mask]))
        mean_sat = float(np.mean(sat[mask]))
        dist = float(np.min(distance[mask]))
        score = _suspicion_score(area, mean_luma, mean_sat, dist, touches_transparency, touches_edge)
        classification = _classify_component(score, touches_transparency, touches_edge, settings)
        components.append(ResidueComponent(
            id=component_id,
            area=area,
            bbox=(x1, y1, x2, y2),
            distance_to_transparency=round(dist, 2),
            mean_luminosity=round(mean_luma, 2),
            mean_saturation=round(mean_sat, 2),
            suspicion_score=score,
            touches_transparency=touches_transparency,
            touches_outer_edge=touches_edge,
            classification=classification,
        ))
    return components


def _suspicion_score(
    area: int,
    luma: float,
    sat: float,
    distance: float,
    touches_transparency: bool,
    touches_edge: bool,
) -> int:
    score = 0
    score += 35 if touches_transparency else 0
    score += 20 if touches_edge else 0
    score += min(25, max(0, int((luma - 220) / 1.4)))
    score += min(15, max(0, int((55 - sat) / 4)))
    score += 12 if distance <= 2 else 0
    score += 8 if area <= 120 else 0
    return min(100, score)


def _classify_component(
    score: int,
    touches_transparency: bool,
    touches_edge: bool,
    settings: ResidueRefineSettings,
) -> str:
    if settings.remove_connected and (touches_transparency or touches_edge) and score >= 62:
        return "borrar"
    if settings.preserve_internal_white and not touches_transparency and not touches_edge:
        return "arte interno" if score < 72 else "ambiguo"
    return "ambiguo"


def _removable_ids(components: list[ResidueComponent], settings: ResidueRefineSettings) -> set[int]:
    manual = set(settings.manual_remove_ids)
    selected = {item.id for item in components if item.classification == "borrar"}
    if settings.remove_small:
        selected.update(item.id for item in components if item.area <= settings.min_area * 4 and item.touches_transparency)
    selected.update(manual)
    if settings.preserve_internal_white:
        protected = {item.id for item in components if item.classification == "arte interno"}
        selected -= protected - manual
    return selected


def _removed_component_count(components: list[ResidueComponent], settings: ResidueRefineSettings) -> int:
    return len(_removable_ids(components, settings))


def _component_mask_preview(
    image: Image.Image,
    components: list[ResidueComponent],
    settings: ResidueRefineSettings,
) -> Image.Image:
    labels, _ = build_residue_component_map(_light_opaque_mask(np.array(image.convert("RGBA")), settings))
    out = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
    for component in components:
        color = (240, 40, 40, 220) if component.classification == "borrar" else (255, 210, 30, 190)
        if component.classification == "arte interno":
            color = (0, 220, 120, 190)
        out[labels == component.id] = color
    return Image.fromarray(out, "RGBA")


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
