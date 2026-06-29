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


@dataclass(frozen=True)
class InternalLightResidueSettings:
    """Controls for trapped white residue inside dark grunge or linework."""

    min_area: int = 4
    max_area: int = 900
    dark_neighbor_threshold: int = 34
    luminosity_threshold: int = 218
    saturation_threshold: int = 58
    auto_remove_high_confidence: bool = False
    manual_remove_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class InternalLightResidueComponent:
    """Measured internal light component and its suggested cleanup action."""

    id: int
    area: int
    bbox: tuple[int, int, int, int]
    mean_luminosity: float
    mean_saturation: float
    distance_to_transparency: float
    dark_neighbor_percent: float
    transparency_border_percent: float
    residue_score: int
    suggested_action: str
    protection_reason: str

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


def detect_internal_light_residue_components(
    image: Image.Image,
    settings: InternalLightResidueSettings | None = None,
) -> list[InternalLightResidueComponent]:
    """Detect trapped white residue surrounded by dark internal artwork."""
    options = settings or InternalLightResidueSettings()
    arr = np.array(image.convert("RGBA"))
    light = _internal_light_mask(arr, options)
    light &= _size_window_mask(light, options.min_area, options.max_area)
    labels, count = build_residue_component_map(light)
    if count == 0:
        return []
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)[:, :, 1]
    alpha = arr[:, :, 3]
    transparent = alpha <= 20
    distance = cv2.distanceTransform((~transparent).astype(np.uint8), cv2.DIST_L2, 3)
    components: list[InternalLightResidueComponent] = []
    for component_id in range(1, count + 1):
        mask = labels == component_id
        if not mask.any():
            continue
        ring = _component_ring(mask, radius=5)
        wide_ring = _component_ring(mask, radius=16)
        visible_ring = ring & (alpha > 20)
        visible_wide_ring = wide_ring & (alpha > 20)
        dark_percent = _percent_in_mask((gray < 72) & visible_ring, visible_ring)
        transparent_percent = _percent_in_mask(transparent & ring, ring)
        color_percent = _percent_in_mask(_colored_neighbor_mask(arr) & visible_wide_ring, visible_wide_ring)
        luma = float(np.mean(gray[mask]))
        saturation = float(np.mean(sat[mask]))
        dist = float(np.min(distance[mask]))
        area = int(mask.sum())
        score = score_light_residue_component(
            area=area,
            mean_luminosity=luma,
            mean_saturation=saturation,
            distance_to_transparency=dist,
            dark_neighbor_percent=dark_percent,
            transparency_border_percent=transparent_percent,
            colored_neighbor_percent=color_percent,
            settings=options,
        )
        action, reason = _internal_action_and_reason(score, area, dark_percent, color_percent, options)
        components.append(InternalLightResidueComponent(
            id=component_id,
            area=area,
            bbox=_bbox(mask),
            mean_luminosity=round(luma, 2),
            mean_saturation=round(saturation, 2),
            distance_to_transparency=round(dist, 2),
            dark_neighbor_percent=round(dark_percent, 2),
            transparency_border_percent=round(transparent_percent, 2),
            residue_score=score,
            suggested_action=action,
            protection_reason=reason,
        ))
    return components


def score_light_residue_component(
    area: int,
    mean_luminosity: float,
    mean_saturation: float,
    distance_to_transparency: float,
    dark_neighbor_percent: float,
    transparency_border_percent: float,
    colored_neighbor_percent: float = 0,
    settings: InternalLightResidueSettings | None = None,
) -> int:
    """Score whether a light component is trapped residue, not intentional white art."""
    options = settings or InternalLightResidueSettings()
    score = 0
    score += min(24, max(0, int((mean_luminosity - options.luminosity_threshold) / 1.4)))
    score += min(18, max(0, int((options.saturation_threshold - mean_saturation) / 3)))
    score += min(36, int(dark_neighbor_percent * 0.55))
    score += 12 if options.min_area <= area <= min(options.max_area, 260) else 0
    score += 8 if distance_to_transparency >= 4 else 0
    score -= min(28, int(colored_neighbor_percent * 0.5))
    score -= 18 if area > options.max_area * 0.65 else 0
    score -= 10 if transparency_border_percent > 20 else 0
    return max(0, min(100, score))


def remove_selected_light_residue_components(
    image: Image.Image,
    components: list[InternalLightResidueComponent],
    settings: InternalLightResidueSettings | None = None,
) -> Image.Image:
    """Remove high-confidence or manually selected internal residue components."""
    options = settings or InternalLightResidueSettings()
    remove_ids = _internal_removable_ids(components, options)
    if not remove_ids:
        return image.convert("RGBA")
    arr = np.array(image.convert("RGBA"))
    labels, _ = build_residue_component_map(_internal_light_mask(arr, options))
    out = arr.copy()
    out[np.isin(labels, list(remove_ids)), 3] = 0
    return Image.fromarray(out, "RGBA")


def force_remove_internal_review_components(
    image: Image.Image,
    components: list[InternalLightResidueComponent],
    settings: InternalLightResidueSettings | None = None,
) -> Image.Image:
    """Debug helper that removes every component marked borrar or revisar."""
    options = settings or InternalLightResidueSettings()
    force_ids = tuple(item.id for item in components if item.suggested_action in {"borrar", "revisar"})
    return remove_selected_light_residue_components(image, components, replace_internal_manual_ids(options, force_ids))


def replace_internal_manual_ids(
    settings: InternalLightResidueSettings,
    manual_ids: tuple[int, ...],
) -> InternalLightResidueSettings:
    """Return settings with manual IDs replaced without importing dataclasses.replace."""
    return InternalLightResidueSettings(
        min_area=settings.min_area,
        max_area=settings.max_area,
        dark_neighbor_threshold=settings.dark_neighbor_threshold,
        luminosity_threshold=settings.luminosity_threshold,
        saturation_threshold=settings.saturation_threshold,
        auto_remove_high_confidence=settings.auto_remove_high_confidence,
        manual_remove_ids=manual_ids,
    )


def build_light_residue_overlay(
    image: Image.Image,
    components: list[InternalLightResidueComponent],
    settings: InternalLightResidueSettings | None = None,
) -> Image.Image:
    """Build a black-background overlay for internal light residue components."""
    options = settings or InternalLightResidueSettings()
    base = compose_on_solid(image, (0, 0, 0)).convert("RGBA")
    labels, _ = build_residue_component_map(_internal_light_mask(np.array(image.convert("RGBA")), options))
    overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    remove_ids = _internal_removable_ids(components, options)
    for component in components:
        color = (40, 130, 255, 175)
        if component.id in remove_ids:
            color = (240, 40, 40, 200)
        elif component.suggested_action == "revisar":
            color = (255, 210, 30, 170)
        overlay[labels == component.id] = color
    base.alpha_composite(Image.fromarray(overlay, "RGBA"))
    draw = ImageDraw.Draw(base)
    for component in components[:80]:
        x1, y1, _, _ = component.bbox
        draw.text((x1 + 1, y1 + 1), str(component.id), fill=(255, 255, 255, 255))
    return base


def build_internal_residue_debug(
    source: Image.Image,
    refined: Image.Image,
    components: list[InternalLightResidueComponent],
    settings: InternalLightResidueSettings,
) -> dict[str, object]:
    """Build previews and report for trapped internal white residue cleanup."""
    diagnostics = _internal_residue_stats(source, refined, components, settings)
    return {
        "components": [component.to_dict() for component in components],
        "stats": diagnostics,
        "previews": {
            "internal_residue_mask": _internal_component_mask(source, components, settings),
            "internal_residue_overlay": build_light_residue_overlay(source, components, settings),
            "preview_green": compose_on_solid(refined, (0, 150, 80)),
            "preview_black": compose_on_solid(refined, (0, 0, 0)),
        },
        "report_json": json.dumps([component.to_dict() for component in components], indent=2, ensure_ascii=False).encode("utf-8"),
    }


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


def _internal_light_mask(arr: np.ndarray, settings: InternalLightResidueSettings) -> np.ndarray:
    rgb = arr[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    alpha = arr[:, :, 3]
    return (alpha > 220) & (gray >= settings.luminosity_threshold) & (sat <= settings.saturation_threshold)


def _component_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = _kernel(radius * 2 + 1)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return dilated & ~mask


def _percent_in_mask(candidate: np.ndarray, mask: np.ndarray) -> float:
    total = int(np.count_nonzero(mask))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(candidate & mask) / total * 100)


def _colored_neighbor_mask(arr: np.ndarray) -> np.ndarray:
    rgb = arr[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    alpha = arr[:, :, 3]
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    warm = (red > 120) & (red > green + 30) & (red > blue + 10)
    cool = (blue > 110) & (blue > red + 20)
    saturated = hsv[:, :, 1] > 55
    return (alpha > 20) & saturated & (warm | cool)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _internal_action_and_reason(
    score: int,
    area: int,
    dark_percent: float,
    color_percent: float,
    settings: InternalLightResidueSettings,
) -> tuple[str, str]:
    if area > settings.max_area * 0.65 or color_percent >= 34:
        reason = "area_grande" if area > settings.max_area * 0.65 else "contexto_color"
        return "conservar", reason
    if score >= 72 and dark_percent >= settings.dark_neighbor_threshold:
        return "borrar", "alto_contraste"
    if score >= 50 and dark_percent >= max(18, settings.dark_neighbor_threshold * 0.65):
        return "revisar", "score_ambiguo"
    return "conservar", "score_bajo"


def _internal_removable_ids(
    components: list[InternalLightResidueComponent],
    settings: InternalLightResidueSettings,
) -> set[int]:
    manual = set(settings.manual_remove_ids)
    found = {item.id for item in components}
    selected = {item.id for item in components if item.id in manual and not _manual_rejected(item, settings)}
    if settings.auto_remove_high_confidence:
        selected.update(item.id for item in components if item.suggested_action == "borrar")
    selected &= found
    return selected


def _manual_rejected(component: InternalLightResidueComponent, settings: InternalLightResidueSettings) -> bool:
    if component.suggested_action != "conservar":
        return False
    if component.protection_reason == "contexto_color":
        return True
    return component.protection_reason == "area_grande" and component.area > settings.max_area * 0.85


def _internal_component_mask(
    image: Image.Image,
    components: list[InternalLightResidueComponent],
    settings: InternalLightResidueSettings,
) -> Image.Image:
    labels, _ = build_residue_component_map(_internal_light_mask(np.array(image.convert("RGBA")), settings))
    out = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
    for component in components:
        color = (40, 130, 255, 210)
        if component.id in _internal_removable_ids(components, settings):
            color = (240, 40, 40, 235)
        elif component.suggested_action == "revisar":
            color = (255, 210, 30, 210)
        out[labels == component.id] = color
    return Image.fromarray(out, "RGBA")


def _internal_residue_stats(
    source: Image.Image,
    refined: Image.Image,
    components: list[InternalLightResidueComponent],
    settings: InternalLightResidueSettings,
) -> dict[str, object]:
    source_alpha = np.array(source.convert("RGBA").getchannel("A"))
    refined_alpha = np.array(refined.convert("RGBA").getchannel("A"))
    removed_pixels = int(np.count_nonzero((source_alpha > 20) & (refined_alpha <= 20)))
    total = max(1, source_alpha.size)
    remove_ids = sorted(_internal_removable_ids(components, settings))
    manual = set(settings.manual_remove_ids)
    found_manual = sorted(manual & {item.id for item in components})
    rejected_manual = sorted(item.id for item in components if item.id in manual and _manual_rejected(item, settings))
    protected = [item for item in components if item.suggested_action == "conservar"]
    review = [item for item in components if item.suggested_action == "revisar"]
    high = [item for item in components if item.suggested_action == "borrar"]
    protection_reasons = {str(item.id): item.protection_reason for item in protected}
    return {
        "internal_components_detected": len(components),
        "internal_components_removed": len(remove_ids),
        "internal_components_protected": len(protected),
        "internal_components_review": len(review),
        "internal_residue_area": sum(item.area for item in components),
        "internal_residue_candidate_area": sum(item.area for item in high + review),
        "internal_removed_area": sum(item.area for item in components if item.id in set(remove_ids)),
        "internal_removed_pixels": removed_pixels,
        "internal_alpha_change_percent": round(removed_pixels / total * 100, 4),
        "internal_high_confidence": len(high),
        "internal_removed_ids": remove_ids,
        "internal_protected_ids": [item.id for item in protected],
        "internal_review_ids": [item.id for item in review],
        "internal_manual_found_ids": found_manual,
        "internal_manual_removed_ids": sorted(set(remove_ids) & manual),
        "internal_manual_rejected_ids": rejected_manual,
        "internal_protection_reasons": protection_reasons,
        "internal_no_pixels_removed": removed_pixels == 0,
    }
