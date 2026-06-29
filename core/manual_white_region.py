from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.white_complex import compose_on_solid


@dataclass(frozen=True)
class ManualWhiteRegionSettings:
    """Manual seed-selection controls for white residual regions."""

    tolerance: int = 42
    luminosity_min: int = 190
    saturation_max: int = 90
    max_area: int = 50000
    connectivity: int = 8


@dataclass(frozen=True)
class ManualWhiteRegionSelection:
    """Selected region mask plus safety report."""

    seed: tuple[int, int]
    mask: np.ndarray
    area: int
    percent: float
    mean_luminosity: float
    mean_saturation: float
    decision: str
    reason: str
    tolerance: int

    def report(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("mask")
        return data


def select_light_region_by_seed(
    image: Image.Image,
    seed: tuple[int, int],
    settings: ManualWhiteRegionSettings | None = None,
) -> ManualWhiteRegionSelection:
    """Select a connected light region from a user-provided seed."""
    options = settings or ManualWhiteRegionSettings()
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    x, y = seed
    empty = np.zeros((h, w), dtype=bool)
    if x < 0 or y < 0 or x >= w or y >= h:
        return _selection(seed, empty, "rechazado", "semilla_fuera_de_rango", options, arr)
    if not _seed_is_light(arr, x, y, options):
        return _selection(seed, empty, "rechazado", "la_region_no_parece_fondo_blanco", options, arr)

    candidate = _candidate_mask(arr, x, y, options)
    labels_count, labels = cv2.connectedComponents(candidate.astype(np.uint8), options.connectivity)
    label = int(labels[y, x]) if labels_count > 1 else 0
    if label == 0:
        return _selection(seed, empty, "rechazado", "sin_region_conectada", options, arr)
    mask = labels == label
    decision, reason = _safety_decision(arr, mask, options)
    return _selection(seed, mask, decision, reason, options, arr)


def remove_light_region_by_seed(
    image: Image.Image,
    seeds: tuple[tuple[int, int], ...],
    settings: ManualWhiteRegionSettings | None = None,
) -> tuple[Image.Image, list[ManualWhiteRegionSelection]]:
    """Remove accepted seed-selected light regions from alpha only."""
    options = settings or ManualWhiteRegionSettings()
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    selections = [select_light_region_by_seed(rgba, seed, options) for seed in seeds]
    accepted = [selection.mask for selection in selections if selection.decision == "aplicado"]
    if not accepted:
        return rgba, selections
    remove_mask = np.logical_or.reduce(accepted)
    out = arr.copy()
    out[remove_mask, 3] = 0
    return Image.fromarray(out, "RGBA"), selections


def preview_selected_region_overlay(
    image: Image.Image,
    selections: list[ManualWhiteRegionSelection],
) -> Image.Image:
    """Render selected manual regions over a black preview."""
    base = compose_on_solid(image, (0, 0, 0)).convert("RGBA")
    overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    for selection in selections:
        color = (240, 40, 40, 190) if selection.decision == "aplicado" else (255, 210, 30, 170)
        overlay[selection.mask] = color
    base.alpha_composite(Image.fromarray(overlay, "RGBA"))
    draw = ImageDraw.Draw(base)
    for selection in selections:
        x, y = selection.seed
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 255, 255, 255), width=2)
    return base


def manual_white_region_report(selections: list[ManualWhiteRegionSelection]) -> dict[str, object]:
    """Return a serializable report for manual white-region cleanup."""
    applied = [selection for selection in selections if selection.decision == "aplicado"]
    rejected = [selection for selection in selections if selection.decision != "aplicado"]
    return {
        "regions": [selection.report() for selection in selections],
        "applied_count": len(applied),
        "rejected_count": len(rejected),
        "applied_area": int(sum(selection.area for selection in applied)),
        "rejected_reasons": [selection.reason for selection in rejected],
        "seeds": [selection.seed for selection in selections],
    }


def manual_white_region_report_json(selections: list[ManualWhiteRegionSelection]) -> bytes:
    """Serialize manual white-region cleanup report for ZIP debug export."""
    return json.dumps(manual_white_region_report(selections), indent=2, ensure_ascii=False).encode("utf-8")


def parse_seed_text(raw: str) -> tuple[tuple[int, int], ...]:
    """Parse 'x,y; x,y' seed text into integer coordinate pairs."""
    seeds: list[tuple[int, int]] = []
    for part in raw.replace("\n", ";").split(";"):
        values = [value.strip() for value in part.split(",")]
        if len(values) != 2:
            continue
        if values[0].lstrip("-").isdigit() and values[1].lstrip("-").isdigit():
            seeds.append((int(values[0]), int(values[1])))
    return tuple(seeds)


def selected_region_mask_image(selections: list[ManualWhiteRegionSelection]) -> Image.Image | None:
    """Build an RGBA mask image for selected manual regions."""
    if not selections:
        return None
    h, w = selections[0].mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for selection in selections:
        color = (240, 40, 40, 220) if selection.decision == "aplicado" else (255, 210, 30, 200)
        out[selection.mask] = color
    return Image.fromarray(out, "RGBA")


def _seed_is_light(arr: np.ndarray, x: int, y: int, settings: ManualWhiteRegionSettings) -> bool:
    if arr[y, x, 3] <= 20:
        return False
    rgb = arr[y, x, :3].astype(np.uint8).reshape(1, 1, 3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[0, 0]
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0, 1]
    return bool(gray >= settings.luminosity_min and sat <= settings.saturation_max)


def _candidate_mask(arr: np.ndarray, x: int, y: int, settings: ManualWhiteRegionSettings) -> np.ndarray:
    rgb = arr[:, :, :3].astype(np.int16)
    seed_rgb = arr[y, x, :3].astype(np.int16)
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)[:, :, 1]
    distance = np.linalg.norm(rgb - seed_rgb, axis=2)
    return (
        (arr[:, :, 3] > 20)
        & (gray >= settings.luminosity_min)
        & (sat <= settings.saturation_max)
        & (distance <= settings.tolerance)
    )


def _safety_decision(
    arr: np.ndarray,
    mask: np.ndarray,
    settings: ManualWhiteRegionSettings,
) -> tuple[str, str]:
    area = int(mask.sum())
    if area <= 0:
        return "rechazado", "sin_region_conectada"
    if area > settings.max_area:
        return "rechazado", "area_demasiado_grande"
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    saturated = _percent((hsv[:, :, 1] > max(110, settings.saturation_max + 25)) & mask, mask)
    dark = _percent((gray < 80) & mask, mask)
    if saturated > 2.0:
        return "rechazado", "region_contiene_color_del_arte"
    if dark > 1.0:
        return "rechazado", "region_contiene_arte_oscuro"
    if _protected_art_context(arr, mask):
        return "rechazado", "region_parece_parte_protegida_del_arte"
    return "aplicado", "region_blanca_conectada"


def _selection(
    seed: tuple[int, int],
    mask: np.ndarray,
    decision: str,
    reason: str,
    settings: ManualWhiteRegionSettings,
    arr: np.ndarray,
) -> ManualWhiteRegionSelection:
    metrics = _region_metrics(arr, mask)
    return ManualWhiteRegionSelection(
        seed=seed,
        mask=mask,
        area=metrics["area"],
        percent=metrics["percent"],
        mean_luminosity=metrics["mean_luminosity"],
        mean_saturation=metrics["mean_saturation"],
        decision=decision,
        reason=reason,
        tolerance=settings.tolerance,
    )


def _region_metrics(arr: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    area = int(mask.sum())
    if area == 0:
        return {"area": 0, "percent": 0.0, "mean_luminosity": 0.0, "mean_saturation": 0.0}
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)[:, :, 1]
    return {
        "area": area,
        "percent": round(area / max(1, mask.size) * 100, 4),
        "mean_luminosity": round(float(np.mean(gray[mask])), 2),
        "mean_saturation": round(float(np.mean(sat[mask])), 2),
    }


def _percent(candidate: np.ndarray, mask: np.ndarray) -> float:
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return float(np.count_nonzero(candidate & mask) / total * 100)


def _protected_art_context(arr: np.ndarray, mask: np.ndarray) -> bool:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    ring = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
    visible = ring & (arr[:, :, 3] > 20)
    if not np.any(visible):
        return False
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    blue = arr[:, :, 2].astype(np.int16)
    colored = (hsv[:, :, 1] > 65) & ((red > green + 28) | (blue > red + 28))
    color_context = _percent(colored & visible, visible)
    return color_context >= 42
