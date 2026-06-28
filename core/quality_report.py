"""Professional quality report for non-destructive DTF processing."""

from __future__ import annotations

import numpy as np
from PIL import Image


def quality_report(
    original: Image.Image,
    result: Image.Image,
    artwork_mask: np.ndarray | None = None,
    background_mask: np.ndarray | None = None,
    risk: dict[str, object] | None = None,
    alpha_quality: dict[str, object] | None = None,
    dtf_prepress: dict[str, object] | None = None,
    small_elements_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return score, warnings, and production QA metrics."""
    src = np.array(original.convert("RGBA"))
    dst = np.array(result.convert("RGBA"))
    total = max(1, src.shape[0] * src.shape[1])
    target_shape = src.shape[:2]
    bg = _normalize_mask(background_mask, target_shape, "background_mask")
    art = _normalize_mask(artwork_mask, target_shape, "artwork_mask")
    removed_background = int(np.count_nonzero(bg)) if bg is not None else _fallback_removed_background(src, dst)
    protected_art = int(np.count_nonzero(art)) if art is not None else 0
    removed_background_percent = round(removed_background / total * 100, 2)
    protected_art_percent = round(protected_art / total * 100, 2)
    risk_detected = bool((risk or {}).get("risk_detected", False))
    warnings = []
    if risk_detected:
        warnings.append("Riesgo de perdida de arte detectado.")
    if protected_art_percent < 1:
        warnings.append("Arte protegido insuficiente.")
    halo_white = bool((dtf_prepress or {}).get("halo_white_risk", False))
    halo_black = bool((dtf_prepress or {}).get("halo_black_risk", False))
    semi = float((alpha_quality or {}).get("semi_transparent_percent", 0))
    small_count = int((small_elements_report or {}).get("count", 0))
    cutline_ready = bool((dtf_prepress or {}).get("cutline_ready", False))
    if halo_white:
        warnings.append("Riesgo de halo blanco en borde.")
    if halo_black:
        warnings.append("Riesgo de halo negro en borde.")
    if small_count:
        warnings.append("Hay detalles menores al tamano minimo imprimible.")
    score = 100 - (35 if risk_detected else 0) - (10 if protected_art_percent < 1 else 0)
    dtf_score = score - (12 if halo_white else 0) - (10 if halo_black else 0) - (8 if semi > 12 else 0) - min(15, small_count * 3)
    return {
        "score": max(0, score),
        "warnings": warnings,
        "removed_background_percent": removed_background_percent,
        "protected_art_percent": protected_art_percent,
        "risk_level": "alto" if risk_detected else "bajo",
        "halo_white_risk": halo_white,
        "halo_black_risk": halo_black,
        "semi_transparent_percent": round(semi, 2),
        "small_elements_count": small_count,
        "cutline_ready": cutline_ready,
        "dtf_ready_score": max(0, int(dtf_score)),
    }


def _normalize_mask(mask: object | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    """Return a 2D boolean mask matching shape, or None when unavailable."""
    if mask is None:
        return None
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("RGBA"))
    else:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 3] if arr.shape[2] >= 4 else np.any(arr != 0, axis=2)
    if arr.ndim != 2:
        raise ValueError(f"{name} debe ser una mascara 2D o imagen convertible; shape={arr.shape}")
    normalized = arr > 0
    if normalized.shape != shape:
        normalized = _resize_mask(normalized, shape)
    return normalized


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    resized = img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.array(resized) > 0


def _fallback_removed_background(src: np.ndarray, dst: np.ndarray) -> int:
    src_alpha = src[:, :, 3] > 20
    dst_alpha = dst[:, :, 3] <= 20
    if dst_alpha.shape != src_alpha.shape:
        dst_alpha = _resize_mask(dst_alpha, src_alpha.shape)
    return int(np.count_nonzero(src_alpha & dst_alpha))
