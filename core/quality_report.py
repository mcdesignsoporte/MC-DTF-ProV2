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
    removed_background = int(np.count_nonzero(background_mask)) if background_mask is not None else int(np.count_nonzero((src[:, :, 3] > 20) & (dst[:, :, 3] <= 20)))
    protected_art = int(np.count_nonzero(artwork_mask)) if artwork_mask is not None else 0
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
