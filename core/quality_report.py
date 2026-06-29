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
    internal_residue_stats: dict[str, object] | None = None,
    autopilot: dict[str, object] | None = None,
    autopilot_quality: dict[str, object] | None = None,
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
    gate = quality_gate(
        dtf_score=max(0, int(dtf_score)),
        halo_white=halo_white,
        halo_black=halo_black,
        small_count=small_count,
        internal_residue_stats=internal_residue_stats,
        autopilot=autopilot,
        autopilot_quality=autopilot_quality,
    )
    warnings.extend(gate["reasons"])
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
        "readiness_status": gate["status"],
        "readiness_reasons": gate["reasons"],
        "readiness_blocked": gate["blocked"],
    }


def quality_gate(
    dtf_score: int,
    halo_white: bool = False,
    halo_black: bool = False,
    small_count: int = 0,
    internal_residue_stats: dict[str, object] | None = None,
    autopilot: dict[str, object] | None = None,
    autopilot_quality: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return an honest DTF readiness status from every known QA signal."""
    reasons: list[str] = []
    internal = internal_residue_stats or {}
    if dtf_score < 80:
        reasons.append(f"Score DTF {dtf_score}% menor al minimo 80%.")
    if halo_white:
        reasons.append("Halo blanco en riesgo.")
    if halo_black:
        reasons.append("Halo negro en riesgo.")
    if small_count > 0:
        reasons.append(f"{small_count} elementos pequenos requieren revision.")
    _internal_residue_reasons(internal, reasons)
    _autopilot_reasons(autopilot, autopilot_quality, reasons)
    status = _readiness_status(dtf_score, reasons)
    return {"status": status, "reasons": reasons, "blocked": status != "Lista para imprimir"}


def _internal_residue_reasons(stats: dict[str, object], reasons: list[str]) -> None:
    review = int(stats.get("internal_components_review", 0) or 0)
    detected = int(stats.get("internal_components_detected", 0) or 0)
    removed = int(stats.get("internal_components_removed", 0) or 0)
    protected_reasons = dict(stats.get("internal_protection_reasons", {}) or {})
    area_grande_ids = [key for key, value in protected_reasons.items() if value == "area_grande"]
    if review > 0:
        reasons.append(f"{review} componentes internos en revision.")
    if area_grande_ids:
        reasons.append("Componentes grandes protegidos requieren revision manual.")
    if detected > removed and detected >= 8:
        pending = detected - removed
        reasons.append(f"{pending} residuos internos pendientes despues del refinamiento.")


def _autopilot_reasons(
    autopilot: dict[str, object] | None,
    autopilot_quality: dict[str, object] | None,
    reasons: list[str],
) -> None:
    for payload in [autopilot or {}, autopilot_quality or {}]:
        if payload.get("traffic_light") == "red":
            reasons.append("AutoPilot esta en rojo.")
        if bool(payload.get("needs_manual_review", False)):
            reasons.append("AutoPilot requiere revision manual.")


def _readiness_status(dtf_score: int, reasons: list[str]) -> str:
    if not reasons:
        return "Lista para imprimir"
    if dtf_score < 50:
        return "No apta todavia"
    hard = any(
        term in reason
        for reason in reasons
        for term in ["Score DTF", "Halo", "revision", "AutoPilot esta en rojo", "pendientes"]
    )
    return "Revision obligatoria" if hard else "Revision recomendada"


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
