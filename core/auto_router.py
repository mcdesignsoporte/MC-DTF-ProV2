from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from core.background_remove import edge_light_residue_score
from core.detector import detect
from core.residue_refine import ResidueRefineSettings, detect_light_residue_components
from core.residue_refine import InternalLightResidueSettings, detect_internal_light_residue_components


@dataclass(frozen=True)
class AutoPilotDecision:
    """Autonomous DTF routing decision with review guardrails."""

    case_type: str
    recommended_mode: str
    confidence: float
    risk_level: str
    needs_manual_review: bool
    reason: str
    suggested_next_step: str
    traffic_light: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def autopilot_route(image: Image.Image, detection: dict[str, object] | None = None) -> dict[str, object]:
    """Analyze an image and choose the safest existing processing route."""
    data = detection or detect(image)
    case_type = _case_type(image, data)
    mode = _mode_for_case(case_type)
    confidence = _confidence_for_case(case_type, data)
    risk = _risk_for_case(case_type, data, image)
    light = _traffic_light(risk, confidence)
    return AutoPilotDecision(
        case_type=case_type,
        recommended_mode=mode,
        confidence=round(confidence, 2),
        risk_level=risk,
        needs_manual_review=light != "green",
        reason=_reason(case_type, data),
        suggested_next_step=_suggested_next_step(case_type, light),
        traffic_light=light,
    ).to_dict()


def autopilot_quality_check(image: Image.Image) -> dict[str, object]:
    """Measure DTF risks visible on a black review background."""
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    residue_components = detect_light_residue_components(
        rgba,
        ResidueRefineSettings(luminosity_threshold=218, saturation_threshold=58, min_area=6, max_area=200000),
    )
    residue_count = sum(1 for item in residue_components if item.classification in {"borrar", "ambiguo"})
    internal_components = detect_internal_light_residue_components(
        rgba,
        InternalLightResidueSettings(min_area=4, max_area=900, dark_neighbor_threshold=30),
    )
    internal_count = sum(1 for item in internal_components if item.suggested_action in {"borrar", "revisar"})
    internal_area = sum(item.area for item in internal_components if item.suggested_action in {"borrar", "revisar"})
    internal_score = min(100, internal_count * 18 + int(internal_area / max(1, alpha.size) * 5000))
    halo = edge_light_residue_score(rgba)
    transparent = round(float(np.count_nonzero(alpha <= 5) / alpha.size * 100), 2)
    small_components = _small_alpha_components(alpha)
    high_risk = halo > 18 or residue_count > 0 or small_components > 24 or internal_score >= 35
    medium_risk = halo > 7 or small_components > 8 or transparent < 1 or internal_score >= 16
    traffic = "red" if high_risk else "yellow" if medium_risk else "green"
    return {
        "traffic_light": traffic,
        "halo_light_score": round(float(halo), 2),
        "opaque_light_residue_components": int(residue_count),
        "transparency_percent": transparent,
        "small_components": int(small_components),
        "internal_light_residue_score": int(internal_score),
        "internal_light_residue_components": int(internal_count),
        "internal_light_residue_area": int(internal_area),
        "risk_level": "alto" if traffic == "red" else "medio" if traffic == "yellow" else "bajo",
        "needs_manual_review": traffic != "green",
        "suggested_next_step": _qa_next_step(traffic, internal_score),
    }


def _case_type(image: Image.Image, data: dict[str, object]) -> str:
    width, height = image.size
    low_resolution = min(width, height) < 450
    transparency = float(data.get("transparency_percent", 0) or 0)
    if transparency > 2:
        return "png_transparent"
    recommended = str(data.get("recommended_mode", ""))
    background = str(data.get("background", "")).lower()
    white = float(data.get("white_percent", 0) or 0)
    black = float(data.get("black_percent", 0) or 0)
    uniformity = float(data.get("background_uniformity", 0) or 0)
    edge = float(data.get("edge_density", 0) or 0)
    color_count = float(data.get("color_count", 0) or 0)
    logo = float(data.get("logo_score", 0) or 0)
    text = float(data.get("text_score", 0) or 0)
    splash = float(data.get("splash_score", 0) or 0)
    noise = float(data.get("noise_score", 0) or 0)
    near_white = _is_near_white(data.get("dominant_color")) and white >= 38 and uniformity >= 52
    if black > 55 or recommended == "black_bg":
        return "black_background"
    if recommended == "dark_artwork" or (black > 30 and str(data.get("type", "")).lower().find("oscuro") >= 0):
        return "dark_design"
    if recommended == "photograph" or bool(data.get("use_ai", False)):
        return "photo"
    if background == "blanco" or white > 55 or near_white:
        return _white_case(uniformity, edge, color_count, logo, text, splash, noise, recommended)
    if recommended == "color_bg" or uniformity > 58:
        return "color_background"
    if low_resolution:
        return "low_resolution"
    if edge > 10 or logo > 50 or text > 35:
        return "high_risk_art"
    return "high_risk_art"


def _white_case(
    uniformity: float,
    edge: float,
    color_count: float,
    logo: float,
    text: float,
    splash: float = 0,
    noise: float = 0,
    recommended: str = "",
) -> str:
    # Splash/noise alone is not enough: simple low-resolution art can look
    # splashy because of hard synthetic edges. Require real complexity signals
    # before routing to AI.
    complex_score = 0
    complex_score += 2 if edge > 5 else 0
    complex_score += 2 if color_count > 90 else 0
    complex_score += 2 if logo > 38 else 0
    complex_score += 1 if text > 18 else 0
    complex_score += 2 if text > 70 else 0
    complex_score += 1 if splash > 35 and (edge > 5 or color_count > 40 or text > 18 or logo > 38) else 0
    complex_score += 1 if splash > 35 and text > 35 else 0
    complex_score += 1 if noise > 14 and (edge > 5 or color_count > 40) else 0
    complex_score += 1 if recommended == "color_bg" and uniformity < 88 and (edge > 5 or color_count > 40 or text > 18) else 0
    complex_score += 1 if uniformity < 76 and (edge > 5 or color_count > 40) else 0
    return "white_background_complex" if complex_score >= 3 else "white_background_simple"


def _is_near_white(value: object) -> bool:
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        return False
    try:
        red, green, blue = (int(text[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return False
    chroma = max(red, green, blue) - min(red, green, blue)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance >= 225 and chroma <= 18


def _mode_for_case(case_type: str) -> str:
    return {
        "png_transparent": "transparent_png",
        "white_background_simple": "color_bg",
        "white_background_complex": "ai_background",
        "black_background": "black_bg",
        "color_background": "color_bg",
        "dark_design": "dark_artwork",
        "photo": "photograph",
        "low_resolution": "professional_safe",
        "high_risk_art": "ai_background",
    }.get(case_type, "professional_safe")


def _confidence_for_case(case_type: str, data: dict[str, object]) -> float:
    base = float(data.get("confidence", 72) or 72)
    if case_type in {"png_transparent", "black_background", "white_background_simple"}:
        return min(98.0, base + 6)
    if case_type in {"white_background_complex", "high_risk_art"}:
        return min(88.0, base)
    if case_type == "low_resolution":
        return min(70.0, base)
    return base


def _risk_for_case(case_type: str, data: dict[str, object], image: Image.Image) -> str:
    if case_type in {"white_background_complex", "high_risk_art", "low_resolution"}:
        return "alto"
    if case_type == "color_background" and float(data.get("noise_score", 0) or 0) > 18 and float(data.get("edge_density", 0) or 0) > 2:
        return "medio"
    if case_type in {"photo", "dark_design"}:
        return "medio"
    if case_type == "png_transparent":
        return autopilot_quality_check(image)["risk_level"]
    return "bajo" if float(data.get("confidence", 0) or 0) >= 72 else "medio"


def _traffic_light(risk: str, confidence: float) -> str:
    if risk == "alto":
        return "red"
    if risk == "medio" or confidence < 78:
        return "yellow"
    return "green"


def _reason(case_type: str, data: dict[str, object]) -> str:
    background = data.get("background", "-")
    return f"Caso {case_type}; fondo={background}; confianza detector={data.get('confidence', 0)}%"


def _suggested_next_step(case_type: str, light: str) -> str:
    if case_type == "white_background_complex":
        return "Probar Recorte IA y revisar sobre verde/negro"
    if case_type == "high_risk_art":
        return "Probar Recorte IA o revisar visualmente antes de exportar"
    if light == "red":
        return "Revisar visualmente antes de exportar"
    if light == "yellow":
        return "Revisar sobre fondo negro"
    return "Exportar"


def _small_alpha_components(alpha: np.ndarray) -> int:
    foreground = alpha > 20
    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    count = 0
    for idx in range(1, labels_count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 1 <= area <= 16:
            count += 1
    return count


def _qa_next_step(traffic: str, internal_score: int) -> str:
    if internal_score >= 16:
        return "Refinar residuos blancos internos"
    if traffic == "red":
        return "Usar refinamiento asistido"
    if traffic == "yellow":
        return "Revisar sobre fondo negro"
    return "Exportar"
