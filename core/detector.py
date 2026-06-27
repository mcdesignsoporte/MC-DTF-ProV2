from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image
from core.background_remove import background_uniformity, dominant_background_color


@dataclass(frozen=True)
class DetectionResult:
    """Image intelligence payload consumed by the UI and pipeline."""

    type: str
    recommended_mode: str
    use_ai: bool
    black_percent: float
    white_percent: float
    edge_density: float
    transparency_percent: float
    text_score: float
    logo_score: float
    noise_score: float
    shadow_score: float
    splash_score: float
    color_count: float
    density_score: float
    confidence: float
    background: str
    dominant_color: str
    background_uniformity: float
    white_protection_level: str
    estimated_seconds: float
    resolution: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dictionary."""
        return asdict(self)


def detect(img: Image.Image) -> dict[str, object]:
    """Analyze artwork and recommend the safest processing mode."""
    arr = np.array(img.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    gray = _gray(rgb)
    metrics = _metrics(rgb, alpha, gray)
    dominant = dominant_background_color(img)
    metrics["background_uniformity"] = background_uniformity(img, dominant)
    kind, mode, use_ai, confidence = _classify(metrics)
    return DetectionResult(
        type=kind,
        recommended_mode=mode,
        use_ai=use_ai,
        background=_background_label(metrics),
        dominant_color=_hex_color(dominant),
        white_protection_level=_white_protection_level(kind, metrics),
        confidence=round(confidence, 2),
        estimated_seconds=_estimate_seconds(img.width, img.height, use_ai),
        resolution=f"{img.width} x {img.height}px",
        **{key: round(value, 2) for key, value in metrics.items()},
    ).to_dict()


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _metrics(rgb: np.ndarray, alpha: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    edges = _edges(gray)
    return {
        "black_percent": _percent((rgb[:, :, 0] < 24) & (rgb[:, :, 1] < 24) & (rgb[:, :, 2] < 24)),
        "white_percent": _percent((rgb[:, :, 0] > 238) & (rgb[:, :, 1] > 238) & (rgb[:, :, 2] > 238)),
        "edge_density": _percent(edges > 0),
        "transparency_percent": _percent(alpha < 250),
        "text_score": _text_score(gray, edges),
        "logo_score": _logo_score(rgb, edges),
        "noise_score": _noise_score(gray),
        "shadow_score": _shadow_score(rgb, gray),
        "splash_score": _splash_score(rgb, edges),
        "color_count": _color_count(rgb),
        "density_score": _density_score(alpha, edges),
    }


def _edges(gray: np.ndarray) -> np.ndarray:
    preview = _downscale_gray(gray, 900)
    return cv2.Canny(preview, 70, 150)


def _downscale_gray(gray: np.ndarray, max_side: int) -> np.ndarray:
    side = max(gray.shape)
    if side <= max_side:
        return gray
    scale = max_side / side
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _percent(mask: np.ndarray) -> float:
    return float(mask.mean() * 100)


def _text_score(gray: np.ndarray, edges: np.ndarray) -> float:
    binary = cv2.adaptiveThreshold(_downscale_gray(gray, 900), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 8)
    components, _, stats, _ = cv2.connectedComponentsWithStats((binary < 128).astype(np.uint8), 8)
    thin = sum(1 for i in range(1, components) if 8 <= stats[i, cv2.CC_STAT_AREA] <= 800)
    return min(100.0, thin * 0.8 + _percent(edges > 0) * 2)


def _logo_score(rgb: np.ndarray, edges: np.ndarray) -> float:
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    solid_color = _percent(chroma > 35)
    return min(100.0, solid_color * 0.8 + _percent(edges > 0) * 2.2)


def _noise_score(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(_downscale_gray(gray, 900), cv2.CV_32F)
    return min(100.0, float(np.std(lap)) / 3)


def _shadow_score(rgb: np.ndarray, gray: np.ndarray) -> float:
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    shadow = (gray > 25) & (gray < 115) & (chroma < 28)
    return _percent(shadow)


def _splash_score(rgb: np.ndarray, edges: np.ndarray) -> float:
    bright = (rgb[:, :, 0] > 210) & (rgb[:, :, 1] > 210) & (rgb[:, :, 2] > 210)
    return min(100.0, _percent(bright) * 0.8 + _percent(edges > 0))


def _color_count(rgb: np.ndarray) -> float:
    preview = rgb
    side = max(preview.shape[:2])
    if side > 420:
        scale = 420 / side
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    quantized = (preview.reshape(-1, 3) // 32) * 32
    return float(len(np.unique(quantized, axis=0)))


def _density_score(alpha: np.ndarray, edges: np.ndarray) -> float:
    opaque = _percent(alpha > 20)
    return min(100.0, opaque * 0.5 + _percent(edges > 0) * 3)


def _classify(m: dict[str, float]) -> tuple[str, str, bool, float]:
    if m["transparency_percent"] > 2:
        return "PNG Transparente", "transparent_png", False, min(98.0, 72 + m["transparency_percent"])
    if m["black_percent"] > 55 and m["text_score"] < 65:
        return "Fondo negro", "black_bg", False, min(96.0, 60 + m["black_percent"] * 0.45)
    if m["background_uniformity"] > 62 and m["edge_density"] > 2:
        return "Fondo de color", "professional_safe", False, min(95.0, 50 + m["background_uniformity"] * 0.5)
    if m["black_percent"] > 32 and (m["shadow_score"] > 8 or m["splash_score"] > 18):
        return "Diseno oscuro", "dark_artwork", False, 84.0
    if m["white_percent"] > 68 and m["edge_density"] > 3:
        return "Fondo blanco", "color_bg", False, min(94.0, 55 + m["white_percent"] * 0.45)
    if m["logo_score"] > 45 and m["text_score"] > 12 and m["color_count"] < 180:
        return "Logo", "professional_safe", False, min(92.0, 62 + m["logo_score"] * 0.35)
    if m["edge_density"] > 10 or m["splash_score"] > 18 or m["density_score"] > 55:
        return "Diseno DTF", "professional_safe", False, 82.0
    return "Fotografia", "photograph", True, 78.0


def _background_label(m: dict[str, float]) -> str:
    if m["transparency_percent"] > 2:
        return "transparente"
    if m["black_percent"] > 45:
        return "negro"
    if m["white_percent"] > 60:
        return "blanco"
    if m["background_uniformity"] > 62:
        return "color dominante"
    return "textura/degradado"


def _white_protection_level(kind: str, m: dict[str, float]) -> str:
    artwork_types = {"Logo", "Diseno DTF", "Diseno oscuro", "Fondo de color"}
    if kind in artwork_types or m["logo_score"] > 38 or m["text_score"] > 18 or m["color_count"] < 160:
        return "maxima"
    return "normal"


def _hex_color(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _estimate_seconds(width: int, height: int, use_ai: bool) -> float:
    megapixels = (width * height) / 1_000_000
    return round(0.8 + megapixels * 0.22 + (2.8 if use_ai else 0.4), 1)
