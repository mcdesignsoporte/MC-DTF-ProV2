from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


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
    background: str
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
    kind, mode, use_ai = _classify(metrics)
    return DetectionResult(
        type=kind,
        recommended_mode=mode,
        use_ai=use_ai,
        background=_background_label(metrics),
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


def _classify(m: dict[str, float]) -> tuple[str, str, bool]:
    if m["transparency_percent"] > 2:
        return "PNG transparente", "transparent_png", False
    if m["black_percent"] > 55 and m["text_score"] < 65:
        return "Fondo negro", "black_bg", False
    if m["black_percent"] > 32 and (m["shadow_score"] > 8 or m["splash_score"] > 18):
        return "Diseno oscuro", "dark_artwork", False
    if m["white_percent"] > 68 and m["edge_density"] > 3:
        return "Fondo blanco", "dtf_ready", False
    if m["logo_score"] > 45 and m["text_score"] > 12:
        return "Logo", "preserve_artwork", False
    if m["edge_density"] > 10 or m["splash_score"] > 18:
        return "Diseno DTF", "preserve_artwork", False
    return "Fotografia", "photograph", True


def _background_label(m: dict[str, float]) -> str:
    if m["transparency_percent"] > 2:
        return "transparente"
    if m["black_percent"] > 45:
        return "negro"
    if m["white_percent"] > 60:
        return "blanco"
    return "mixto"


def _estimate_seconds(width: int, height: int, use_ai: bool) -> float:
    megapixels = (width * height) / 1_000_000
    return round(0.8 + megapixels * 0.22 + (2.8 if use_ai else 0.4), 1)
