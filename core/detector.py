from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectionResult:
    """Structured image analysis used by the app and tests."""

    type: str
    recommended_mode: str
    use_ai: bool
    black_percent: float
    white_percent: float
    edge_density: float
    transparency_percent: float
    estimated_seconds: float
    resolution: str

    def to_dict(self) -> dict[str, object]:
        """Return a Streamlit-friendly dictionary."""
        return asdict(self)


def _rgba_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA"))


def _transparency_percent(arr: np.ndarray) -> float:
    return float((arr[:, :, 3] < 250).mean() * 100)


def _black_percent(rgb: np.ndarray) -> float:
    mask = (rgb[:, :, 0] < 24) & (rgb[:, :, 1] < 24) & (rgb[:, :, 2] < 24)
    return float(mask.mean() * 100)


def _white_percent(rgb: np.ndarray) -> float:
    mask = (rgb[:, :, 0] > 238) & (rgb[:, :, 1] > 238) & (rgb[:, :, 2] > 238)
    return float(mask.mean() * 100)


def _edge_density(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    max_side = max(gray.shape)
    if max_side > 900:
        scale = 900 / max_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 70, 150)
    return float((edges > 0).mean() * 100)


def _classify(transparent: float, black: float, white: float, edges: float) -> tuple[str, str, bool]:
    if transparent > 2:
        return "transparent png", "transparent_png", False
    if black > 55:
        return "black background", "black_bg", False
    if black > 35 and edges > 8:
        return "dark artwork", "dark_artwork", False
    if white > 70 and edges > 4:
        return "white background", "dtf_ready", False
    if edges > 13 and max(black, white) < 55:
        return "dtf artwork", "preserve_artwork", False
    if edges > 8:
        return "logo", "preserve_artwork", False
    return "photograph", "photograph", True


def _estimate_seconds(width: int, height: int, use_ai: bool) -> float:
    megapixels = (width * height) / 1_000_000
    base = 1.0 + megapixels * 0.35
    return round(base + (3.5 if use_ai else 0.8), 1)


def detect(img: Image.Image) -> dict[str, object]:
    """Detect image type and recommend the safest processing mode."""
    rgba = _rgba_array(img)
    rgb = rgba[:, :, :3]
    transparent = _transparency_percent(rgba)
    black = _black_percent(rgb)
    white = _white_percent(rgb)
    edges = _edge_density(rgb)
    kind, mode, use_ai = _classify(transparent, black, white, edges)

    result = DetectionResult(
        type=kind,
        recommended_mode=mode,
        use_ai=use_ai,
        black_percent=round(black, 2),
        white_percent=round(white, 2),
        edge_density=round(edges, 2),
        transparency_percent=round(transparent, 2),
        estimated_seconds=_estimate_seconds(img.width, img.height, use_ai),
        resolution=f"{img.width} x {img.height}px",
    )
    return result.to_dict()
