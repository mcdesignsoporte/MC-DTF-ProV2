from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AlphaStats:
    """Alpha channel distribution for DTF QA."""

    transparent_percent: float
    solid_percent: float
    semitransparent_percent: float


@dataclass(frozen=True)
class QualityReport:
    """DTF print readiness report."""

    score: int
    stars: str
    checks: dict[str, bool]
    alpha: AlphaStats
    status: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable quality report."""
        data = asdict(self)
        data["alpha"] = asdict(self.alpha)
        return data


def alpha_histogram(img: Image.Image) -> AlphaStats:
    """Measure transparent, solid, and semi-transparent alpha percentages."""
    alpha = np.array(img.convert("RGBA").getchannel("A"))
    total = alpha.size
    transparent = np.count_nonzero(alpha <= 5) / total * 100
    solid = np.count_nonzero(alpha >= 250) / total * 100
    semi = 100 - transparent - solid
    return AlphaStats(round(transparent, 2), round(solid, 2), round(semi, 2))


def evaluate_dtf_quality(img: Image.Image, dpi: int = 300) -> QualityReport:
    """Evaluate whether the image is ready for DTF printing."""
    rgba = img.convert("RGBA")
    alpha = alpha_histogram(rgba)
    checks = {
        "Sin fondo": alpha.transparent_percent > 1,
        "Bordes limpios": _edge_quality(rgba) >= 0.72,
        "Resolucion suficiente": min(rgba.size) >= 900,
        "Alfa limpio": alpha.semitransparent_percent <= 18,
        "Tamano de impresion valido": _print_area_cm(rgba, dpi) >= 20,
    }
    score = int(round(sum(checks.values()) / len(checks) * 100))
    stars = "★" * max(1, round(score / 20)) + "☆" * (5 - max(1, round(score / 20)))
    status = "Lista para imprimir" if score >= 80 else "Revisar antes de imprimir"
    return QualityReport(score, stars, checks, alpha, status)


def _edge_quality(img: Image.Image) -> float:
    alpha = np.array(img.getchannel("A"))
    edges = cv2.Canny(alpha, 40, 120)
    if not np.any(edges):
        return 1.0
    semi = (alpha > 5) & (alpha < 250)
    edge_area = cv2.dilate((edges > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    semi_on_edge = np.count_nonzero(semi & edge_area)
    edge_count = max(1, np.count_nonzero(edge_area))
    return 1.0 - min(1.0, semi_on_edge / edge_count)


def _print_area_cm(img: Image.Image, dpi: int) -> float:
    width_cm = img.width / dpi * 2.54
    height_cm = img.height / dpi * 2.54
    return width_cm * height_cm
