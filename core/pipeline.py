from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from core.background import apply_ai_alpha_to_original, has_transparency, remove_background_ai, resize_for_ai, should_use_ai
from core.black_remove import remove_black_background
from core.background_remove import remove_background_opencv, remove_dominant_background
from core.clean import clean_alpha, trim_transparent
from core.export import build_export_package
from core.resize import fit_to_print_size, upscale_and_sharpen
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineSettings:
    """Processing settings independent from Streamlit widgets."""

    mode_key: str
    use_ai: bool
    remove_black: bool
    remove_color: bool
    clean_enabled: bool
    trim: bool
    alpha_cut: int
    despeckle_area: int
    edge_contract: int
    black_threshold: int
    color_tolerance: int
    protect_details: bool
    max_ai_side: int
    upscale: int
    dpi: int
    width_cm: float
    height_cm: float


def process_artwork(
    img: Image.Image,
    detection: dict[str, object],
    settings: PipelineSettings,
    session_factory: Callable[[], object] | None = None,
    prefix: str = "mc_dtf_pro_v4",
) -> dict[str, object]:
    """Run the production image pipeline and preserve export resolution unless requested."""
    logger.info("Processing %s with mode=%s", prefix, settings.mode_key)
    work = img.convert("RGBA")
    auto_photo = settings.mode_key == "auto" and detection.get("recommended_mode") == "photograph"
    if (settings.use_ai or auto_photo) and should_use_ai(detection, "photograph") and not has_transparency(work):
        ai_img = resize_for_ai(work, max_side=settings.max_ai_side)
        ai_result = remove_background_ai(ai_img, session=session_factory() if session_factory else None)
        work = apply_ai_alpha_to_original(work, ai_result)
    if settings.remove_black:
        work = remove_black_background(work, threshold=settings.black_threshold, softness=12, protect_details=settings.protect_details)
    if (settings.remove_color or settings.mode_key == "auto") and not auto_photo:
        work = _remove_auto_background(work, detection, settings)
    if settings.clean_enabled:
        work = clean_alpha(work, alpha_cut=settings.alpha_cut, despeckle_area=settings.despeckle_area, edge_contract=settings.edge_contract)
    if settings.trim:
        work = trim_transparent(work, padding=20)
    work = fit_to_print_size(work, width_cm=settings.width_cm, height_cm=settings.height_cm, dpi=settings.dpi)
    if settings.upscale > 1:
        work = upscale_and_sharpen(work, scale=settings.upscale)
    exports = build_export_package(work, dpi=settings.dpi, prefix=prefix, mode=settings.mode_key)
    return {"image": work, **exports}


def _remove_auto_background(img: Image.Image, detection: dict[str, object], settings: PipelineSettings) -> Image.Image:
    """Choose color, chroma, OpenCV, or hybrid background removal."""
    mode = str(detection.get("recommended_mode", ""))
    uniformity = float(detection.get("background_uniformity", 0))
    if mode == "black_bg":
        return remove_black_background(img, threshold=settings.black_threshold, protect_details=settings.protect_details)
    if uniformity >= 52 or mode == "color_bg":
        return remove_dominant_background(img, tolerance=settings.color_tolerance, protect_details=settings.protect_details)
    if settings.mode_key == "auto" and detection.get("type") == "Fotografia":
        return img
    return remove_background_opencv(img, protect_details=settings.protect_details)
