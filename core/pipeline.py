from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from core.background import apply_ai_alpha_to_original, has_transparency, remove_background_ai, resize_for_ai, should_use_ai
from core.black_remove import remove_black_background
from core.background_remove import remove_background_opencv, remove_dominant_background
from core.clean import clean_alpha_with_stats, trim_transparent
from core.export import build_export_package
from core.resize import fit_to_print_size, upscale_and_sharpen
from core.logger import get_logger
from core.non_destructive_clean import estimate_art_loss_risk, non_destructive_clean, restore_artwork_pixels
from core.white_protection import protect_white_regions

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
    black_level: str
    color_tolerance: int
    protect_details: bool
    protect_white_details: bool
    white_protection_level: str
    fine_detail_level: str
    safe_mode: bool
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
    started = perf_counter()
    logger.info("Processing %s with mode=%s", prefix, settings.mode_key)
    work = img.convert("RGBA")
    white_stats: dict[str, object] | None = None
    white_mask = None
    fine_stats: dict[str, object] | None = None
    fine_mask = None
    nd_result = None
    white_source = work.copy()
    white_level = _white_level(settings, detection)
    fine_level = _fine_level(settings, detection)
    auto_photo = settings.mode_key == "auto" and detection.get("recommended_mode") == "photograph"
    if (settings.use_ai or auto_photo) and should_use_ai(detection, "photograph") and not has_transparency(work):
        ai_img = resize_for_ai(work, max_side=settings.max_ai_side)
        ai_result = remove_background_ai(ai_img, session=session_factory() if session_factory else None)
        work = apply_ai_alpha_to_original(work, ai_result)
    use_non_destructive = settings.safe_mode or settings.mode_key == "professional_safe"
    if use_non_destructive and not auto_photo:
        nd_result = non_destructive_clean(
            work,
            min_area=settings.despeckle_area,
            background_tolerance=settings.color_tolerance,
            safe_mode=True,
        )
        work = nd_result.image
    elif settings.remove_black:
        work = remove_black_background(work, threshold=settings.black_threshold, softness=12, protect_details=settings.protect_details, level=settings.black_level)
    if not use_non_destructive and (settings.remove_color or settings.mode_key == "auto") and not auto_photo:
        work = _remove_auto_background(work, detection, settings)
    if settings.protect_white_details and not auto_photo:
        work, white_mask, stats = protect_white_regions(white_source, work, level=white_level)
        white_stats = stats.to_dict()
    if settings.clean_enabled:
        work, stats, fine_mask = clean_alpha_with_stats(
            work,
            alpha_cut=settings.alpha_cut,
            despeckle_area=settings.despeckle_area,
            edge_contract=settings.edge_contract,
            protect_details=settings.protect_details,
            fine_detail_level=fine_level,
        )
        fine_stats = stats.to_dict()
    if nd_result is not None:
        risk_after_clean = estimate_art_loss_risk(white_source, work, nd_result.artwork_mask)
        if bool(risk_after_clean.get("risk_detected", False)):
            work, restored_after_clean = restore_artwork_pixels(white_source, work, nd_result.artwork_mask)
            nd_result.stats["restored_pixels"] = int(nd_result.stats.get("restored_pixels", 0)) + int(restored_after_clean.sum())
            nd_result.risk.clear()
            nd_result.risk.update(estimate_art_loss_risk(white_source, work, nd_result.artwork_mask))
    if settings.trim:
        work = trim_transparent(work, padding=20)
    work = fit_to_print_size(work, width_cm=settings.width_cm, height_cm=settings.height_cm, dpi=settings.dpi)
    if settings.upscale > 1:
        work = upscale_and_sharpen(work, scale=settings.upscale)
    metadata_extra = _metadata_extra(nd_result)
    exports = build_export_package(
        work,
        dpi=settings.dpi,
        prefix=prefix,
        mode=settings.mode_key,
        original=img,
        processing_seconds=round(perf_counter() - started, 3),
        metadata_extra=metadata_extra,
    )
    return {
        "image": work,
        "white_protection": white_stats,
        "white_mask": white_mask,
        "fine_detail_protection": fine_stats,
        "fine_detail_mask": fine_mask,
        "artwork_mask": nd_result.artwork_mask if nd_result else None,
        "background_mask": nd_result.background_mask if nd_result else None,
        "doubtful_mask": nd_result.doubtful_mask if nd_result else None,
        "restored_mask": nd_result.restored_mask if nd_result else None,
        "art_loss_risk": nd_result.risk if nd_result else None,
        "non_destructive_stats": nd_result.stats if nd_result else None,
        **exports,
    }


def _remove_auto_background(img: Image.Image, detection: dict[str, object], settings: PipelineSettings) -> Image.Image:
    """Choose color, chroma, OpenCV, or hybrid background removal."""
    mode = str(detection.get("recommended_mode", ""))
    uniformity = float(detection.get("background_uniformity", 0))
    if mode == "black_bg":
        return remove_black_background(img, threshold=settings.black_threshold, protect_details=settings.protect_details, level=settings.black_level)
    if uniformity >= 52 or mode == "color_bg":
        return remove_dominant_background(img, tolerance=settings.color_tolerance, protect_details=settings.protect_details)
    if settings.mode_key == "auto" and detection.get("type") == "Fotografia":
        return img
    return remove_background_opencv(img, protect_details=settings.protect_details)


def _white_level(settings: PipelineSettings, detection: dict[str, object]) -> str:
    """Choose manual or automatic white protection level."""
    detected_level = str(detection.get("white_protection_level", "")).lower()
    if detected_level in {"maxima", "maxima_auto"}:
        return "maxima"
    return settings.white_protection_level


def _fine_level(settings: PipelineSettings, detection: dict[str, object]) -> str:
    """Choose manual or automatic fine-detail protection level."""
    kind = str(detection.get("type", "")).lower()
    mode = str(detection.get("recommended_mode", "")).lower()
    if any(term in kind for term in ["logo", "diseno", "sticker", "caricatura", "anime", "vector"]):
        return "maxima"
    if mode in {"preserve_artwork", "color_bg", "black_bg", "dark_artwork"}:
        return "maxima"
    return settings.fine_detail_level


def _metadata_extra(nd_result) -> dict[str, str]:
    if nd_result is None:
        return {}
    return {
        "riesgo_perdida": str(nd_result.risk.get("risk_detected", False)),
        "pixeles_restaurados": str(nd_result.stats.get("restored_pixels", 0)),
        "fondo_eliminado": str(nd_result.stats.get("background_removed", 0)),
        "arte_protegido": str(nd_result.stats.get("artwork_protected", 0)),
    }
