from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
import json

from PIL import Image

from core.background import apply_ai_alpha_to_original, has_transparency, remove_background_ai, resize_for_ai, should_use_ai
from core.black_remove import remove_black_background
from core.background_remove import cleanup_light_background_residue, cleanup_light_edge_matte, remove_background_opencv, remove_dominant_background
from core.clean import clean_alpha_with_stats, trim_transparent
from core.dtf_prepress import DTFPrepressSettings, mask_png_bytes, prepare_dtf
from core.export import build_export_package
from core.logo_tools import (
    black_to_transparent,
    detect_dominant_colors,
    export_color_layers,
    is_photo_like,
    reduce_colors,
    separate_colors,
    unify_similar_colors,
    white_to_transparent,
)
from core.resize import fit_to_print_size, upscale_and_sharpen
from core.logger import get_logger
from core.non_destructive_clean import estimate_art_loss_risk, non_destructive_clean, restore_artwork_pixels
from core.white_protection import protect_white_regions
from core.white_complex import ComplexWhiteSettings, debug_previews, remove_complex_white_background

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
    enable_dtf_prepress: bool
    remove_white_halo: bool
    remove_black_halo: bool
    halo_strength: str
    expand_edge_px: int
    bleed_px: int
    create_cutline: bool
    min_printable_mm: float
    logo_detect_colors: bool
    logo_reduce_colors: bool
    logo_black_to_transparent: bool
    logo_white_to_transparent: bool
    logo_unify_colors: bool
    logo_separate_colors: bool
    logo_export_layers: bool
    logo_max_colors: int
    logo_color_tolerance: int
    max_ai_side: int
    upscale: int
    dpi: int
    width_cm: float
    height_cm: float
    complex_white_preset: str = "Balanceado"
    complex_white_tolerance: int = 58
    complex_white_luminosity: int = 224
    complex_white_saturation: int = 42
    complex_white_preserve_internal: bool = True
    complex_white_halo_cleanup: bool = True
    complex_white_mask_offset: int = 0
    complex_white_alpha_smoothing: int = 1
    complex_white_export_debug: bool = False


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
    dtf_result = None
    logo_report: dict[str, object] | None = None
    logo_layers: dict[str, object] | None = None
    complex_debug: dict[str, object] | None = None
    white_source = work.copy()
    white_level = _white_level(settings, detection)
    fine_level = _fine_level(settings, detection)
    auto_photo = settings.mode_key == "auto" and detection.get("recommended_mode") == "photograph"
    auto_removable_background = settings.mode_key == "auto" and detection.get("recommended_mode") in {"black_bg", "color_bg"}
    if settings.mode_key == "complex_white_bg":
        complex_result = remove_complex_white_background(
            work,
            ComplexWhiteSettings(
                white_tolerance=settings.complex_white_tolerance,
                luminosity_threshold=settings.complex_white_luminosity,
                saturation_threshold=settings.complex_white_saturation,
                preserve_internal_white=settings.complex_white_preserve_internal,
                halo_cleanup=settings.complex_white_halo_cleanup,
                mask_offset=settings.complex_white_mask_offset,
                alpha_smoothing=settings.complex_white_alpha_smoothing,
            ),
        )
        work = complex_result.image
        complex_debug = {
            "stats": complex_result.stats,
            "previews": debug_previews(complex_result),
        }
    elif (settings.use_ai or auto_photo) and should_use_ai(detection, "photograph") and not has_transparency(work):
        ai_img = resize_for_ai(work, max_side=settings.max_ai_side)
        ai_result = remove_background_ai(ai_img, session=session_factory() if session_factory else None)
        work = apply_ai_alpha_to_original(work, ai_result)
    use_non_destructive = (settings.safe_mode or settings.mode_key == "professional_safe") and not auto_removable_background
    if use_non_destructive and not auto_photo and settings.mode_key != "complex_white_bg":
        nd_result = non_destructive_clean(
            work,
            min_area=settings.despeckle_area,
            background_tolerance=settings.color_tolerance,
            safe_mode=True,
        )
        work = nd_result.image
    elif settings.remove_black:
        work = remove_black_background(work, threshold=settings.black_threshold, softness=12, protect_details=settings.protect_details, level=settings.black_level)
    if settings.mode_key != "complex_white_bg" and not use_non_destructive and (settings.remove_color or settings.mode_key == "auto") and not auto_photo:
        work = _remove_auto_background(work, detection, settings)
        if _should_cleanup_light_residue(settings, detection):
            work = cleanup_light_background_residue(work, tolerance=max(settings.color_tolerance + 18, 58))
            work = cleanup_light_edge_matte(work, tolerance=max(settings.color_tolerance + 26, 66))
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
    work, logo_report, logo_layers = _apply_logo_tools(work, settings, detection)
    if settings.enable_dtf_prepress:
        dtf_result = prepare_dtf(work, DTFPrepressSettings(
            enable_dtf_prepress=settings.enable_dtf_prepress,
            remove_white_halo=settings.remove_white_halo,
            remove_black_halo=settings.remove_black_halo,
            halo_strength=settings.halo_strength,
            expand_edge_px=settings.expand_edge_px,
            bleed_px=settings.bleed_px,
            create_cutline=settings.create_cutline,
            min_printable_mm=settings.min_printable_mm,
            dpi=settings.dpi,
        ))
        work = dtf_result.image
    metadata_extra = _metadata_extra(nd_result)
    metadata_extra.update(_dtf_metadata_extra(dtf_result))
    metadata_extra.update(_logo_metadata_extra(logo_report))
    metadata_extra.update(_complex_white_metadata_extra(complex_debug))
    extra_files = _dtf_extra_files(dtf_result)
    extra_files.update(_logo_extra_files(logo_layers, logo_report))
    extra_files.update(_complex_white_extra_files(complex_debug, settings))
    exports = build_export_package(
        work,
        dpi=settings.dpi,
        prefix=prefix,
        mode=settings.mode_key,
        original=img,
        processing_seconds=round(perf_counter() - started, 3),
        metadata_extra=metadata_extra,
        extra_files=extra_files,
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
        "dtf_prepress": dtf_result.metadata if dtf_result else None,
        "alpha_quality": dtf_result.alpha_quality if dtf_result else None,
        "white_halo_mask": dtf_result.white_halo_mask if dtf_result else None,
        "black_halo_mask": dtf_result.black_halo_mask if dtf_result else None,
        "bleed_mask": dtf_result.bleed_mask if dtf_result else None,
        "cutline_mask": dtf_result.cutline_mask if dtf_result else None,
        "small_elements_mask": dtf_result.small_elements_mask if dtf_result else None,
        "small_elements_report": dtf_result.small_elements_report if dtf_result else None,
        "metadata_extra": metadata_extra,
        "dtf_extra_files": extra_files,
        "logo_report": logo_report,
        "logo_palette": logo_report.get("palette", []) if logo_report else None,
        "logo_layers": logo_layers.get("layers", []) if logo_layers else None,
        "complex_white_debug": complex_debug,
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


def _should_cleanup_light_residue(settings: PipelineSettings, detection: dict[str, object]) -> bool:
    recommended = str(detection.get("recommended_mode", ""))
    background = str(detection.get("background", "")).lower()
    return settings.mode_key == "color_bg" or (
        settings.mode_key == "auto"
        and recommended == "color_bg"
        and background in {"blanco", "color dominante"}
    )


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


def _dtf_metadata_extra(dtf_result) -> dict[str, str]:
    if dtf_result is None:
        return {}
    return {
        "halo_white_risk": str(dtf_result.metadata.get("halo_white_risk", False)),
        "halo_black_risk": str(dtf_result.metadata.get("halo_black_risk", False)),
        "small_elements_count": str(dtf_result.metadata.get("small_elements_count", 0)),
        "cutline_ready": str(dtf_result.metadata.get("cutline_ready", False)),
    }


def _dtf_extra_files(dtf_result) -> dict[str, bytes]:
    if dtf_result is None:
        return {}
    files = {
        "alpha_quality.json": json.dumps(dtf_result.alpha_quality, indent=2, ensure_ascii=False).encode("utf-8"),
        "small_elements_report.json": json.dumps(dtf_result.small_elements_report, indent=2, ensure_ascii=False).encode("utf-8"),
    }
    if bool(dtf_result.metadata.get("cutline_ready", False)):
        files["cutline_mask.png"] = mask_png_bytes(dtf_result.cutline_mask)
    return files


def _apply_logo_tools(img: Image.Image, settings: PipelineSettings, detection: dict[str, object]) -> tuple[Image.Image, dict[str, object] | None, dict[str, object] | None]:
    enabled = any([
        settings.logo_detect_colors,
        settings.logo_reduce_colors,
        settings.logo_black_to_transparent,
        settings.logo_white_to_transparent,
        settings.logo_unify_colors,
        settings.logo_separate_colors,
        settings.logo_export_layers,
    ])
    if not enabled:
        return img, None, None
    work = img.convert("RGBA")
    warning = bool(is_photo_like(work) or str(detection.get("recommended_mode", "")) == "photograph")
    if settings.logo_black_to_transparent:
        work = black_to_transparent(work, tolerance=settings.logo_color_tolerance)
    if settings.logo_white_to_transparent:
        work = white_to_transparent(work, tolerance=settings.logo_color_tolerance)
    if settings.logo_unify_colors:
        work = unify_similar_colors(work, tolerance=settings.logo_color_tolerance)
    if settings.logo_reduce_colors:
        work = reduce_colors(work, max_colors=settings.logo_max_colors)
    palette = detect_dominant_colors(work, max_colors=settings.logo_max_colors) if settings.logo_detect_colors or settings.logo_separate_colors or settings.logo_export_layers else []
    layers = export_color_layers(work, max_colors=settings.logo_max_colors) if settings.logo_separate_colors or settings.logo_export_layers else None
    report = {
        "palette": palette,
        "colors_detected": len(palette),
        "photo_warning": warning,
        "layers": int((layers or {}).get("report", {}).get("layers", 0)),
    }
    return work, report, layers


def _logo_metadata_extra(report: dict[str, object] | None) -> dict[str, str]:
    if not report:
        return {}
    return {
        "logo_colors_detected": str(report.get("colors_detected", 0)),
        "logo_layers": str(report.get("layers", 0)),
        "logo_photo_warning": str(report.get("photo_warning", False)),
    }


def _logo_extra_files(layers: dict[str, object] | None, report: dict[str, object] | None) -> dict[str, bytes]:
    if not report:
        return {}
    files: dict[str, bytes] = {
        "palette.json": json.dumps(report.get("palette", []), indent=2, ensure_ascii=False).encode("utf-8"),
        "logo_report.json": json.dumps({key: value for key, value in report.items() if key != "palette"}, indent=2, ensure_ascii=False).encode("utf-8"),
    }
    if layers:
        files.update(layers.get("files", {}))
    return files


def _complex_white_metadata_extra(debug: dict[str, object] | None) -> dict[str, str]:
    if not debug:
        return {}
    stats = dict(debug.get("stats") or {})
    return {f"complex_white_{key}": str(value) for key, value in stats.items()}


def _complex_white_extra_files(debug: dict[str, object] | None, settings: PipelineSettings) -> dict[str, bytes]:
    if not debug or not settings.complex_white_export_debug:
        return {}
    from core.export import png_bytes

    previews = dict(debug.get("previews") or {})
    files: dict[str, bytes] = {}
    names = {
        "alpha_mask": "debug_alpha_mask.png",
        "background_mask": "debug_background_mask.png",
        "preview_black": "debug_preview_black.png",
        "preview_red": "debug_preview_red.png",
    }
    for key, filename in names.items():
        image = previews.get(key)
        if image is not None:
            files[filename] = png_bytes(image, dpi=settings.dpi)
    return files
