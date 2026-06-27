from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from core.background import apply_ai_alpha_to_original, has_transparency, remove_background_ai, resize_for_ai, should_use_ai
from core.black_remove import remove_black_background
from core.clean import clean_alpha, trim_transparent
from core.export import build_export_package
from core.resize import fit_to_print_size, upscale_and_sharpen


@dataclass(frozen=True)
class PipelineSettings:
    """Processing settings independent from Streamlit widgets."""

    mode_key: str
    use_ai: bool
    remove_black: bool
    clean_enabled: bool
    trim: bool
    alpha_cut: int
    despeckle_area: int
    edge_contract: int
    black_threshold: int
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
    work = img.convert("RGBA")
    if settings.use_ai and should_use_ai(detection, settings.mode_key) and not has_transparency(work):
        ai_img = resize_for_ai(work, max_side=settings.max_ai_side)
        ai_result = remove_background_ai(ai_img, session=session_factory() if session_factory else None)
        work = apply_ai_alpha_to_original(work, ai_result)
    if settings.remove_black:
        work = remove_black_background(work, threshold=settings.black_threshold, softness=12, protect_details=settings.protect_details)
    if settings.clean_enabled:
        work = clean_alpha(work, alpha_cut=settings.alpha_cut, despeckle_area=settings.despeckle_area, edge_contract=settings.edge_contract)
    if settings.trim:
        work = trim_transparent(work, padding=20)
    work = fit_to_print_size(work, width_cm=settings.width_cm, height_cm=settings.height_cm, dpi=settings.dpi)
    if settings.upscale > 1:
        work = upscale_and_sharpen(work, scale=settings.upscale)
    exports = build_export_package(work, dpi=settings.dpi, prefix=prefix, mode=settings.mode_key)
    return {"image": work, **exports}
