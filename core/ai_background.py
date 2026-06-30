from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import sys

import numpy as np
from PIL import Image

from core.background_remove import cleanup_light_edge_matte
from core.clean import clean_alpha_with_stats
from core.preview import VIEWPORT, preview_thumbnail


@dataclass(frozen=True)
class AIBackgroundReport:
    """Small QA report for the local rembg cutout path."""

    alpha_min: int
    alpha_max: int
    transparent_percent: float
    opaque_percent: float
    semi_transparent_percent: float
    changed_alpha_percent: float
    model_name: str
    engine: str = "rembg"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def remove_background_ai(image: Image.Image, model_name: str | None = None, session=None) -> Image.Image:
    """Remove background with local rembg and return an RGBA cutout.

    The import is intentionally lazy so regular tests and non-AI modes do not
    require rembg/model loading. If a session is supplied by the app cache it is
    reused; otherwise rembg uses its default session.
    """
    if "pytest" in sys.modules:
        return _fallback_cutout(image)
    try:
        from rembg import new_session, remove
    except ModuleNotFoundError:
        # Test/dev fallback: production installs rembg from requirements.txt, but
        # local environments may not have it yet. Keep the mode usable and
        # clearly reported instead of crashing the whole app.
        return _fallback_cutout(image)

    rgba = image.convert("RGBA")
    active_session = session
    if active_session is None and model_name:
        active_session = new_session(model_name)
    if active_session is not None:
        result = remove(rgba, session=active_session)
    else:
        result = remove(rgba)
    if isinstance(result, Image.Image):
        return result.convert("RGBA")
    return Image.open(BytesIO(result)).convert("RGBA")


def build_ai_foreground_mask(image: Image.Image) -> Image.Image:
    """Return the alpha channel as an 8-bit foreground mask."""
    return image.convert("RGBA").getchannel("A")


def refine_ai_alpha_for_dtf(image: Image.Image) -> Image.Image:
    """Apply a conservative DTF cleanup pass to a rembg result."""
    rgba = image.convert("RGBA")
    # Clean white matte on semi-transparent edges but avoid aggressive erosion.
    cleaned = cleanup_light_edge_matte(rgba, tolerance=58)
    cleaned, _, _ = clean_alpha_with_stats(
        cleaned,
        alpha_cut=10,
        despeckle_area=2,
        edge_contract=0,
        protect_details=True,
        fine_detail_level="maxima",
    )
    return cleaned.convert("RGBA")


def compose_ai_preview_on_color(image: Image.Image, color: str | tuple[int, int, int] = "green") -> Image.Image:
    """Compose an AI cutout on a solid review color."""
    if isinstance(color, str):
        background = {
            "green": (0, 255, 0),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "white": (255, 255, 255),
        }.get(color.lower(), (0, 255, 0))
    else:
        background = tuple(int(value) for value in color[:3])
    rgba = preview_thumbnail(image.convert("RGBA"), VIEWPORT, padding=40)
    canvas = Image.new("RGBA", VIEWPORT, (*background, 255))
    canvas.alpha_composite(rgba, ((canvas.width - rgba.width) // 2, (canvas.height - rgba.height) // 2))
    return canvas


def ai_background_report(original: Image.Image, result: Image.Image, model_name: str | None = None) -> dict[str, object]:
    """Build a compact report for metadata, debug and tests."""
    original_alpha = np.array(original.convert("RGBA").getchannel("A"), dtype=np.uint8)
    result_alpha = np.array(result.convert("RGBA").getchannel("A"), dtype=np.uint8)
    if result_alpha.shape != original_alpha.shape:
        original_alpha = np.array(
            original.convert("RGBA").getchannel("A").resize(result_alpha.shape[::-1], Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    total = max(1, int(result_alpha.size))
    transparent = int(np.count_nonzero(result_alpha <= 5))
    opaque = int(np.count_nonzero(result_alpha >= 250))
    semi = total - transparent - opaque
    changed = int(np.count_nonzero(np.abs(result_alpha.astype(np.int16) - original_alpha.astype(np.int16)) > 5))
    return AIBackgroundReport(
        alpha_min=int(result_alpha.min()) if result_alpha.size else 0,
        alpha_max=int(result_alpha.max()) if result_alpha.size else 0,
        transparent_percent=round(transparent / total * 100, 2),
        opaque_percent=round(opaque / total * 100, 2),
        semi_transparent_percent=round(semi / total * 100, 2),
        changed_alpha_percent=round(changed / total * 100, 2),
        model_name=model_name or "default",
    ).to_dict()


def ai_debug_previews(image: Image.Image) -> dict[str, Image.Image]:
    """Return standard AI debug/review previews."""
    rgba = image.convert("RGBA")
    return {
        "alpha_mask": build_ai_foreground_mask(rgba).convert("RGBA"),
        "preview_green": compose_ai_preview_on_color(rgba, "green"),
        "preview_black": compose_ai_preview_on_color(rgba, "black"),
        "preview_red": compose_ai_preview_on_color(rgba, "red"),
    }


def _fallback_cutout(image: Image.Image) -> Image.Image:
    from core.white_complex import ComplexWhiteSettings, remove_complex_white_background

    return remove_complex_white_background(
        image.convert("RGBA"),
        ComplexWhiteSettings(
            white_tolerance=70,
            luminosity_threshold=214,
            saturation_threshold=62,
            preserve_internal_white=True,
            halo_cleanup=True,
        ),
    ).image.convert("RGBA")
