from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from core.background_remove import cleanup_light_background_residue, cleanup_light_edge_matte, dominant_background_color
from core.preview import composite_preview


@dataclass(frozen=True)
class ComplexWhiteSettings:
    """Manual controls for complex artwork on white backgrounds."""

    white_tolerance: int = 58
    luminosity_threshold: int = 224
    saturation_threshold: int = 42
    preserve_internal_white: bool = True
    halo_cleanup: bool = True
    mask_offset: int = 0
    alpha_smoothing: int = 1


@dataclass(frozen=True)
class ComplexWhiteResult:
    """Processed image and debug masks for QA."""

    image: Image.Image
    alpha_mask: Image.Image
    background_mask: Image.Image
    preview_black: Image.Image
    preview_red: Image.Image
    stats: dict[str, object]


def remove_complex_white_background(
    image: Image.Image,
    settings: ComplexWhiteSettings | None = None,
) -> ComplexWhiteResult:
    """Remove only exterior white background from complex illustration artwork."""
    options = settings or ComplexWhiteSettings()
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    bg = dominant_background_color(rgba)
    candidate = _white_background_candidate(rgb, alpha, bg, options)
    background = _edge_connected(candidate)
    background = _apply_mask_offset(background, options.mask_offset)
    if options.preserve_internal_white:
        background &= _edge_connected(background | (alpha <= 20))

    out = arr.copy()
    before_white = _opaque_light_pixels(out)
    out[background, 3] = 0
    result = Image.fromarray(out, "RGBA")

    if options.halo_cleanup:
        result = cleanup_light_background_residue(result, bg, tolerance=max(options.white_tolerance, 58))
        result = cleanup_light_edge_matte(result, bg, tolerance=max(options.white_tolerance + 10, 68))
    if options.alpha_smoothing > 0:
        result = _smooth_alpha(result, options.alpha_smoothing)

    after = np.array(result.convert("RGBA"))
    stats = {
        "background_removed": int(np.count_nonzero(background)),
        "opaque_light_before": int(before_white),
        "opaque_light_after": int(_opaque_light_pixels(after)),
        "background_color": "#{:02x}{:02x}{:02x}".format(*bg),
    }
    return ComplexWhiteResult(
        image=result,
        alpha_mask=_alpha_mask_image(result),
        background_mask=_mask_image(background),
        preview_black=compose_on_solid(result, (0, 0, 0)),
        preview_red=compose_on_solid(result, (180, 0, 0)),
        stats=stats,
    )


def compose_on_solid(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Composite artwork over a solid color for DTF inspection."""
    rgba = image.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (*color, 255))
    bg.alpha_composite(rgba)
    return bg


def debug_previews(result: ComplexWhiteResult) -> dict[str, Image.Image]:
    """Return named debug previews for UI and export."""
    return {
        "alpha_mask": result.alpha_mask,
        "background_mask": result.background_mask,
        "preview_black": result.preview_black,
        "preview_red": result.preview_red,
    }


def preview_for_ui(image: Image.Image, background: str) -> Image.Image:
    """Create a fitted debug preview without changing export resolution."""
    return composite_preview(image, background)


def _white_background_candidate(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg: tuple[int, int, int],
    settings: ComplexWhiteSettings,
) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    distance = np.linalg.norm(rgb.astype(np.int16) - np.array(bg, dtype=np.int16), axis=2)
    return (
        (alpha > 20)
        & (gray >= settings.luminosity_threshold)
        & (hsv[:, :, 1] <= settings.saturation_threshold)
        & (distance <= settings.white_tolerance)
    )


def _edge_connected(mask: np.ndarray) -> np.ndarray:
    work = mask.astype(np.uint8) * 255
    flood = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    h, w = mask.shape
    for point in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if work[point[1], point[0]]:
            cv2.floodFill(work, flood, point, 128)
    return work == 128


def _apply_mask_offset(mask: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return mask
    radius = abs(offset)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.dilate if offset > 0 else cv2.erode
    return op(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _smooth_alpha(image: Image.Image, radius: int) -> Image.Image:
    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    if not np.any((alpha > 0) & (alpha < 255)):
        return image.convert("RGBA")
    blurred = cv2.GaussianBlur(alpha, (radius * 2 + 1, radius * 2 + 1), 0)
    edge = cv2.Canny(alpha, 20, 80) > 0
    arr[edge, 3] = blurred[edge]
    return Image.fromarray(arr, "RGBA")


def _opaque_light_pixels(arr: np.ndarray) -> int:
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return int(np.count_nonzero((alpha > 220) & (gray >= 224) & (hsv[:, :, 1] <= 42)))


def _alpha_mask_image(image: Image.Image) -> Image.Image:
    alpha = image.convert("RGBA").getchannel("A")
    return Image.merge("RGBA", (alpha, alpha, alpha, Image.new("L", alpha.size, 255)))


def _mask_image(mask: np.ndarray) -> Image.Image:
    out = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    out[mask] = (0, 220, 120, 255)
    return Image.fromarray(out, "RGBA")
