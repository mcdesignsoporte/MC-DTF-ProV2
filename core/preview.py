from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageDraw

VIEWPORT = (1080, 760)


def preview_thumbnail(img: Image.Image, viewport: tuple[int, int] = VIEWPORT, padding: int = 40) -> Image.Image:
    """Create an auto-zoomed thumbnail that fits the visible preview area."""
    rgba = img.convert("RGBA")
    max_w = max(120, viewport[0] - padding * 2)
    max_h = max(120, viewport[1] - padding * 2)
    preview = rgba.copy()
    preview.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return preview


def _fit_preview(img: Image.Image, max_side: int) -> Image.Image:
    rgba = img.convert("RGBA")
    if max_side <= 0 or max(rgba.size) <= max_side:
        return rgba
    return preview_thumbnail(rgba, (max_side, max_side), padding=0)


def checkerboard(size: tuple[int, int], tile: int = 28) -> Image.Image:
    """Generate a fast NumPy checkerboard for the viewport."""
    width, height = size
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    arr = np.empty((height, width, 4), dtype=np.uint8)
    arr[mask] = (176, 176, 176, 255)
    arr[~mask] = (224, 224, 224, 255)
    return Image.fromarray(arr, "RGBA")


def _garment_background(size: tuple[int, int], color: tuple[int, int, int], label: str) -> Image.Image:
    bg = _studio_background(size)
    draw = ImageDraw.Draw(bg)
    w, h = size
    draw.ellipse((w * 0.18, h * 0.82, w * 0.82, h * 0.94), fill=(0, 0, 0, 36))
    shirt = [(w * 0.22, h * 0.18), (w * 0.78, h * 0.18), (w * 0.9, h * 0.45), (w * 0.74, h * 0.55), (w * 0.72, h * 0.9), (w * 0.28, h * 0.9), (w * 0.26, h * 0.55), (w * 0.1, h * 0.45)]
    draw.polygon(shirt, fill=(*color, 255))
    draw.line((w * 0.37, h * 0.18, w * 0.5, h * 0.3, w * 0.63, h * 0.18), fill=(120, 120, 120, 90), width=3)
    draw.text((18, 18), label, fill=(120, 120, 120, 255))
    return bg


def _object_background(size: tuple[int, int], label: str) -> Image.Image:
    bg = _studio_background(size)
    draw = ImageDraw.Draw(bg)
    w, h = size
    lower = label.lower()
    if "mug" in lower or "taza" in lower or "tarro" in lower:
        draw.ellipse((w * 0.24, h * 0.78, w * 0.78, h * 0.9), fill=(0, 0, 0, 32))
        draw.rounded_rectangle((w * 0.25, h * 0.2, w * 0.7, h * 0.82), radius=30, fill=(255, 255, 255, 255), outline=(180, 180, 180, 255), width=3)
        draw.ellipse((w * 0.64, h * 0.34, w * 0.86, h * 0.62), outline=(180, 180, 180, 255), width=8)
        if "beer" in lower or "tarro" in lower:
            draw.rectangle((w * 0.29, h * 0.22, w * 0.66, h * 0.36), fill=(231, 177, 67, 220))
    else:
        draw.ellipse((w * 0.2, h * 0.78, w * 0.8, h * 0.9), fill=(0, 0, 0, 25))
        draw.rounded_rectangle((w * 0.16, h * 0.18, w * 0.84, h * 0.82), radius=20, fill=(255, 255, 255, 255), outline=(190, 190, 190, 255), width=3)
    draw.text((18, 18), label, fill=(120, 120, 120, 255))
    return bg


def _studio_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    y = np.linspace(245, 225, height, dtype=np.uint8)[:, None]
    arr = np.repeat(y, width, axis=1)
    bg = np.dstack([arr, arr, arr, np.full_like(arr, 255)])
    return Image.fromarray(bg, "RGBA")


def _background(size: tuple[int, int], mode: str) -> Image.Image:
    lower = mode.lower()
    if "transparent" in lower or "transparente" in lower:
        return checkerboard(size)
    if "black shirt" in lower or "playera negra" in lower:
        return _garment_background(size, (12, 12, 12), "Playera negra")
    if "white shirt" in lower or "playera blanca" in lower:
        return _garment_background(size, (250, 250, 250), "Playera blanca")
    if "hoodie" in lower or "sudadera" in lower:
        return _garment_background(size, (38, 38, 42), "Sudadera")
    if "mug" in lower or "taza" in lower or "tarro" in lower:
        return _object_background(size, mode)
    if "sticker" in lower or "calcomania" in lower or "calcomanía" in lower:
        return _object_background(size, "Calcomania")
    return Image.new("RGBA", size, (130, 130, 130, 255))


def composite_preview(img: Image.Image, mode: str = "Transparente", max_side: int = 1400) -> Image.Image:
    """Create fast commercial previews without changing export resolution."""
    rgba = preview_thumbnail(img, (max_side, min(max_side, 900)))
    canvas = _background(VIEWPORT, mode)
    x = (canvas.width - rgba.width) // 2
    y = (canvas.height - rgba.height) // 2
    canvas.alpha_composite(rgba, (x, y))
    return canvas


def alpha_preview(img: Image.Image, max_side: int = 1400) -> Image.Image:
    """Render the alpha channel as a fitted grayscale preview."""
    thumb = preview_thumbnail(img, (max_side, min(max_side, 900)))
    alpha = thumb.getchannel("A")
    alpha_rgba = Image.merge("RGBA", (alpha, alpha, alpha, Image.new("L", alpha.size, 255)))
    canvas = Image.new("RGBA", VIEWPORT, (24, 24, 24, 255))
    canvas.alpha_composite(alpha_rgba, ((canvas.width - alpha.width) // 2, (canvas.height - alpha.height) // 2))
    return canvas


def before_after_preview(before: Image.Image, after: Image.Image, mode: str = "Transparente", max_side: int = 1200) -> Image.Image:
    """Create a side-by-side visual comparison with a center divider."""
    left = composite_preview(before, mode=mode, max_side=max_side // 2)
    right = composite_preview(after, mode=mode, max_side=max_side // 2)
    height = max(left.height, right.height)
    canvas = Image.new("RGBA", (left.width + right.width + 6, height), (32, 32, 32, 255))
    canvas.alpha_composite(_center(left, height), (0, 0))
    canvas.alpha_composite(_center(right, height), (left.width + 6, 0))
    return canvas


def alpha_difference_preview(before: Image.Image, after: Image.Image, max_side: int = 1200) -> Image.Image:
    """Show changed alpha as a quick QA layer."""
    before_alpha = preview_thumbnail(before, VIEWPORT, padding=40).getchannel("A")
    after_alpha = preview_thumbnail(after, VIEWPORT, padding=40).getchannel("A")
    if before_alpha.size != after_alpha.size:
        after_alpha = after_alpha.resize(before_alpha.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(before_alpha, after_alpha)
    heat = Image.new("RGBA", before_alpha.size, (255, 210, 0, 0))
    heat.putalpha(diff)
    base = composite_preview(before, "Playera negra", max_side=max_side)
    x = (base.width - heat.width) // 2
    y = (base.height - heat.height) // 2
    base.alpha_composite(heat, (x, y))
    return base


def _center(img: Image.Image, height: int) -> Image.Image:
    panel = Image.new("RGBA", (img.width, height), (0, 0, 0, 0))
    panel.alpha_composite(img, (0, (height - img.height) // 2))
    return panel
