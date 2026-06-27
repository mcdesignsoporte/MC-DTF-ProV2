from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageDraw


def _fit_preview(img: Image.Image, max_side: int) -> Image.Image:
    rgba = img.convert("RGBA")
    if max_side <= 0 or max(rgba.size) <= max_side:
        return rgba
    preview = rgba.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return preview


def _checkerboard(size: tuple[int, int], tile: int = 32) -> Image.Image:
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
    if "mug" in label.lower():
        draw.ellipse((w * 0.24, h * 0.78, w * 0.78, h * 0.9), fill=(0, 0, 0, 32))
        draw.rounded_rectangle((w * 0.25, h * 0.2, w * 0.7, h * 0.82), radius=30, fill=(255, 255, 255, 255), outline=(180, 180, 180, 255), width=3)
        draw.ellipse((w * 0.64, h * 0.34, w * 0.86, h * 0.62), outline=(180, 180, 180, 255), width=8)
        if "beer" in label.lower():
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
    if "transparent" in lower:
        return _checkerboard(size)
    if "black shirt" in lower:
        return _garment_background(size, (12, 12, 12), "Black shirt")
    if "white shirt" in lower:
        return _garment_background(size, (250, 250, 250), "White shirt")
    if "hoodie" in lower:
        return _garment_background(size, (38, 38, 42), "Hoodie")
    if "mug" in lower:
        return _object_background(size, mode)
    if "sticker" in lower:
        return _object_background(size, "Sticker")
    return Image.new("RGBA", size, (130, 130, 130, 255))


def composite_preview(img: Image.Image, mode: str = "Transparent", max_side: int = 1400) -> Image.Image:
    """Create fast commercial previews without changing export resolution."""
    rgba = _fit_preview(img, max_side)
    bg = _background(rgba.size, mode)
    bg.alpha_composite(rgba)
    return bg


def before_after_preview(before: Image.Image, after: Image.Image, mode: str = "Transparent", max_side: int = 1200) -> Image.Image:
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
    before_alpha = _fit_preview(before, max_side).getchannel("A")
    after_alpha = _fit_preview(after, max_side).getchannel("A")
    if before_alpha.size != after_alpha.size:
        after_alpha = after_alpha.resize(before_alpha.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(before_alpha, after_alpha)
    heat = Image.new("RGBA", before_alpha.size, (255, 210, 0, 0))
    heat.putalpha(diff)
    base = composite_preview(before, "Black shirt", max_side=max_side)
    base.alpha_composite(heat)
    return base


def _center(img: Image.Image, height: int) -> Image.Image:
    panel = Image.new("RGBA", (img.width, height), (0, 0, 0, 0))
    panel.alpha_composite(img, (0, (height - img.height) // 2))
    return panel
