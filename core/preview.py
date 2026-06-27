from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


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
    bg = Image.new("RGBA", size, (238, 238, 238, 255))
    draw = ImageDraw.Draw(bg)
    w, h = size
    shirt = [(w * 0.22, h * 0.18), (w * 0.78, h * 0.18), (w * 0.9, h * 0.45), (w * 0.74, h * 0.55), (w * 0.72, h * 0.9), (w * 0.28, h * 0.9), (w * 0.26, h * 0.55), (w * 0.1, h * 0.45)]
    draw.polygon(shirt, fill=(*color, 255))
    draw.text((18, 18), label, fill=(120, 120, 120, 255))
    return bg


def _object_background(size: tuple[int, int], label: str) -> Image.Image:
    bg = Image.new("RGBA", size, (245, 245, 242, 255))
    draw = ImageDraw.Draw(bg)
    w, h = size
    if "mug" in label.lower():
        draw.rounded_rectangle((w * 0.25, h * 0.2, w * 0.7, h * 0.82), radius=30, fill=(255, 255, 255, 255), outline=(180, 180, 180, 255), width=3)
        draw.ellipse((w * 0.64, h * 0.34, w * 0.86, h * 0.62), outline=(180, 180, 180, 255), width=8)
    else:
        draw.rounded_rectangle((w * 0.16, h * 0.18, w * 0.84, h * 0.82), radius=20, fill=(255, 255, 255, 255), outline=(190, 190, 190, 255), width=3)
    draw.text((18, 18), label, fill=(120, 120, 120, 255))
    return bg


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
