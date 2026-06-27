import numpy as np
from PIL import Image


def _fit_preview(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0 or max(img.size) <= max_side:
        return img
    preview = img.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return preview


def _checkerboard(size: tuple[int, int], tile: int = 32) -> Image.Image:
    width, height = size
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    arr = np.empty((height, width, 4), dtype=np.uint8)
    arr[mask] = (170, 170, 170, 255)
    arr[~mask] = (220, 220, 220, 255)
    return Image.fromarray(arr, "RGBA")


def composite_preview(img: Image.Image, mode: str = "Gris", max_side: int = 1800) -> Image.Image:
    rgba = img.convert("RGBA")
    rgba = _fit_preview(rgba, max_side)
    mode = (mode or "Gris").lower()
    if "negro" in mode:
        bg_color = (0, 0, 0, 255)
    elif "blanco" in mode:
        bg_color = (255, 255, 255, 255)
    elif "transparente" in mode:
        bg = _checkerboard(rgba.size)
        bg.alpha_composite(rgba)
        return bg
    else:
        bg_color = (128, 128, 128, 255)
    bg = Image.new("RGBA", rgba.size, bg_color)
    bg.alpha_composite(rgba)
    return bg
