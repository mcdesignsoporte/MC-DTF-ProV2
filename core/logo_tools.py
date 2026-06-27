from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def detect_dominant_colors(image: Image.Image, max_colors: int = 8) -> list[dict[str, object]]:
    """Detect dominant opaque colors in a logo-like image."""
    rgba = np.array(image.convert("RGBA"))
    pixels = rgba[rgba[:, :, 3] > 20][:, :3]
    if len(pixels) == 0:
        return []
    colors, counts = _quantized_palette(pixels, max_colors)
    total = max(1, counts.sum())
    return [
        {"rgb": tuple(int(v) for v in color), "hex": _hex(color), "percent": round(float(count / total * 100), 2)}
        for color, count in zip(colors, counts)
    ]


def reduce_colors(image: Image.Image, max_colors: int = 8) -> Image.Image:
    """Reduce opaque pixels to a limited palette while preserving alpha."""
    rgba = np.array(image.convert("RGBA"))
    opaque = rgba[:, :, 3] > 20
    if not np.any(opaque):
        return image.convert("RGBA")
    pixels = rgba[opaque][:, :3]
    palette, _ = _kmeans_palette(pixels, max_colors)
    nearest = _nearest_palette(pixels, palette)
    rgba[opaque, :3] = palette[nearest]
    return Image.fromarray(rgba, "RGBA")


def replace_color_with_transparency(image: Image.Image, target_color: tuple[int, int, int], tolerance: int = 24) -> Image.Image:
    """Make pixels near target color transparent without touching other alpha."""
    rgba = np.array(image.convert("RGBA"))
    rgb = rgba[:, :, :3].astype(np.int16)
    target = np.array(target_color, dtype=np.int16)
    distance = np.linalg.norm(rgb - target, axis=2)
    mask = (distance <= tolerance) & (rgba[:, :, 3] > 0)
    rgba[mask, 3] = 0
    return Image.fromarray(rgba, "RGBA")


def black_to_transparent(image: Image.Image, tolerance: int = 24) -> Image.Image:
    """Convert near-black pixels to transparency."""
    return replace_color_with_transparency(image, (0, 0, 0), tolerance)


def white_to_transparent(image: Image.Image, tolerance: int = 24) -> Image.Image:
    """Convert near-white pixels to transparency."""
    return replace_color_with_transparency(image, (255, 255, 255), tolerance)


def unify_similar_colors(image: Image.Image, tolerance: int = 24) -> Image.Image:
    """Snap similar colors to a shared representative color."""
    max_colors = max(2, min(16, int(256 / max(8, tolerance))))
    return reduce_colors(image, max_colors=max_colors)


def separate_colors(image: Image.Image, max_colors: int = 8) -> list[dict[str, object]]:
    """Create one transparent layer per dominant color."""
    reduced = reduce_colors(image, max_colors=max_colors)
    rgba = np.array(reduced.convert("RGBA"))
    opaque = rgba[:, :, 3] > 20
    if not np.any(opaque):
        return []
    unique, counts = np.unique(rgba[opaque][:, :3], axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:max_colors]
    layers = []
    for index in order:
        color = unique[index]
        mask = opaque & np.all(rgba[:, :, :3] == color, axis=2)
        layer = np.zeros_like(rgba)
        layer[mask] = rgba[mask]
        layers.append({"color": tuple(int(v) for v in color), "hex": _hex(color), "image": Image.fromarray(layer, "RGBA"), "pixels": int(np.count_nonzero(mask))})
    return layers


def export_color_layers(image: Image.Image, max_colors: int = 8) -> dict[str, object]:
    """Export separated color layers as PNG bytes and palette metadata."""
    layers = separate_colors(image, max_colors=max_colors)
    files: dict[str, bytes] = {}
    palette = []
    for index, layer in enumerate(layers, start=1):
        files[f"layer_{index:02d}_color.png"] = _png_bytes(layer["image"])
        palette.append({"index": index, "hex": layer["hex"], "rgb": layer["color"], "pixels": layer["pixels"]})
    report = {"layers": len(layers), "palette": palette}
    return {"files": files, "palette": palette, "report": report, "layers": layers}


def is_photo_like(image: Image.Image) -> bool:
    """Return whether color separation is likely not recommended."""
    rgba = np.array(image.convert("RGBA"))
    pixels = rgba[rgba[:, :, 3] > 20][:, :3]
    if len(pixels) == 0:
        return False
    sample = pixels[:: max(1, len(pixels) // 5000)]
    quantized = (sample // 16) * 16
    return len(np.unique(quantized, axis=0)) > 96


def _quantized_palette(pixels: np.ndarray, max_colors: int) -> tuple[np.ndarray, np.ndarray]:
    quantized = (pixels // 16) * 16
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:max_colors]
    return colors[order].astype(np.uint8), counts[order]


def _kmeans_palette(pixels: np.ndarray, max_colors: int) -> tuple[np.ndarray, np.ndarray]:
    sample = pixels.astype(np.float32)
    if len(sample) > 12000:
        sample = sample[:: len(sample) // 12000]
    k = max(1, min(max_colors, len(np.unique(sample.astype(np.uint8), axis=0))))
    if k == 1:
        color = np.mean(sample, axis=0, keepdims=True).astype(np.uint8)
        return color, np.array([len(pixels)])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    counts = np.bincount(labels.ravel(), minlength=k)
    order = np.argsort(counts)[::-1]
    return centers[order], counts[order]


def _nearest_palette(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(pixels[:, None, :].astype(np.int16) - palette[None, :, :].astype(np.int16), axis=2)
    return np.argmin(distance, axis=1)


def _png_bytes(image: Image.Image) -> bytes:
    bio = BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def _hex(color: np.ndarray | tuple[int, int, int]) -> str:
    values = [int(v) for v in color]
    return "#{:02x}{:02x}{:02x}".format(*values)
