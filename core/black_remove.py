import cv2
import numpy as np
from PIL import Image


def remove_black_background(
    img: Image.Image,
    threshold: int = 20,
    softness: int = 12,
    protect_details: bool = True,
) -> Image.Image:
    """Remove near-black background while preserving bright/colored design details.

    This is a color-key method, not AI. It is best for DTF designs on black canvas.
    - Pixels where RGB channels are all below threshold become transparent.
    - A soft transition removes dark halos.
    - If protect_details=True, black pixels near non-black edges are less aggressively removed.
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3].astype(np.float32)

    maxc = rgb.max(axis=2).astype(np.float32)
    minc = rgb.min(axis=2).astype(np.float32)
    chroma = maxc - minc

    # Pure / near-pure black mask.
    black_core = maxc <= threshold

    # Soft edge: fade almost-black values instead of hard cutting everything.
    fade_limit = threshold + max(1, softness)
    fade = np.clip((maxc - threshold) / max(1, softness), 0, 1)

    new_alpha = alpha.copy()
    new_alpha[black_core] = 0
    near_black = (maxc > threshold) & (maxc < fade_limit) & (chroma < 18)
    new_alpha[near_black] = new_alpha[near_black] * fade[near_black]

    if protect_details:
        # Preserve black strokes that are part of text/vehicle details by detecting adjacency to bright pixels.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bright = (gray > max(threshold + 35, 60)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        around_bright = cv2.dilate(bright, kernel, iterations=1) > 0

        # Very dark pixels close to bright text/splash often form outlines; preserve some alpha.
        protected = black_core & around_bright
        new_alpha[protected] = np.maximum(new_alpha[protected], alpha[protected] * 0.45)

    arr[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")
