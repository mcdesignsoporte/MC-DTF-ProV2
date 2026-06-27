import cv2
import numpy as np
from PIL import Image


def detail_protection_mask(img: Image.Image, threshold: int = 20, radius: int = 5) -> np.ndarray:
    rgba = img.convert("RGBA")
    rgb = np.array(rgba)[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = (gray > max(threshold + 35, 60)).astype(np.uint8) * 255
    kernel = np.ones((max(3, radius), max(3, radius)), np.uint8)
    return cv2.dilate(bright, kernel, iterations=1) > 0


def protect_dark_detail_alpha(alpha: np.ndarray, black_core: np.ndarray, mask: np.ndarray, strength: float = 0.45) -> np.ndarray:
    protected_alpha = alpha.copy()
    protected = black_core & mask
    protected_alpha[protected] = np.maximum(protected_alpha[protected], alpha[protected] * strength)
    return protected_alpha
