import cv2
import numpy as np
from PIL import Image


def clean_alpha(img: Image.Image, alpha_cut: int = 70, despeckle_area: int = 2, edge_contract: int = 0) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]

    # Convert semitransparency to solid/transparent.
    alpha = np.where(alpha >= alpha_cut, 255, 0).astype(np.uint8)

    # Remove isolated trash but keep fine details with low defaults.
    if despeckle_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
        cleaned = np.zeros_like(alpha)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= despeckle_area:
                cleaned[labels == i] = 255
        alpha = cleaned

    if edge_contract > 0:
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=edge_contract)

    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def trim_transparent(img: Image.Image, padding: int = 20) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return rgba
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgba.width, right + padding)
    bottom = min(rgba.height, bottom + padding)
    return rgba.crop((left, top, right, bottom))
