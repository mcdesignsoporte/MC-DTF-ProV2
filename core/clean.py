import cv2
import numpy as np
from PIL import Image


def clean_alpha(img: Image.Image, alpha_cut: int = 80, despeckle_area: int = 3, edge_contract: int = 0) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]

    # Transparencia binaria: evita semitransparencias problemáticas en DTF.
    alpha = np.where(alpha >= alpha_cut, 255, 0).astype(np.uint8)

    # Limpieza suave de máscara.
    kernel = np.ones((3, 3), np.uint8)
    if despeckle_area > 1:
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Quitar componentes pequeños.
    if despeckle_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
        cleaned = np.zeros_like(alpha)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= despeckle_area:
                cleaned[labels == i] = 255
        alpha = cleaned

    # Contraer borde contra halos, opcional.
    if edge_contract > 0:
        alpha = cv2.erode(alpha, kernel, iterations=int(edge_contract))

    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def trim_transparent(img: Image.Image, padding: int = 20) -> Image.Image:
    img = img.convert("RGBA")
    bbox = img.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)
    return img.crop((left, top, right, bottom))
