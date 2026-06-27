from collections import deque
import numpy as np
from PIL import Image


def clean_alpha(img: Image.Image, alpha_cut: int = 70, despeckle_area: int = 2, edge_contract: int = 0) -> Image.Image:
    import cv2
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    alpha = np.where(alpha >= int(alpha_cut), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    if despeckle_area > 1:
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
        cleaned = np.zeros_like(alpha)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= int(despeckle_area):
                cleaned[labels == i] = 255
        alpha = cleaned

    if edge_contract > 0:
        alpha = cv2.erode(alpha, kernel, iterations=int(edge_contract))

    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def trim_transparent(img: Image.Image, padding: int = 20) -> Image.Image:
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)
    return img.crop((left, top, right, bottom))


def remove_black_background(img: Image.Image, threshold: int = 45, tolerance: int = 22) -> Image.Image:
    """Remove near-black pixels connected to image borders; keeps internal black details better than global keying."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    brightness = rgb.mean(axis=2)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    dark = (brightness <= threshold) & (spread <= max(5, tolerance)) & (alpha > 0)

    h, w = dark.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    def add(y, x):
        if 0 <= y < h and 0 <= x < w and dark[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))

    for x in range(w):
        add(0, x)
        add(h - 1, x)
    for y in range(h):
        add(y, 0)
        add(y, w - 1)

    while q:
        y, x = q.popleft()
        add(y - 1, x)
        add(y + 1, x)
        add(y, x - 1)
        add(y, x + 1)

    arr[visited, 3] = 0
    return Image.fromarray(arr, "RGBA")
