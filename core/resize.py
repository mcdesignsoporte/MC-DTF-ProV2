from PIL import Image, ImageFilter


def upscale_and_sharpen(img: Image.Image, scale: int = 1) -> Image.Image:
    scale = max(1, min(int(scale), 4))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    if scale > 1:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))
    return img


def fit_to_print_size(img: Image.Image, width_cm: float = 0, height_cm: float = 0, dpi: int = 300) -> Image.Image:
    if width_cm <= 0 and height_cm <= 0:
        return img
    img = img.convert("RGBA")
    w, h = img.size

    if width_cm > 0 and height_cm > 0:
        px_w = int(width_cm / 2.54 * dpi)
        px_h = int(height_cm / 2.54 * dpi)
        img.thumbnail((px_w, px_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (px_w, px_h), (255, 255, 255, 0))
        canvas.alpha_composite(img, ((px_w - img.width)//2, (px_h - img.height)//2))
        return canvas

    if width_cm > 0:
        px_w = int(width_cm / 2.54 * dpi)
        ratio = px_w / w
        return img.resize((px_w, max(1, int(h * ratio))), Image.Resampling.LANCZOS)

    px_h = int(height_cm / 2.54 * dpi)
    ratio = px_h / h
    return img.resize((max(1, int(w * ratio)), px_h), Image.Resampling.LANCZOS)
