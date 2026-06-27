from PIL import Image, ImageFilter


def fit_to_print_size(img: Image.Image, width_cm: float = 0, height_cm: float = 0, dpi: int = 300) -> Image.Image:
    rgba = img.convert("RGBA")
    if width_cm <= 0 and height_cm <= 0:
        return rgba

    if width_cm > 0 and height_cm > 0:
        px_w = int(width_cm / 2.54 * dpi)
        px_h = int(height_cm / 2.54 * dpi)
        canvas = Image.new("RGBA", (px_w, px_h), (255, 255, 255, 0))
        copy = rgba.copy()
        copy.thumbnail((px_w, px_h), Image.Resampling.LANCZOS)
        x = (px_w - copy.width) // 2
        y = (px_h - copy.height) // 2
        canvas.alpha_composite(copy, (x, y))
        return canvas

    if width_cm > 0:
        px_w = int(width_cm / 2.54 * dpi)
        scale = px_w / rgba.width
        px_h = int(rgba.height * scale)
        return rgba.resize((px_w, px_h), Image.Resampling.LANCZOS)

    px_h = int(height_cm / 2.54 * dpi)
    scale = px_h / rgba.height
    px_w = int(rgba.width * scale)
    return rgba.resize((px_w, px_h), Image.Resampling.LANCZOS)


def upscale_and_sharpen(img: Image.Image, scale: int = 1) -> Image.Image:
    rgba = img.convert("RGBA")
    scale = max(1, min(int(scale), 4))
    if scale > 1:
        rgba = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.LANCZOS)
    return rgba.filter(ImageFilter.UnsharpMask(radius=1.1, percent=110, threshold=3))
