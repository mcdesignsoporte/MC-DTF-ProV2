from PIL import Image, ImageFilter


def fit_to_print_size(img: Image.Image, width_cm: float = 0.0, height_cm: float = 0.0, dpi: int = 300) -> Image.Image:
    img = img.convert("RGBA")
    if width_cm <= 0 and height_cm <= 0:
        return img
    if width_cm > 0 and height_cm > 0:
        target = (int(width_cm / 2.54 * dpi), int(height_cm / 2.54 * dpi))
        canvas = Image.new("RGBA", target, (255, 255, 255, 0))
        temp = img.copy()
        temp.thumbnail(target, Image.Resampling.LANCZOS)
        x = (target[0] - temp.width) // 2
        y = (target[1] - temp.height) // 2
        canvas.alpha_composite(temp, (x, y))
        return canvas
    if width_cm > 0:
        new_w = int(width_cm / 2.54 * dpi)
        scale = new_w / img.width
        new_h = int(img.height * scale)
    else:
        new_h = int(height_cm / 2.54 * dpi)
        scale = new_h / img.height
        new_w = int(img.width * scale)
    return img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)


def upscale_and_sharpen(img: Image.Image, scale: int = 1) -> Image.Image:
    scale = max(1, min(int(scale), 4))
    img = img.convert("RGBA")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=1.1, percent=115, threshold=3))
