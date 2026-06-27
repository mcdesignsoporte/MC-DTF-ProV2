from PIL import Image


def has_transparency(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    alpha = img.getchannel("A")
    return alpha.getextrema()[0] < 255


def resize_for_ai(img: Image.Image, max_side: int = 1600) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def get_rembg_session():
    from rembg import new_session
    return new_session("u2net")


def remove_background_ai(img: Image.Image, session=None) -> Image.Image:
    """Run rembg for photographs only; callers must gate this with detection."""
    from rembg import remove
    img = img.convert("RGBA")
    return remove(img, session=session).convert("RGBA")


def should_use_ai(detection: dict[str, object], mode_key: str) -> bool:
    """Allow AI only for photograph mode and photograph detections."""
    blocked_modes = {"transparent_png", "black_bg", "dark_artwork", "preserve_artwork", "dtf_ready"}
    if mode_key in blocked_modes:
        return False
    return bool(detection.get("use_ai")) and detection.get("type") == "photograph"


def apply_ai_alpha_to_original(original: Image.Image, ai_result: Image.Image) -> Image.Image:
    original = original.convert("RGBA")
    alpha = ai_result.convert("RGBA").getchannel("A")
    if alpha.size != original.size:
        alpha = alpha.resize(original.size, Image.Resampling.LANCZOS)

    result = original.copy()
    result.putalpha(alpha)
    return result
