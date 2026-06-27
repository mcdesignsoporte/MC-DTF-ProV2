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
    from rembg import remove
    img = img.convert("RGBA")
    return remove(img, session=session).convert("RGBA")
