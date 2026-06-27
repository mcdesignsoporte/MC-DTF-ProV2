from io import BytesIO
from PIL import Image, ImageOps


def load_uploaded_image(uploaded) -> Image.Image:
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img)
    img.load()
    return img.convert("RGBA")


def image_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    bio = BytesIO()
    img.convert("RGBA").save(bio, format="PNG", dpi=(dpi, dpi), optimize=True)
    return bio.getvalue()


def image_to_pdf_bytes(img: Image.Image, dpi: int = 300, white_background: bool = True) -> bytes:
    rgba = img.convert("RGBA")
    if white_background:
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
    else:
        bg = rgba.convert("RGB")
    bio = BytesIO()
    bg.save(bio, format="PDF", resolution=dpi)
    return bio.getvalue()
