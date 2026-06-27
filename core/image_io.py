from io import BytesIO
from PIL import Image, ImageOps

MAX_PIXELS = 9000 * 9000


def load_uploaded_image(uploaded_file) -> Image.Image:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    img.load()
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("La imagen es demasiado grande. Usa una imagen menor a 9000 x 9000 px.")
    return img.convert("RGBA")


def image_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    bio = BytesIO()
    img.save(bio, format="PNG", dpi=(dpi, dpi), optimize=True)
    return bio.getvalue()


def image_to_pdf_bytes(img: Image.Image, dpi: int = 300, white_background: bool = True) -> bytes:
    bio = BytesIO()
    if white_background:
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.getchannel("A"))
    else:
        bg = img.convert("RGB")
    bg.save(bio, format="PDF", resolution=dpi)
    return bio.getvalue()
