from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from PIL import Image, ImageOps


def load_uploaded_image(uploaded_file) -> Image.Image:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    img.load()
    return img.convert("RGBA")


def image_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    bio = BytesIO()
    img.convert("RGBA").save(bio, format="PNG", dpi=(dpi, dpi), optimize=True)
    return bio.getvalue()


def image_to_pdf_bytes(img: Image.Image, dpi: int = 300, white_background: bool = True) -> bytes:
    bio = BytesIO()
    rgba = img.convert("RGBA")
    if white_background:
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
    else:
        bg = rgba.convert("RGB")
    bg.save(bio, format="PDF", resolution=dpi)
    return bio.getvalue()


def make_zip_bytes(files: dict[str, bytes]) -> bytes:
    bio = BytesIO()
    with ZipFile(bio, "w", compression=ZIP_DEFLATED) as zf:
        for filename, payload in files.items():
            if payload:
                zf.writestr(filename, payload)
    return bio.getvalue()
