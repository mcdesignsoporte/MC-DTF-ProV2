from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageOps

from core.constants import SUPPORTED_FORMATS


def load_uploaded_image(uploaded_file) -> Image.Image:
    """Open, transpose, validate, and normalize an uploaded image."""
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower() if hasattr(uploaded_file, "name") else ""
    if suffix and suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Formato no soportado: {suffix}")
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    img.load()
    return img.convert("RGBA")


def image_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    """Serialize a transparent PNG."""
    bio = BytesIO()
    img.convert("RGBA").save(bio, format="PNG", dpi=(dpi, dpi), optimize=True)
    return bio.getvalue()


def image_to_pdf_bytes(img: Image.Image, dpi: int = 300, white_background: bool = True) -> bytes:
    """Serialize a PDF for print review."""
    bio = BytesIO()
    rgba = img.convert("RGBA")
    page = Image.new("RGB", rgba.size, (255, 255, 255)) if white_background else rgba.convert("RGB")
    if white_background:
        page.paste(rgba, mask=rgba.getchannel("A"))
    page.save(bio, format="PDF", resolution=dpi)
    return bio.getvalue()


def make_zip_bytes(files: dict[str, bytes]) -> bytes:
    """Create an in-memory ZIP from named byte payloads."""
    bio = BytesIO()
    with ZipFile(bio, "w", compression=ZIP_DEFLATED) as zf:
        for filename, payload in files.items():
            if payload:
                zf.writestr(filename, payload)
    return bio.getvalue()
