from PIL import Image

from core.image_io import image_to_pdf_bytes, image_to_png_bytes, make_zip_bytes


def build_export_package(img: Image.Image, dpi: int = 300, prefix: str = "mc_dtf_pro_v4", extra_files: dict[str, bytes] | None = None) -> dict[str, bytes]:
    png = image_to_png_bytes(img, dpi=dpi)
    pdf = image_to_pdf_bytes(img, dpi=dpi, white_background=True)
    files = {
        f"{prefix}.png": png,
        f"{prefix}.pdf": pdf,
    }
    if extra_files:
        files.update(extra_files)
    return {
        "png": png,
        "pdf": pdf,
        "zip": make_zip_bytes(files),
    }
