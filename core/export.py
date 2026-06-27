from __future__ import annotations

from datetime import datetime, timezone

from PIL import Image, PngImagePlugin

from core.image_io import image_to_pdf_bytes, make_zip_bytes


def png_bytes(img: Image.Image, dpi: int = 300, metadata: dict[str, str] | None = None) -> bytes:
    """Export a transparent PNG at the requested DPI with metadata."""
    from io import BytesIO

    info = PngImagePlugin.PngInfo()
    for key, value in (metadata or {}).items():
        info.add_text(key, value)
    bio = BytesIO()
    img.convert("RGBA").save(bio, format="PNG", dpi=(dpi, dpi), pnginfo=info, optimize=True)
    return bio.getvalue()


def pdf_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    """Export a print-friendly PDF with a white page background."""
    return image_to_pdf_bytes(img, dpi=dpi, white_background=True)


def default_metadata(mode: str, dpi: int) -> dict[str, str]:
    """Build stable export metadata."""
    return {
        "Software": "MC DTF Pro V4",
        "Mode": mode,
        "DPI": str(dpi),
        "CreatedUTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_export_package(
    img: Image.Image,
    dpi: int = 300,
    prefix: str = "mc_dtf_pro_v4",
    mode: str = "dtf",
    extra_files: dict[str, bytes] | None = None,
) -> dict[str, bytes]:
    """Create PNG, PDF, and ZIP payloads without changing image dimensions."""
    metadata = default_metadata(mode, dpi)
    png = png_bytes(img, dpi=dpi, metadata=metadata)
    pdf = pdf_bytes(img, dpi=dpi)
    files = {f"{prefix}.png": png, f"{prefix}.pdf": pdf}
    if extra_files:
        files.update(extra_files)
    return {"png": png, "pdf": pdf, "zip": make_zip_bytes(files)}
