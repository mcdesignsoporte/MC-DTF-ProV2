from __future__ import annotations

import json
from datetime import datetime, timezone

from PIL import Image, PngImagePlugin

from core.image_io import image_to_pdf_bytes, make_zip_bytes
from core.version import VERSION


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


def default_metadata(mode: str, dpi: int, processing_seconds: float = 0.0, resolution: str = "") -> dict[str, str]:
    """Build stable export metadata."""
    return {
        "Software": "MC DTF Pro V4",
        "version": VERSION,
        "modo": mode,
        "dpi": str(dpi),
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiempo": str(processing_seconds),
        "resolucion": resolution,
    }


def build_export_package(
    img: Image.Image,
    dpi: int = 300,
    prefix: str = "mc_dtf_pro_v4",
    mode: str = "dtf",
    extra_files: dict[str, bytes] | None = None,
    original: Image.Image | None = None,
    processing_seconds: float = 0.0,
) -> dict[str, bytes]:
    """Create PNG, PDF, and ZIP payloads without changing image dimensions."""
    metadata = default_metadata(mode, dpi, processing_seconds, f"{img.width} x {img.height}px")
    png = png_bytes(img, dpi=dpi, metadata=metadata)
    pdf = pdf_bytes(img, dpi=dpi)
    files = {
        "procesado.png": png,
        "procesado.pdf": pdf,
        "metadata.json": json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8"),
    }
    if original is not None:
        files["original.png"] = png_bytes(original, dpi=dpi, metadata={"Software": "MC DTF Pro V4", "tipo": "original"})
    else:
        files["original.png"] = png
    if extra_files:
        files.update(extra_files)
    return {"png": png, "pdf": pdf, "zip": make_zip_bytes(files)}
