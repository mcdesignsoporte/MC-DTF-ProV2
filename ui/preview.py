import streamlit as st
from PIL import Image

from core.preview import VIEWPORT, alpha_difference_preview, alpha_preview, before_after_preview, composite_preview, preview_thumbnail
from core.quality import evaluate_dtf_quality
from core.white_protection import mask_preview
from ui.quality import render_quality

BACKGROUNDS = ["Transparente", "Playera negra", "Playera blanca", "Sudadera", "Sticker", "Tarro", "Taza", "Calcomania"]


def render_input_summary(original: Image.Image, mode_name: str, mode: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(original, use_container_width=True)
        st.caption(f"{original.width} x {original.height}px")
    with right:
        st.subheader("Modo seleccionado")
        st.success(mode_name)
        if mode["key"] in ["black_bg", "dark_artwork"]:
            st.warning("Elimina negro puro y conserva letras, grises, cromados y efectos blancos.")
        elif mode["key"] == "preserve_artwork":
            st.info("Conserva el diseno completo sin eliminar fondo.")


def render_result_workspace(
    original: Image.Image,
    result: Image.Image,
    dpi: int,
    png_bytes: bytes,
    white_mask: object | None = None,
    white_stats: dict[str, object] | None = None,
    fine_mask: object | None = None,
    fine_stats: dict[str, object] | None = None,
) -> None:
    """Render a commercial viewport for large DTF artwork."""
    report = evaluate_dtf_quality(result, dpi=dpi)
    background = st.selectbox("Fondo de vista", BACKGROUNDS, index=0)
    tabs = st.tabs(["Resultado final", "Antes / Despues", "Cambios de transparencia", "Alpha", "Blancos protegidos", "Detalles protegidos", "Original"])

    with tabs[0]:
        st.image(composite_preview(result, background), use_container_width=True)
    with tabs[1]:
        st.image(before_after_preview(original, result, background), use_container_width=True)
    with tabs[2]:
        st.image(alpha_difference_preview(original, result), use_container_width=True)
    with tabs[3]:
        st.image(alpha_preview(result), use_container_width=True)
    with tabs[4]:
        st.image(_white_mask_preview(original, white_mask), use_container_width=True)
    with tabs[5]:
        st.image(_green_mask_preview(original, fine_mask), use_container_width=True)
    with tabs[6]:
        st.image(composite_preview(original, background), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        render_technical_info(result, dpi, png_bytes, report.status)
        render_white_stats(white_stats)
        render_fine_detail_stats(fine_stats)
    with right:
        render_quality(report)
        if white_stats and bool(white_stats.get("possible_detail_loss", False)):
            st.warning("Posible perdida de detalles blancos.")


def render_technical_info(img: Image.Image, dpi: int, png_bytes: bytes, status: str) -> None:
    """Render practical production data for print operators."""
    rgba = img.convert("RGBA")
    width_cm = rgba.width / dpi * 2.54
    height_cm = rgba.height / dpi * 2.54
    has_alpha = rgba.getextrema()[3][0] < 255
    st.subheader("Informacion tecnica")
    st.metric("Tamano", f"{rgba.width} x {rgba.height} px")
    st.metric("Impresion", f"{width_cm:.1f} x {height_cm:.1f} cm")
    st.metric("DPI", str(dpi))
    st.metric("Formato", "PNG RGBA")
    st.metric("Transparencia", "Si" if has_alpha else "No")
    st.metric("Peso PNG", _format_size(len(png_bytes)))
    st.success(f"Estado DTF: {status}")


def _format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.2f} MB"
    return f"{size / 1024:.1f} KB"


def _white_mask_preview(original: Image.Image, white_mask: object | None) -> Image.Image:
    return _green_mask_preview(original, white_mask)


def _green_mask_preview(original: Image.Image, mask: object | None) -> Image.Image:
    if mask is None:
        return composite_preview(Image.new("RGBA", original.size, (0, 0, 0, 0)), "Transparente")
    base = composite_preview(original, "Playera negra")
    overlay = mask_preview(mask, original.size)
    thumb = preview_thumbnail(overlay, VIEWPORT, padding=40)
    x = (base.width - thumb.width) // 2
    y = (base.height - thumb.height) // 2
    base.alpha_composite(thumb, (x, y))
    return base


def render_white_stats(stats: dict[str, object] | None) -> None:
    """Render white detail protection statistics."""
    if not stats:
        return
    st.subheader("Blancos protegidos")
    st.metric("Blancos detectados", str(stats.get("white_detected", 0)))
    st.metric("Blancos protegidos", str(stats.get("white_protected", 0)))
    st.metric("Blancos eliminados", str(stats.get("white_removed", 0)))
    st.metric("Porcentaje protegido", f"{stats.get('protected_percent', 0)}%")


def render_fine_detail_stats(stats: dict[str, object] | None) -> None:
    """Render fine detail protection statistics."""
    if not stats:
        return
    st.subheader("Detalles protegidos")
    st.metric("Detalles finos detectados", str(stats.get("fine_details_detected", 0)))
    st.metric("Detalles protegidos", str(stats.get("details_protected", 0)))
    st.metric("Ruido eliminado", str(stats.get("noise_removed", 0)))
    st.metric("Componentes eliminados", str(stats.get("components_removed", 0)))
