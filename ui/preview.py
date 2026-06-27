import streamlit as st
from PIL import Image

from core.preview import alpha_difference_preview, alpha_preview, before_after_preview, composite_preview
from core.quality import evaluate_dtf_quality
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


def render_result_workspace(original: Image.Image, result: Image.Image, dpi: int, png_bytes: bytes) -> None:
    """Render a commercial viewport for large DTF artwork."""
    report = evaluate_dtf_quality(result, dpi=dpi)
    background = st.selectbox("Fondo de vista", BACKGROUNDS, index=0)
    tabs = st.tabs(["Resultado final", "Antes / Despues", "Cambios de transparencia", "Alpha", "Original"])

    with tabs[0]:
        st.image(composite_preview(result, background), use_container_width=True)
    with tabs[1]:
        st.image(before_after_preview(original, result, background), use_container_width=True)
    with tabs[2]:
        st.image(alpha_difference_preview(original, result), use_container_width=True)
    with tabs[3]:
        st.image(alpha_preview(result), use_container_width=True)
    with tabs[4]:
        st.image(composite_preview(original, background), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        render_technical_info(result, dpi, png_bytes, report.status)
    with right:
        render_quality(report)


def render_technical_info(img: Image.Image, dpi: int, png_bytes: bytes, status: str) -> None:
    """Render practical production data for print operators."""
    rgba = img.convert("RGBA")
    width_cm = rgba.width / dpi * 2.54
    height_cm = rgba.height / dpi * 2.54
    has_alpha = rgba.getextrema()[3][0] < 255
    st.subheader("Informacion tecnica")
    c1, c2 = st.columns(2)
    c1.metric("Tamano", f"{rgba.width} x {rgba.height} px")
    c2.metric("Impresion", f"{width_cm:.1f} x {height_cm:.1f} cm")
    c3, c4 = st.columns(2)
    c3.metric("DPI", str(dpi))
    c4.metric("Formato", "PNG RGBA")
    c5, c6 = st.columns(2)
    c5.metric("Transparencia", "Si" if has_alpha else "No")
    c6.metric("Peso PNG", _format_size(len(png_bytes)))
    st.success(f"Estado DTF: {status}")


def _format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.2f} MB"
    return f"{size / 1024:.1f} KB"
