import streamlit as st
from PIL import Image

from core.preview import VIEWPORT, alpha_preview, before_after_preview, composite_preview, preview_thumbnail
from core.quality import evaluate_dtf_quality
from core.quality_report import quality_report
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
    artwork_mask: object | None = None,
    background_mask: object | None = None,
    doubtful_mask: object | None = None,
    restored_mask: object | None = None,
    risk: dict[str, object] | None = None,
    non_destructive_stats: dict[str, object] | None = None,
    dtf_prepress: dict[str, object] | None = None,
    alpha_quality: dict[str, object] | None = None,
    white_halo_mask: object | None = None,
    black_halo_mask: object | None = None,
    bleed_mask: object | None = None,
    cutline_mask: object | None = None,
    small_elements_mask: object | None = None,
    small_elements_report: dict[str, object] | None = None,
    logo_report: dict[str, object] | None = None,
    logo_palette: list[dict[str, object]] | None = None,
    logo_layers: list[dict[str, object]] | None = None,
) -> None:
    """Render a commercial viewport for large DTF artwork."""
    report = evaluate_dtf_quality(result, dpi=dpi)
    production_report = quality_report(original, result, artwork_mask, background_mask, risk, alpha_quality, dtf_prepress, small_elements_report)
    background = st.selectbox("Fondo de vista", BACKGROUNDS, index=0)
    tabs = st.tabs([
        "Resultado final",
        "Antes / Despues",
        "Alpha limpio",
        "Halo detectado",
        "Sangrado",
        "Borde de corte",
        "Elementos pequenos",
        "Paleta detectada",
        "Capas por color",
        "Arte protegido",
        "Riesgo de perdida",
        "Original",
    ])

    with tabs[0]:
        st.image(composite_preview(result, background), use_container_width=True)
    with tabs[1]:
        st.image(before_after_preview(original, result, background), use_container_width=True)
    with tabs[2]:
        st.image(alpha_preview(result), use_container_width=True)
    with tabs[3]:
        st.image(_combined_halo_preview(original, white_halo_mask, black_halo_mask), use_container_width=True)
    with tabs[4]:
        st.image(_blue_mask_preview(original, bleed_mask), use_container_width=True)
    with tabs[5]:
        st.image(_green_mask_preview(original, cutline_mask), use_container_width=True)
    with tabs[6]:
        st.image(_yellow_mask_preview(original, small_elements_mask), use_container_width=True)
    with tabs[7]:
        render_palette_preview(logo_palette)
    with tabs[8]:
        render_logo_layers(logo_layers)
    with tabs[9]:
        st.image(_green_mask_preview(original, artwork_mask if artwork_mask is not None else white_mask), use_container_width=True)
    with tabs[10]:
        st.image(_risk_preview(original, doubtful_mask, restored_mask), use_container_width=True)
    with tabs[11]:
        st.image(composite_preview(original, background), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        render_technical_info(result, dpi, png_bytes, report.status)
        render_white_stats(white_stats)
        render_fine_detail_stats(fine_stats)
        render_non_destructive_stats(non_destructive_stats)
        render_dtf_prepress_stats(alpha_quality, dtf_prepress, small_elements_report)
        render_logo_stats(logo_report, logo_palette)
        render_quality_report(production_report)
    with right:
        render_quality(report)
        st.success("Modo seguro activo")
        if white_stats and bool(white_stats.get("possible_detail_loss", False)):
            st.warning("Posible perdida de detalles blancos.")
        if risk and bool(risk.get("risk_detected", False)):
            st.warning("Riesgo de perdida de arte detectado. Usa modo Conservador Profesional.")
        elif risk:
            st.success("Sin perdida de detalles")


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
    return _colored_mask_preview(original, mask, (0, 220, 120, 210))


def _red_mask_preview(original: Image.Image, mask: object | None) -> Image.Image:
    return _colored_mask_preview(original, mask, (230, 40, 40, 210))


def _blue_mask_preview(original: Image.Image, mask: object | None) -> Image.Image:
    return _colored_mask_preview(original, mask, (40, 130, 255, 210))


def _yellow_mask_preview(original: Image.Image, mask: object | None) -> Image.Image:
    return _colored_mask_preview(original, mask, (245, 210, 35, 210))


def _combined_halo_preview(original: Image.Image, white_halo: object | None, black_halo: object | None) -> Image.Image:
    base = composite_preview(original, "Playera negra")
    for mask, color in [(white_halo, (230, 40, 40, 210)), (black_halo, (255, 120, 0, 210))]:
        if mask is None:
            continue
        overlay = mask_preview(mask, original.size)
        overlay = _recolor_overlay(overlay, color)
        thumb = preview_thumbnail(overlay, VIEWPORT, padding=40)
        base.alpha_composite(thumb, ((base.width - thumb.width) // 2, (base.height - thumb.height) // 2))
    return base


def _risk_preview(original: Image.Image, doubtful_mask: object | None, restored_mask: object | None) -> Image.Image:
    base = composite_preview(original, "Playera negra")
    for mask, color in [(doubtful_mask, (245, 210, 35, 170)), (restored_mask, (40, 130, 255, 220))]:
        if mask is None:
            continue
        overlay = mask_preview(mask, original.size)
        overlay = _recolor_overlay(overlay, color)
        thumb = preview_thumbnail(overlay, VIEWPORT, padding=40)
        base.alpha_composite(thumb, ((base.width - thumb.width) // 2, (base.height - thumb.height) // 2))
    return base


def _colored_mask_preview(original: Image.Image, mask: object | None, color: tuple[int, int, int, int]) -> Image.Image:
    if mask is None:
        return composite_preview(Image.new("RGBA", original.size, (0, 0, 0, 0)), "Transparente")
    base = composite_preview(original, "Playera negra")
    overlay = mask_preview(mask, original.size)
    overlay = _recolor_overlay(overlay, color)
    thumb = preview_thumbnail(overlay, VIEWPORT, padding=40)
    x = (base.width - thumb.width) // 2
    y = (base.height - thumb.height) // 2
    base.alpha_composite(thumb, (x, y))
    return base


def _recolor_overlay(img: Image.Image, color: tuple[int, int, int, int]) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    out = Image.new("RGBA", rgba.size, color)
    out.putalpha(alpha.point(lambda value: min(value, color[3])))
    return out


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


def render_non_destructive_stats(stats: dict[str, object] | None) -> None:
    """Render conservative engine metrics."""
    if not stats:
        return
    st.subheader("Motor no destructivo")
    st.metric("Fondo confirmado eliminado", str(stats.get("background_removed", 0)))
    st.metric("Arte protegido", str(stats.get("artwork_protected", 0)))
    st.metric("Pixeles restaurados", str(stats.get("restored_pixels", 0)))


def render_dtf_prepress_stats(alpha_quality: dict[str, object] | None, metadata: dict[str, object] | None, small_report: dict[str, object] | None) -> None:
    """Render DTF prepress QA metrics."""
    if not alpha_quality and not metadata and not small_report:
        return
    st.subheader("Preparacion DTF")
    if alpha_quality:
        st.metric("Semitransparencia", f"{alpha_quality.get('semi_transparent_percent', 0)}%")
        st.metric("Ruido alfa", str(alpha_quality.get("alpha_noise_score", 0)))
    if metadata:
        st.metric("Halo blanco", "Riesgo" if metadata.get("halo_white_risk") else "OK")
        st.metric("Halo negro", "Riesgo" if metadata.get("halo_black_risk") else "OK")
        st.metric("Borde de corte", "Listo" if metadata.get("cutline_ready") else "No activo")
    if small_report:
        count = int(small_report.get("count", 0))
        st.metric("Elementos pequenos", str(count))
        if count:
            st.warning("Hay detalles menores al tamano minimo imprimible.")


def render_quality_report(report: dict[str, object]) -> None:
    """Render professional quality report metrics."""
    st.subheader("Reporte de calidad")
    st.metric("Score", f"{report.get('score', 0)}%")
    st.metric("Fondo eliminado", f"{report.get('removed_background_percent', 0)}%")
    st.metric("Arte protegido", f"{report.get('protected_art_percent', 0)}%")
    st.metric("Riesgo", str(report.get("risk_level", "bajo")))
    st.metric("Score DTF", f"{report.get('dtf_ready_score', report.get('score', 0))}%")
    for warning in report.get("warnings", []):
        st.warning(str(warning))


def render_palette_preview(palette: list[dict[str, object]] | None) -> None:
    """Render detected logo palette."""
    st.subheader("Paleta detectada")
    if not palette:
        st.info("Sin paleta detectada.")
        return
    dominant = palette[0]
    st.metric("Color dominante", str(dominant.get("hex", "-")))
    for color in palette:
        st.color_picker(str(color.get("hex")), str(color.get("hex")), disabled=True)
        st.caption(f"{color.get('percent', 0)}%")


def render_logo_layers(layers: list[dict[str, object]] | None) -> None:
    """Render separated logo color layers."""
    st.subheader("Capas por color")
    if not layers:
        st.info("Sin capas separadas.")
        return
    for index, layer in enumerate(layers[:8], start=1):
        st.caption(f"Capa {index} | {layer.get('hex')}")
        st.image(composite_preview(layer["image"], "Transparente"), use_container_width=True)


def render_logo_stats(report: dict[str, object] | None, palette: list[dict[str, object]] | None) -> None:
    """Render logo tool metrics."""
    if not report and not palette:
        return
    st.subheader("Herramientas para logos")
    st.metric("Colores detectados", str((report or {}).get("colors_detected", len(palette or []))))
    st.metric("Capas por color", str((report or {}).get("layers", 0)))
    if report and report.get("photo_warning"):
        st.warning("La separacion por colores no es recomendada para fotografias.")
