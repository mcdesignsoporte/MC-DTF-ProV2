import time
from pathlib import Path

import streamlit as st

from core.background import (
    get_rembg_session,
)
from core.detector import detect
from core.export import build_export_package
from core.halftone import make_halftone
from core.image_io import image_to_pdf_bytes, image_to_png_bytes, load_uploaded_image
from core.modes import MODES
from core.pipeline import PipelineSettings, process_artwork
from core.presets import preset_for_mode
from core.preview import composite_preview
from core.constants import SUPPORTED_FORMATS
from core.version import AUTHOR, NAME, VERSION
from ui.batch import render_batch_table
from ui.detection import detection_value
from ui.downloads import render_downloads
from ui.preview import render_input_summary, render_result_workspace
from ui.professional_center import render_professional_center
from ui.sidebar import render_sidebar

APP_VERSION = f"V{VERSION}"
MODE_KEYS = {mode["key"]: name for name, mode in MODES.items()}
MODE_KEYS["color_bg"] = "Fondo de color"

st.set_page_config(page_title=f"{NAME} {APP_VERSION}", page_icon="MC", layout="wide")
st.markdown(f"<style>{Path('assets/styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(f'<div class="mc-tag">{AUTHOR}</div>', unsafe_allow_html=True)
st.markdown(f'<h1 class="mc-title">{NAME} {APP_VERSION}</h1>', unsafe_allow_html=True)
st.caption("Software profesional DTF con deteccion automatica, modos inteligentes, previews comerciales y exportacion PNG/PDF/ZIP.")
render_professional_center()


@st.cache_resource(show_spinner="Cargando modelo IA de fondo...")
def cached_session():
    return get_rembg_session()


def current_settings(mode: dict[str, object], options) -> PipelineSettings:
    """Convert UI options into reusable core pipeline settings."""
    return PipelineSettings(
        mode_key=str(mode["key"]),
        use_ai=options.use_ai,
        remove_black=options.remove_black,
        remove_color=options.remove_color,
        clean_enabled=options.clean_enabled,
        trim=options.trim,
        alpha_cut=options.alpha_cut,
        despeckle_area=options.despeckle_area,
        edge_contract=options.edge_contract,
        black_threshold=options.black_threshold,
        black_level=options.black_level,
        color_tolerance=options.color_tolerance,
        protect_details=options.protect_details,
        protect_white_details=options.protect_white_details,
        white_protection_level=options.white_protection_level,
        fine_detail_level=options.fine_detail_level,
        safe_mode=options.safe_mode,
        enable_dtf_prepress=options.enable_dtf_prepress,
        remove_white_halo=options.remove_white_halo,
        remove_black_halo=options.remove_black_halo,
        halo_strength=options.halo_strength,
        expand_edge_px=options.expand_edge_px,
        bleed_px=options.bleed_px,
        create_cutline=options.create_cutline,
        min_printable_mm=options.min_printable_mm,
        logo_detect_colors=options.logo_detect_colors,
        logo_reduce_colors=options.logo_reduce_colors,
        logo_black_to_transparent=options.logo_black_to_transparent,
        logo_white_to_transparent=options.logo_white_to_transparent,
        logo_unify_colors=options.logo_unify_colors,
        logo_separate_colors=options.logo_separate_colors,
        logo_export_layers=options.logo_export_layers,
        logo_max_colors=options.logo_max_colors,
        logo_color_tolerance=options.logo_color_tolerance,
        max_ai_side=options.max_ai_side,
        upscale=options.upscale,
        dpi=options.dpi,
        width_cm=options.width_cm,
        height_cm=options.height_cm,
        complex_white_preset=options.complex_white_preset,
        complex_white_tolerance=options.complex_white_tolerance,
        complex_white_luminosity=options.complex_white_luminosity,
        complex_white_saturation=options.complex_white_saturation,
        complex_white_preserve_internal=options.complex_white_preserve_internal,
        complex_white_halo_cleanup=options.complex_white_halo_cleanup,
        complex_white_mask_offset=options.complex_white_mask_offset,
        complex_white_alpha_smoothing=options.complex_white_alpha_smoothing,
        complex_white_export_debug=options.complex_white_export_debug,
        residue_refine_enabled=options.residue_refine_enabled,
        residue_luminosity=options.residue_luminosity,
        residue_saturation=options.residue_saturation,
        residue_min_area=options.residue_min_area,
        residue_max_area=options.residue_max_area,
        residue_remove_connected=options.residue_remove_connected,
        residue_remove_small=options.residue_remove_small,
        residue_preserve_internal=options.residue_preserve_internal,
        residue_manual_ids=options.residue_manual_ids,
    )


uploaded_files = st.file_uploader(
    "Subir imagenes",
    type=list(SUPPORTED_FORMATS),
    accept_multiple_files=True,
)
uploaded = uploaded_files[0] if uploaded_files else None
original = None
detected = None
recommended_mode_name = None

if uploaded:
    try:
        original = load_uploaded_image(uploaded)
        detected = detect(original)
        recommended_mode_name = MODE_KEYS.get(str(detected["recommended_mode"]))
    except Exception as exc:
        st.error(f"No se pudo abrir la imagen: {exc}")
        st.stop()

if "selected_mode" not in st.session_state:
    st.session_state["selected_mode"] = recommended_mode_name or list(MODES.keys())[3]

if uploaded and recommended_mode_name and st.session_state.get("last_upload_name") != uploaded.name:
    st.session_state["selected_mode"] = recommended_mode_name
    st.session_state["last_upload_name"] = uploaded.name

options = render_sidebar(st.session_state["selected_mode"])
mode_name = options.mode_name
mode = options.mode

if original is None:
    st.info("Sube una imagen para comenzar.")
    st.stop()

if detected and recommended_mode_name:
    st.success(f"Imagen analizada | Tipo detectado: {detection_value(detected, 'type')} | Recomendacion: {recommended_mode_name}")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Resolucion", str(detection_value(detected, "resolution")))
    c2.metric("Confianza", f"{detection_value(detected, 'confidence', 0)}%")
    c3.metric("Transparencia", f"{detection_value(detected, 'transparency_percent', 0)}%")
    c4.metric("Fondo", str(detection_value(detected, "background")))
    c5.metric("Color", str(detection_value(detected, "dominant_color")))
    c6.metric("Uniformidad", f"{detection_value(detected, 'background_uniformity', 0)}%")
    c7.metric("Tiempo", f"{detection_value(detected, 'estimated_seconds', 0)}s")
    b1, b2 = st.columns(2)
    if b1.button("Usar configuracion recomendada", use_container_width=True):
        st.session_state["selected_mode"] = recommended_mode_name
        st.rerun()
    if b2.button("Configurar manualmente", use_container_width=True):
        st.session_state["manual_config"] = True
    with st.expander("Configuracion manual", expanded=bool(st.session_state.get("manual_config", False))):
        st.caption("Selecciona otro modo en la barra superior y ajusta la configuracion avanzada si lo necesitas.")

render_input_summary(original, mode_name, mode)

def process_image(original_img, detection: dict[str, object], filename: str = "image", settings: PipelineSettings | None = None) -> dict[str, object]:
    """Run one image through the shared production pipeline."""
    return process_artwork(original_img, detection, settings or current_settings(mode, options), session_factory=cached_session, prefix=filename)


if st.button("Procesar imagen", type="primary", use_container_width=True):
    progress = st.progress(0)
    log = st.empty()
    t0 = time.time()

    try:
        log.write("1/3 Procesando imagen...")
        progress.progress(10)
        result_payload = process_image(original, detected or {}, "mc_dtf_pro_v4")
        work = result_payload["image"]
        st.session_state["original_img"] = original.copy()
        st.session_state["result_img"] = work
        st.session_state["white_protection"] = result_payload.get("white_protection")
        st.session_state["white_mask"] = result_payload.get("white_mask")
        st.session_state["fine_detail_protection"] = result_payload.get("fine_detail_protection")
        st.session_state["fine_detail_mask"] = result_payload.get("fine_detail_mask")
        st.session_state["artwork_mask"] = result_payload.get("artwork_mask")
        st.session_state["background_mask"] = result_payload.get("background_mask")
        st.session_state["doubtful_mask"] = result_payload.get("doubtful_mask")
        st.session_state["restored_mask"] = result_payload.get("restored_mask")
        st.session_state["art_loss_risk"] = result_payload.get("art_loss_risk")
        st.session_state["non_destructive_stats"] = result_payload.get("non_destructive_stats")
        st.session_state["dtf_prepress"] = result_payload.get("dtf_prepress")
        st.session_state["alpha_quality"] = result_payload.get("alpha_quality")
        st.session_state["white_halo_mask"] = result_payload.get("white_halo_mask")
        st.session_state["black_halo_mask"] = result_payload.get("black_halo_mask")
        st.session_state["bleed_mask"] = result_payload.get("bleed_mask")
        st.session_state["cutline_mask"] = result_payload.get("cutline_mask")
        st.session_state["small_elements_mask"] = result_payload.get("small_elements_mask")
        st.session_state["small_elements_report"] = result_payload.get("small_elements_report")
        st.session_state["logo_report"] = result_payload.get("logo_report")
        st.session_state["logo_palette"] = result_payload.get("logo_palette")
        st.session_state["logo_layers"] = result_payload.get("logo_layers")
        st.session_state["complex_white_debug"] = result_payload.get("complex_white_debug")
        st.session_state["autopilot"] = result_payload.get("autopilot")
        st.session_state["autopilot_quality"] = result_payload.get("autopilot_quality")
        extra_files = dict(result_payload.get("dtf_extra_files") or {})
        metadata_extra = dict(result_payload.get("metadata_extra") or {})

        if options.make_halftone:
            ht = make_halftone(work, dot_size=options.dot_size, angle=options.angle, invert=options.invert_halftone)
            st.session_state["halftone_img"] = ht
            st.session_state["halftone_png"] = image_to_png_bytes(ht, dpi=options.dpi)
            st.session_state["halftone_pdf"] = image_to_pdf_bytes(ht, dpi=options.dpi, white_background=True)
            extra_files["mc_dtf_pro_v4_semitono.png"] = st.session_state["halftone_png"]
            extra_files["mc_dtf_pro_v4_semitono.pdf"] = st.session_state["halftone_pdf"]
        else:
            for key in ["halftone_img", "halftone_png", "halftone_pdf"]:
                st.session_state.pop(key, None)

        log.write("2/3 Preparando descargas...")
        exports = build_export_package(work, dpi=options.dpi, mode=str(mode["key"]), extra_files=extra_files, original=original, processing_seconds=round(time.time() - t0, 3), metadata_extra=metadata_extra)
        st.session_state["result_png"] = exports["png"]
        st.session_state["result_pdf"] = exports["pdf"]
        st.session_state["result_zip"] = exports["zip"]
        progress.progress(100)
        log.success(f"Listo en {time.time() - t0:.1f} segundos")

    except Exception as exc:
        log.error(f"Error al procesar: {exc}")

if len(uploaded_files or []) > 1 and st.button("Procesar lote", use_container_width=True):
    batch_files = {}
    batch_rows = []
    progress = st.progress(0)
    for index, file in enumerate(uploaded_files or [], start=1):
        started = time.time()
        try:
            img = load_uploaded_image(file)
            file_detection = detect(img)
            stem = file.name.rsplit(".", 1)[0]
            recommended_preset = preset_for_mode(str(file_detection.get("recommended_mode", "dtf_ready")))
            payload = process_image(img, file_detection, stem, current_settings(recommended_preset, options))
            batch_files[f"{stem}.png"] = payload["png"]
            batch_files[f"{stem}.pdf"] = payload["pdf"]
            batch_rows.append({
                "Archivo": file.name,
                "Estado": "ok",
                "Tipo": file_detection["type"],
                "Modo": recommended_preset["label"],
                "Tiempo": round(time.time() - started, 2),
            })
        except Exception as exc:
            batch_rows.append({
                "Archivo": file.name,
                "Estado": "error",
                "Tipo": "-",
                "Modo": "-",
                "Tiempo": round(time.time() - started, 2),
                "Detalle": str(exc),
            })
        progress.progress(index / len(uploaded_files))
    from core.image_io import make_zip_bytes

    st.session_state["batch_rows"] = batch_rows
    render_batch_table(batch_rows)
    if batch_files:
        st.download_button("Descargar ZIP del lote", make_zip_bytes(batch_files), "mc_dtf_pro_v4_lote.zip", "application/zip", use_container_width=True)

if "batch_rows" in st.session_state:
    render_batch_table(st.session_state["batch_rows"])

if "result_img" in st.session_state:
    st.divider()
    st.subheader("Resultado")
    render_result_workspace(
        st.session_state["original_img"],
        st.session_state["result_img"],
        options.dpi,
        st.session_state["result_png"],
        st.session_state.get("white_mask"),
        st.session_state.get("white_protection"),
        st.session_state.get("fine_detail_mask"),
        st.session_state.get("fine_detail_protection"),
        st.session_state.get("artwork_mask"),
        st.session_state.get("background_mask"),
        st.session_state.get("doubtful_mask"),
        st.session_state.get("restored_mask"),
        st.session_state.get("art_loss_risk"),
        st.session_state.get("non_destructive_stats"),
        st.session_state.get("dtf_prepress"),
        st.session_state.get("alpha_quality"),
        st.session_state.get("white_halo_mask"),
        st.session_state.get("black_halo_mask"),
        st.session_state.get("bleed_mask"),
        st.session_state.get("cutline_mask"),
        st.session_state.get("small_elements_mask"),
        st.session_state.get("small_elements_report"),
        st.session_state.get("logo_report"),
        st.session_state.get("logo_palette"),
        st.session_state.get("logo_layers"),
        st.session_state.get("complex_white_debug"),
        st.session_state.get("autopilot"),
        st.session_state.get("autopilot_quality"),
    )
    with st.expander("Descargas", expanded=True):
        render_downloads(
            st.session_state["result_img"],
            st.session_state["result_png"],
            st.session_state["result_pdf"],
            st.session_state["result_zip"],
        )

    if "halftone_img" in st.session_state:
        st.subheader("Semitono")
        st.image(composite_preview(st.session_state["halftone_img"], "Transparente"), use_container_width=True)
        st.download_button("Descargar PNG semitono", st.session_state["halftone_png"], "mc_dtf_pro_v4_semitono.png", "image/png")
        st.download_button("Descargar PDF semitono", st.session_state["halftone_pdf"], "mc_dtf_pro_v4_semitono.pdf", "application/pdf")
