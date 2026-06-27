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
from core.preview import alpha_difference_preview, before_after_preview, composite_preview
from core.constants import SUPPORTED_FORMATS
from core.version import AUTHOR, NAME, VERSION
from ui.batch import render_batch_table
from ui.downloads import render_downloads
from ui.preview import render_input_summary, render_result_preview
from ui.sidebar import render_sidebar

APP_VERSION = f"V{VERSION}"
MODE_KEYS = {mode["key"]: name for name, mode in MODES.items()}

st.set_page_config(page_title=f"{NAME} {APP_VERSION}", page_icon="MC", layout="wide")
st.markdown(f"<style>{Path('assets/styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(f'<div class="mc-tag">{AUTHOR}</div>', unsafe_allow_html=True)
st.markdown(f'<h1 class="mc-title">{NAME} {APP_VERSION}</h1>', unsafe_allow_html=True)
st.caption("Detector automatico, modos inteligentes, limpieza de fondo negro, proteccion de letras y exportacion PNG/PDF/ZIP.")


@st.cache_resource(show_spinner="Cargando modelo IA de fondo...")
def cached_session():
    return get_rembg_session()


def current_settings(mode: dict[str, object], options) -> PipelineSettings:
    """Convert UI options into reusable core pipeline settings."""
    return PipelineSettings(
        mode_key=str(mode["key"]),
        use_ai=options.use_ai,
        remove_black=options.remove_black,
        clean_enabled=options.clean_enabled,
        trim=options.trim,
        alpha_cut=options.alpha_cut,
        despeckle_area=options.despeckle_area,
        edge_contract=options.edge_contract,
        black_threshold=options.black_threshold,
        protect_details=options.protect_details,
        max_ai_side=options.max_ai_side,
        upscale=options.upscale,
        dpi=options.dpi,
        width_cm=options.width_cm,
        height_cm=options.height_cm,
    )


uploaded_files = st.file_uploader(
    "Upload one or more images",
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
    st.success(f"Detected: {detected['type']} | Recommendation: {recommended_mode_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated time", f"{detected['estimated_seconds']}s")
    c2.metric("Resolution", str(detected["resolution"]))
    c3.metric("Black", f"{detected['black_percent']}%")
    c4.metric("Edges", f"{detected['edge_density']}%")

render_input_summary(original, mode_name, mode)

def process_image(original_img, detection: dict[str, object], filename: str = "image") -> dict[str, object]:
    """Run one image through the shared production pipeline."""
    return process_artwork(original_img, detection, current_settings(mode, options), session_factory=cached_session, prefix=filename)


if st.button("Process image", type="primary", use_container_width=True):
    progress = st.progress(0)
    log = st.empty()
    t0 = time.time()

    try:
        log.write("1/3 Processing image...")
        progress.progress(10)
        result_payload = process_image(original, detected or {}, "mc_dtf_pro_v4")
        work = result_payload["image"]
        st.session_state["original_img"] = original.copy()
        st.session_state["result_img"] = work
        extra_files = {}

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

        log.write("2/3 Preparing downloads...")
        exports = build_export_package(work, dpi=options.dpi, mode=str(mode["key"]), extra_files=extra_files)
        st.session_state["result_png"] = exports["png"]
        st.session_state["result_pdf"] = exports["pdf"]
        st.session_state["result_zip"] = exports["zip"]
        progress.progress(100)
        log.success(f"Listo en {time.time() - t0:.1f} segundos")

    except Exception as exc:
        log.error(f"Error al procesar: {exc}")

if len(uploaded_files or []) > 1 and st.button("Process batch", use_container_width=True):
    batch_files = {}
    batch_rows = []
    progress = st.progress(0)
    for index, file in enumerate(uploaded_files or [], start=1):
        started = time.time()
        try:
            img = load_uploaded_image(file)
            file_detection = detect(img)
            stem = file.name.rsplit(".", 1)[0]
            payload = process_image(img, file_detection, stem)
            batch_files[f"{stem}.png"] = payload["png"]
            batch_files[f"{stem}.pdf"] = payload["pdf"]
            batch_rows.append({
                "file": file.name,
                "status": "ok",
                "detected": file_detection["type"],
                "mode": mode_name,
                "seconds": round(time.time() - started, 2),
                "resolution": file_detection["resolution"],
            })
        except Exception as exc:
            batch_rows.append({
                "file": file.name,
                "status": "error",
                "detected": "-",
                "mode": mode_name,
                "seconds": round(time.time() - started, 2),
                "resolution": str(exc),
            })
        progress.progress(index / len(uploaded_files))
    from core.image_io import make_zip_bytes

    st.session_state["batch_rows"] = batch_rows
    render_batch_table(batch_rows)
    if batch_files:
        st.download_button("Download batch ZIP", make_zip_bytes(batch_files), "mc_dtf_pro_v4_batch.zip", "application/zip", use_container_width=True)

if "batch_rows" in st.session_state:
    render_batch_table(st.session_state["batch_rows"])

if "result_img" in st.session_state:
    st.divider()
    st.subheader("Resultado")
    bg_mode = st.radio("Preview", ["Transparent", "Black shirt", "White shirt", "Sticker", "Mug", "Beer mug", "Hoodie"], horizontal=True)
    preview = composite_preview(st.session_state["result_img"], bg_mode)
    qa_mode = st.radio("QA view", ["Final preview", "Before / After", "Alpha changes"], horizontal=True)
    if qa_mode == "Before / After":
        preview = before_after_preview(st.session_state["original_img"], st.session_state["result_img"], bg_mode)
    elif qa_mode == "Alpha changes":
        preview = alpha_difference_preview(st.session_state["original_img"], st.session_state["result_img"])

    col_a, col_b = st.columns([2, 1])
    with col_a:
        render_result_preview(preview)
    with col_b:
        render_downloads(
            st.session_state["result_img"],
            st.session_state["result_png"],
            st.session_state["result_pdf"],
            st.session_state["result_zip"],
        )

    if "halftone_img" in st.session_state:
        st.subheader("Semitono")
        st.image(composite_preview(st.session_state["halftone_img"], bg_mode), use_container_width=True)
        st.download_button("Descargar PNG semitono", st.session_state["halftone_png"], "mc_dtf_pro_v4_semitono.png", "image/png")
        st.download_button("Descargar PDF semitono", st.session_state["halftone_pdf"], "mc_dtf_pro_v4_semitono.pdf", "application/pdf")
