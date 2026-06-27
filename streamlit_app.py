import time

import streamlit as st

from core.background import (
    apply_ai_alpha_to_original,
    get_rembg_session,
    has_transparency,
    remove_background_ai,
    resize_for_ai,
)
from core.black_remove import remove_black_background
from core.clean import clean_alpha, trim_transparent
from core.detector import detect
from core.export import build_export_package
from core.halftone import make_halftone
from core.image_io import image_to_pdf_bytes, image_to_png_bytes, load_uploaded_image
from core.modes import MODES
from core.preview import composite_preview
from core.resize import fit_to_print_size, upscale_and_sharpen
from ui.downloads import render_downloads
from ui.preview import render_input_summary, render_result_preview
from ui.sidebar import render_sidebar

APP_VERSION = "V4.0.0"
MODE_KEYS = {mode["key"]: name for name, mode in MODES.items()}
DETECTOR_TO_MODE_KEY = {
    "photo": "photo_bg",
    "png": "transparent_png",
    "dark": "black_bg",
    "white": "dtf_ready",
}

st.set_page_config(page_title=f"MC DTF Pro {APP_VERSION}", page_icon="MC", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1450px;}
.mc-title {font-size: 3rem; font-weight: 900; margin-bottom: 0;}
.mc-tag {color:#d9aa32; text-transform: uppercase; letter-spacing:.18rem; font-weight:800;}
.mc-help {padding: .85rem 1rem; border: 1px solid #3b3b3b; border-radius: 8px; background:#151515;}
.stButton button {font-weight: 800;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="mc-tag">MC Creative Studio</div>', unsafe_allow_html=True)
st.markdown(f'<h1 class="mc-title">MC DTF Pro {APP_VERSION}</h1>', unsafe_allow_html=True)
st.caption("Detector automatico, modos inteligentes, limpieza de fondo negro, proteccion de letras y exportacion PNG/PDF/ZIP.")


@st.cache_resource(show_spinner="Cargando modelo IA de fondo...")
def cached_session():
    return get_rembg_session()


uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
original = None
detected = None
recommended_mode_name = None

if uploaded:
    try:
        original = load_uploaded_image(uploaded)
        detected = detect(original)
        recommended_key = DETECTOR_TO_MODE_KEY.get(detected["recommended"], "dtf_ready")
        recommended_mode_name = MODE_KEYS.get(recommended_key)
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
    st.success(f"Detector automatico: {detected['name']} -> modo recomendado: {recommended_mode_name}")

render_input_summary(original, mode_name, mode)

if st.button("Procesar imagen", type="primary", use_container_width=True):
    progress = st.progress(0)
    log = st.empty()
    t0 = time.time()

    try:
        work = original.copy()
        log.write("1/7 Preparando imagen...")
        progress.progress(10)

        if options.use_ai and not has_transparency(work):
            log.write("2/7 Quitando fondo con IA...")
            ai_img = resize_for_ai(work, max_side=options.max_ai_side)
            session = cached_session()
            ai_result = remove_background_ai(ai_img, session=session)
            work = apply_ai_alpha_to_original(work, ai_result)
        else:
            log.write("2/7 IA saltada...")
        progress.progress(25)

        if options.remove_black:
            log.write("3/7 Quitando fondo negro sin borrar letras...")
            work = remove_black_background(work, threshold=options.black_threshold, softness=12, protect_details=options.protect_details)
        else:
            log.write("3/7 Quitar negro saltado...")
        progress.progress(45)

        if options.clean_enabled:
            log.write("4/7 Limpiando semitransparencias y pixeles basura...")
            work = clean_alpha(work, alpha_cut=options.alpha_cut, despeckle_area=options.despeckle_area, edge_contract=options.edge_contract)
        else:
            log.write("4/7 Limpieza alfa saltada para conservar diseno...")
        progress.progress(60)

        if options.trim:
            log.write("5/7 Recortando espacio transparente...")
            work = trim_transparent(work, padding=20)
        else:
            log.write("5/7 Recorte saltado...")
        progress.progress(70)

        log.write("6/7 Ajustando medida/resolucion...")
        work = fit_to_print_size(work, width_cm=options.width_cm, height_cm=options.height_cm, dpi=options.dpi)
        if options.upscale > 1:
            work = upscale_and_sharpen(work, scale=options.upscale)
        progress.progress(82)

        log.write("7/7 Preparando descargas...")
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

        exports = build_export_package(work, dpi=options.dpi, extra_files=extra_files)
        st.session_state["result_png"] = exports["png"]
        st.session_state["result_pdf"] = exports["pdf"]
        st.session_state["result_zip"] = exports["zip"]
        progress.progress(100)
        log.success(f"Listo en {time.time() - t0:.1f} segundos")

    except Exception as exc:
        log.error(f"Error al procesar: {exc}")

if "result_img" in st.session_state:
    st.divider()
    st.subheader("Resultado")
    bg_mode = st.radio("Vista previa", ["Transparente", "Negro", "Blanco", "Gris"], horizontal=True)
    preview = composite_preview(st.session_state["result_img"], bg_mode)

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
