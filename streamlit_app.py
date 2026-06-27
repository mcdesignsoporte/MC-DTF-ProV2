import time

import streamlit as st

from core.background import get_rembg_session, has_transparency, remove_background_ai, resize_for_ai
from core.clean import clean_alpha, remove_black_background, trim_transparent
from core.halftone import make_halftone
from core.image_io import image_to_pdf_bytes, image_to_png_bytes, load_uploaded_image
from core.presets import MODES
from core.preview import composite_preview
from core.resize import fit_to_print_size, upscale_and_sharpen

st.set_page_config(page_title="MC DTF Pro v3", page_icon="🎨", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.mc-title {font-size: 3.2rem; font-weight: 900; margin-bottom: 0; letter-spacing:-.04em;}
.mc-tag {color:#d9aa32; text-transform: uppercase; letter-spacing: .18rem; font-weight: 900;}
.mc-sub {font-size:1.1rem; color:#888; margin-bottom:1rem;}
.stButton>button {font-weight:800;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mc-tag">MC Creative Studio</div>', unsafe_allow_html=True)
st.markdown('<h1 class="mc-title">MC DTF Pro v3</h1>', unsafe_allow_html=True)
st.markdown('<div class="mc-sub">Flujo por modos: fotografía, PNG transparente, diseño completo, fondo negro y preparación DTF.</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Cargando modelo IA de fondo...")
def cached_session():
    return get_rembg_session()

with st.sidebar:
    st.header("¿Qué deseas hacer?")
    mode_name = st.radio(
        "Modo de trabajo",
        list(MODES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    mode = MODES[mode_name]
    st.info(mode["description"])

    # Reglas blindadas por modo
    force_no_ai = mode_name in ["Conservar diseño completo", "Quitar fondo negro", "Limpiar PNG transparente", "Preparar para DTF"]

    st.subheader("Opciones")
    if force_no_ai:
        use_ai = False
        st.caption("IA desactivada en este modo para evitar borrar letras, fondos o efectos importantes.")
    else:
        use_ai = st.checkbox("Quitar fondo con IA", value=mode.get("use_ai", True))

    skip_if_transparent = st.checkbox("Saltar IA si ya tiene transparencia", value=mode.get("skip_if_transparent", True))
    trim = st.checkbox("Recortar espacio transparente", value=mode.get("trim", True))
    clean_enabled = st.checkbox("Limpiar semitransparencias/pixeles", value=mode.get("clean_alpha", True))

    st.subheader("Limpieza")
    alpha_cut = st.slider("Corte de transparencia", 1, 254, int(mode.get("alpha_cut", 70)))
    despeckle_area = st.slider("Quitar basura menor a", 1, 500, int(mode.get("despeckle_area", 2)))
    edge_contract = st.slider("Contraer borde", 0, 4, int(mode.get("edge_contract", 0)))

    if mode.get("remove_black", False):
        st.subheader("Fondo negro")
        black_threshold = st.slider("Sensibilidad negro", 5, 120, int(mode.get("black_threshold", 45)))
        black_tolerance = st.slider("Tolerancia color negro", 5, 80, int(mode.get("black_tolerance", 22)))
    else:
        black_threshold = int(mode.get("black_threshold", 45))
        black_tolerance = int(mode.get("black_tolerance", 22))

    st.subheader("Tamaño / Exportación")
    max_ai_side = st.slider("Tamaño máximo para IA", 800, 2400, int(mode.get("max_ai_side", 1600)), step=100)
    upscale = st.selectbox("Alta resolución", [1, 2, 3, 4], index=0, format_func=lambda x: "Original" if x == 1 else f"{x}x")
    dpi = st.number_input("DPI", min_value=72, max_value=600, value=300, step=1)
    width_cm = st.number_input("Ancho final cm opcional", min_value=0.0, value=0.0, step=0.5)
    height_cm = st.number_input("Alto final cm opcional", min_value=0.0, value=0.0, step=0.5)
    generate_pdf = st.checkbox("Generar PDF", value=True)

    st.subheader("Semitono")
    make_ht = st.checkbox("Crear semitono", value=False)
    dot_size = st.slider("Tamaño de punto", 4, 40, 8)
    angle = st.slider("Ángulo", 0, 90, 15)
    invert_ht = st.checkbox("Invertir semitono", value=False)

uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])

if not uploaded:
    st.info("Sube una imagen para comenzar.")
    st.stop()

try:
    original = load_uploaded_image(uploaded)
except Exception as exc:
    st.error(f"No se pudo abrir la imagen: {exc}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(original, use_container_width=True)
    st.caption(f"{original.width} × {original.height}px")
with col2:
    st.subheader("Modo activo")
    st.success(mode_name)
    if mode_name == "Conservar diseño completo":
        st.warning("Este modo NO quita fondo. Es para mantener letras, splash, sombras y composición completa.")
    if mode_name == "Quitar fondo negro":
        st.warning("Este modo elimina negro conectado a los bordes. Si borra detalles, baja la sensibilidad.")

if st.button("Procesar imagen", type="primary", use_container_width=True):
    progress = st.progress(0)
    log = st.empty()
    t0 = time.time()

    try:
        log.write("1/7 Preparando imagen...")
        work = original.copy()
        progress.progress(10)

        if use_ai and not (skip_if_transparent and has_transparency(work)):
            log.write("2/7 Quitando fondo con IA...")
            ai_img = resize_for_ai(work, max_side=max_ai_side)
            session = cached_session()
            work = remove_background_ai(ai_img, session=session)
        else:
            log.write("2/7 IA saltada por el modo seleccionado.")
        progress.progress(28)

        if mode.get("remove_black", False):
            log.write("3/7 Quitando fondo negro conectado al borde...")
            work = remove_black_background(work, threshold=black_threshold, tolerance=black_tolerance)
        else:
            log.write("3/7 Fondo negro no aplicado.")
        progress.progress(42)

        if clean_enabled:
            log.write("4/7 Limpiando transparencia y pixeles basura...")
            work = clean_alpha(work, alpha_cut=alpha_cut, despeckle_area=despeckle_area, edge_contract=edge_contract)
        else:
            log.write("4/7 Limpieza alpha saltada para conservar diseño completo.")
        progress.progress(56)

        if trim:
            log.write("5/7 Recortando espacio transparente...")
            work = trim_transparent(work, padding=20)
        else:
            log.write("5/7 Recorte saltado.")
        progress.progress(66)

        log.write("6/7 Ajustando tamaño...")
        work = fit_to_print_size(work, width_cm=width_cm, height_cm=height_cm, dpi=int(dpi))
        if upscale > 1:
            work = upscale_and_sharpen(work, scale=upscale)
        progress.progress(78)

        log.write("7/7 Preparando descargas...")
        st.session_state["result_img"] = work
        st.session_state["result_png"] = image_to_png_bytes(work, dpi=int(dpi))
        st.session_state["result_pdf"] = image_to_pdf_bytes(work, dpi=int(dpi), white_background=True) if generate_pdf else None

        if make_ht:
            ht = make_halftone(work, dot_size=dot_size, angle=angle, invert=invert_ht)
            st.session_state["halftone_img"] = ht
            st.session_state["halftone_png"] = image_to_png_bytes(ht, dpi=int(dpi))
            st.session_state["halftone_pdf"] = image_to_pdf_bytes(ht, dpi=int(dpi), white_background=True) if generate_pdf else None
        else:
            for key in ["halftone_img", "halftone_png", "halftone_pdf"]:
                st.session_state.pop(key, None)

        progress.progress(100)
        log.success(f"Listo en {time.time() - t0:.1f} segundos")
    except Exception as exc:
        log.error(f"Error al procesar: {exc}")

if "result_img" in st.session_state:
    st.divider()
    st.subheader("Resultado")
    bg_mode = st.radio("Vista previa", ["Transparente", "Negro", "Blanco", "Gris"], horizontal=True)
    preview = composite_preview(st.session_state["result_img"], bg_mode)

    colA, colB = st.columns([2, 1])
    with colA:
        st.image(preview, use_container_width=True)
    with colB:
        result_img = st.session_state["result_img"]
        st.metric("Tamaño final", f"{result_img.width} × {result_img.height}px")
        st.download_button("Descargar PNG", data=st.session_state["result_png"], file_name="mc_dtf_pro_v3.png", mime="image/png", use_container_width=True)
        if st.session_state.get("result_pdf"):
            st.download_button("Descargar PDF", data=st.session_state["result_pdf"], file_name="mc_dtf_pro_v3.pdf", mime="application/pdf", use_container_width=True)

    if "halftone_img" in st.session_state:
        st.subheader("Semitono")
        st.image(composite_preview(st.session_state["halftone_img"], bg_mode), use_container_width=True)
        st.download_button("Descargar PNG semitono", data=st.session_state["halftone_png"], file_name="mc_dtf_pro_v3_semitono.png", mime="image/png")
        if st.session_state.get("halftone_pdf"):
            st.download_button("Descargar PDF semitono", data=st.session_state["halftone_pdf"], file_name="mc_dtf_pro_v3_semitono.pdf", mime="application/pdf")
