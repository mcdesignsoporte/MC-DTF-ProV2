import time
from io import BytesIO

import streamlit as st
from PIL import Image

from core.background import get_rembg_session, has_transparency, remove_background_ai, resize_for_ai
from core.clean import clean_alpha, trim_transparent
from core.halftone import make_halftone
from core.image_io import image_to_pdf_bytes, image_to_png_bytes, load_uploaded_image
from core.presets import PRESETS
from core.preview import composite_preview
from core.resize import fit_to_print_size, upscale_and_sharpen

st.set_page_config(page_title="MC DTF Pro", page_icon="🎨", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.mc-title {font-size: 3rem; font-weight: 900; margin-bottom: 0;}
.mc-tag {color:#d9aa32; text-transform: uppercase; letter-spacing: .18rem; font-weight: 800;}
.mc-card {border:1px solid #333; border-radius: 18px; padding: 1rem; background:#151515;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mc-tag">MC Creative Studio</div>', unsafe_allow_html=True)
st.markdown('<h1 class="mc-title">MC DTF Pro v2</h1>', unsafe_allow_html=True)
st.caption("Quita fondo, limpia semitransparencias, elimina pixeles basura y exporta archivos para DTF.")

@st.cache_resource(show_spinner="Cargando modelo IA de fondo...")
def cached_session():
    return get_rembg_session()

with st.sidebar:
    st.header("Ajustes")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    use_ai = st.checkbox("Quitar fondo con IA", value=True)
    skip_if_transparent = st.checkbox("Saltar IA si ya tiene transparencia", value=True)
    trim = st.checkbox("Recortar espacio transparente", value=True)

    st.subheader("Limpieza")
    alpha_cut = st.slider("Corte de transparencia", 1, 254, int(preset["alpha_cut"]))
    despeckle_area = st.slider("Quitar basura menor a", 1, 500, int(preset["despeckle_area"]))
    edge_contract = st.slider("Contraer borde", 0, 4, int(preset["edge_contract"]))

    st.subheader("Tamaño")
    max_ai_side = st.slider("Tamaño máximo para IA", 800, 2400, int(preset["max_ai_side"]), step=100)
    upscale = st.selectbox("Alta resolución", [1, 2, 3, 4], index=0, format_func=lambda x: "Original" if x == 1 else f"{x}x")
    dpi = st.number_input("DPI", min_value=72, max_value=600, value=300, step=1)
    width_cm = st.number_input("Ancho final cm (opcional)", min_value=0.0, value=0.0, step=0.5)
    height_cm = st.number_input("Alto final cm (opcional)", min_value=0.0, value=0.0, step=0.5)

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
    st.caption(f"{original.width} x {original.height} px")

if st.button("Procesar imagen", type="primary", use_container_width=True):
    progress = st.progress(0)
    log = st.empty()
    t0 = time.time()

    try:
        log.write("1/6 Preparando imagen...")
        work = original.copy()
        progress.progress(10)

        if use_ai and not (skip_if_transparent and has_transparency(work)):
            log.write("2/6 Quitando fondo con IA...")
            ai_img = resize_for_ai(work, max_ai_side=max_ai_side)
            session = cached_session()
            work = remove_background_ai(ai_img, session=session)
        else:
            log.write("2/6 IA saltada...")
        progress.progress(35)

        log.write("3/6 Limpiando transparencia y pixeles basura...")
        work = clean_alpha(work, alpha_cut=alpha_cut, despeckle_area=despeckle_area, edge_contract=edge_contract)
        if trim:
            work = trim_transparent(work, padding=20)
        progress.progress(55)

        log.write("4/6 Ajustando tamaño...")
        work = fit_to_print_size(work, width_cm=width_cm, height_cm=height_cm, dpi=int(dpi))
        work = upscale_and_sharpen(work, scale=upscale)
        progress.progress(70)

        st.session_state["result_img"] = work
        st.session_state["result_png"] = image_to_png_bytes(work, dpi=int(dpi))
        progress.progress(85)

        log.write("5/6 Preparando exportaciones...")
        st.session_state["result_pdf"] = image_to_pdf_bytes(work, dpi=int(dpi), white_background=True)

        if make_ht:
            log.write("6/6 Generando semitono...")
            ht = make_halftone(work, dot_size=dot_size, angle=angle, invert=invert_ht)
            st.session_state["halftone_img"] = ht
            st.session_state["halftone_png"] = image_to_png_bytes(ht, dpi=int(dpi))
            st.session_state["halftone_pdf"] = image_to_pdf_bytes(ht, dpi=int(dpi), white_background=True)
        else:
            st.session_state.pop("halftone_img", None)
            st.session_state.pop("halftone_png", None)
            st.session_state.pop("halftone_pdf", None)

        progress.progress(100)
        elapsed = time.time() - t0
        log.success(f"Listo en {elapsed:.1f} segundos")
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
        st.metric("Tamaño final", f"{result_img.width} x {result_img.height}px")
        st.download_button("Descargar PNG transparente", data=st.session_state["result_png"], file_name="mc_dtf_pro_limpio.png", mime="image/png", use_container_width=True)
        st.download_button("Descargar PDF", data=st.session_state["result_pdf"], file_name="mc_dtf_pro_limpio.pdf", mime="application/pdf", use_container_width=True)

    if "halftone_img" in st.session_state:
        st.subheader("Semitono")
        st.image(composite_preview(st.session_state["halftone_img"], bg_mode), use_container_width=True)
        st.download_button("Descargar PNG semitono", data=st.session_state["halftone_png"], file_name="mc_dtf_pro_semitono.png", mime="image/png")
        st.download_button("Descargar PDF semitono", data=st.session_state["halftone_pdf"], file_name="mc_dtf_pro_semitono.pdf", mime="application/pdf")
