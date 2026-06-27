import time

import streamlit as st

from core.background import apply_ai_alpha_to_original, get_rembg_session, has_transparency, remove_background_ai, resize_for_ai
from core.black_remove import remove_black_background
from core.clean import clean_alpha, trim_transparent
from core.halftone import make_halftone
from core.image_io import image_to_pdf_bytes, image_to_png_bytes, load_uploaded_image
from core.modes import MODES
from core.preview import composite_preview
from core.resize import fit_to_print_size, upscale_and_sharpen

st.set_page_config(page_title="MC DTF Pro v3", page_icon="🎨", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1450px;}
.mc-title {font-size: 3rem; font-weight: 900; margin-bottom: 0;}
.mc-tag {color:#d9aa32; text-transform: uppercase; letter-spacing:.18rem; font-weight:800;}
.mc-help {padding: .85rem 1rem; border: 1px solid #3b3b3b; border-radius: 14px; background:#151515;}
.stButton button {font-weight: 800;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="mc-tag">MC Creative Studio</div>', unsafe_allow_html=True)
st.markdown('<h1 class="mc-title">MC DTF Pro v3</h1>', unsafe_allow_html=True)
st.caption("Herramienta DTF con modos separados: fotografía, PNG transparente, diseño completo, fondo negro y preparación para impresión.")


@st.cache_resource(show_spinner="Cargando modelo IA de fondo...")
def cached_session():
    return get_rembg_session()


with st.sidebar:
    st.header("¿Qué deseas hacer?")
    mode_name = st.radio("Modo de trabajo", list(MODES.keys()), index=3)
    mode = MODES[mode_name]
    st.markdown(f"<div class='mc-help'>{mode['description']}</div>", unsafe_allow_html=True)

    st.subheader("Opciones")
    use_ai = st.checkbox("Usar IA", value=bool(mode["use_ai"]), disabled=not bool(mode["use_ai"]))
    remove_black = st.checkbox("Quitar negro", value=bool(mode["remove_black"]))
    clean_enabled = st.checkbox("Limpiar alfa / semitransparencias", value=bool(mode["clean_alpha"]))
    trim = st.checkbox("Recortar espacio transparente", value=bool(mode["trim"]))

    st.subheader("Ajustes finos")
    alpha_cut = st.slider("Corte transparencia", 1, 254, int(mode["alpha_cut"]))
    despeckle_area = st.slider("Basura menor a", 1, 500, int(mode["despeckle_area"]))
    edge_contract = st.slider("Contraer borde", 0, 4, int(mode["edge_contract"]))
    black_threshold = st.slider("Umbral negro", 1, 80, int(mode["black_threshold"]))
    protect_details = st.checkbox("Proteger letras/detalles", value=bool(mode["protect_details"]))

    st.subheader("Tamaño / salida")
    max_ai_side = st.slider("Tamaño IA", 800, 2400, int(mode["max_ai_side"]), step=100)
    upscale = st.selectbox("Alta resolución", [1, 2, 3, 4], index=0, format_func=lambda x: "Original" if x == 1 else f"{x}x")
    dpi = st.number_input("DPI", min_value=72, max_value=600, value=300, step=1)
    width_cm = st.number_input("Ancho final cm", min_value=0.0, value=0.0, step=0.5)
    height_cm = st.number_input("Alto final cm", min_value=0.0, value=0.0, step=0.5)

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

left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(original, use_container_width=True)
    st.caption(f"{original.width} × {original.height}px")
with right:
    st.subheader("Modo seleccionado")
    st.success(mode_name)
    if mode["key"] in ["black_bg", "dark_design"]:
        st.warning("Para diseños tipo Mustang/Silverado usa este modo: elimina negro puro, conserva letras y efectos blancos.")
    elif mode["key"] == "preserve_full":
        st.info("Este modo NO elimina fondo: conserva todo el diseño completo.")

if st.button("Procesar imagen", type="primary", use_container_width=True):
    progress = st.progress(0)
    log = st.empty()
    t0 = time.time()

    try:
        work = original.copy()
        log.write("1/7 Preparando imagen...")
        progress.progress(10)

        if use_ai and not has_transparency(work):
            log.write("2/7 Quitando fondo con IA...")
            ai_img = resize_for_ai(work, max_side=max_ai_side)
            session = cached_session()
            ai_result = remove_background_ai(ai_img, session=session)
            work = apply_ai_alpha_to_original(work, ai_result)
        else:
            log.write("2/7 IA saltada...")
        progress.progress(25)

        if remove_black:
            log.write("3/7 Quitando fondo negro sin borrar letras...")
            work = remove_black_background(work, threshold=black_threshold, softness=12, protect_details=protect_details)
        else:
            log.write("3/7 Quitar negro saltado...")
        progress.progress(45)

        if clean_enabled:
            log.write("4/7 Limpiando semitransparencias y píxeles basura...")
            work = clean_alpha(work, alpha_cut=alpha_cut, despeckle_area=despeckle_area, edge_contract=edge_contract)
        else:
            log.write("4/7 Limpieza alfa saltada para conservar diseño...")
        progress.progress(60)

        if trim:
            log.write("5/7 Recortando espacio transparente...")
            work = trim_transparent(work, padding=20)
        else:
            log.write("5/7 Recorte saltado...")
        progress.progress(70)

        log.write("6/7 Ajustando medida/resolución...")
        work = fit_to_print_size(work, width_cm=width_cm, height_cm=height_cm, dpi=int(dpi))
        if upscale > 1:
            work = upscale_and_sharpen(work, scale=upscale)
        progress.progress(82)

        log.write("7/7 Preparando descargas...")
        st.session_state["result_img"] = work
        st.session_state["result_png"] = image_to_png_bytes(work, dpi=int(dpi))
        st.session_state["result_pdf"] = image_to_pdf_bytes(work, dpi=int(dpi), white_background=True)

        if make_ht:
            ht = make_halftone(work, dot_size=dot_size, angle=angle, invert=invert_ht)
            st.session_state["halftone_img"] = ht
            st.session_state["halftone_png"] = image_to_png_bytes(ht, dpi=int(dpi))
            st.session_state["halftone_pdf"] = image_to_pdf_bytes(ht, dpi=int(dpi), white_background=True)
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

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.image(preview, use_container_width=True)
    with col_b:
        result = st.session_state["result_img"]
        st.metric("Tamaño final", f"{result.width} × {result.height}px")
        st.download_button("Descargar PNG transparente", st.session_state["result_png"], "mc_dtf_pro_v3.png", "image/png", use_container_width=True)
        st.download_button("Descargar PDF", st.session_state["result_pdf"], "mc_dtf_pro_v3.pdf", "application/pdf", use_container_width=True)

    if "halftone_img" in st.session_state:
        st.subheader("Semitono")
        st.image(composite_preview(st.session_state["halftone_img"], bg_mode), use_container_width=True)
        st.download_button("Descargar PNG semitono", st.session_state["halftone_png"], "mc_dtf_pro_v3_semitono.png", "image/png")
        st.download_button("Descargar PDF semitono", st.session_state["halftone_pdf"], "mc_dtf_pro_v3_semitono.pdf", "application/pdf")
