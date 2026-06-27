from dataclasses import dataclass

import streamlit as st

from core.modes import MODES


@dataclass(frozen=True)
class ProcessingOptions:
    mode_name: str
    mode: dict
    use_ai: bool
    remove_black: bool
    clean_enabled: bool
    trim: bool
    alpha_cut: int
    despeckle_area: int
    edge_contract: int
    black_threshold: int
    protect_details: bool
    max_ai_side: int
    upscale: int
    dpi: int
    width_cm: float
    height_cm: float
    make_halftone: bool
    dot_size: int
    angle: int
    invert_halftone: bool


def render_sidebar(selected_mode: str) -> ProcessingOptions:
    with st.sidebar:
        st.header("Modo de trabajo")
        mode_names = list(MODES.keys())
        mode_index = mode_names.index(selected_mode) if selected_mode in mode_names else 3
        mode_name = st.radio("Seleccion", mode_names, index=mode_index, key="selected_mode")
        mode = MODES[mode_name]
        st.markdown(f"<div class='mc-help'>{mode['description']}</div>", unsafe_allow_html=True)

        st.subheader("Opciones inteligentes")
        use_ai = st.checkbox("Usar IA", value=bool(mode["use_ai"]), disabled=not bool(mode["use_ai"]))
        remove_black = st.checkbox("Quitar negro", value=bool(mode["remove_black"]))
        clean_enabled = st.checkbox("Limpiar alfa", value=bool(mode["clean_alpha"]))
        trim = st.checkbox("Recortar transparente", value=bool(mode["trim"]))

        st.subheader("Ajustes avanzados")
        alpha_cut = st.slider("Corte transparencia", 1, 254, int(mode["alpha_cut"]))
        despeckle_area = st.slider("Basura menor a", 1, 500, int(mode["despeckle_area"]))
        edge_contract = st.slider("Contraer borde", 0, 4, int(mode["edge_contract"]))
        black_threshold = st.slider("Umbral negro", 1, 80, int(mode["black_threshold"]))
        protect_details = st.checkbox("Proteger letras/detalles", value=bool(mode["protect_details"]))

        st.subheader("Salida")
        max_ai_side = st.slider("Tamano IA", 800, 2400, int(mode["max_ai_side"]), step=100)
        upscale = st.selectbox("Alta resolucion", [1, 2, 3, 4], index=0, format_func=lambda x: "Original" if x == 1 else f"{x}x")
        dpi = st.number_input("DPI", min_value=72, max_value=600, value=300, step=1)
        width_cm = st.number_input("Ancho final cm", min_value=0.0, value=0.0, step=0.5)
        height_cm = st.number_input("Alto final cm", min_value=0.0, value=0.0, step=0.5)

        st.subheader("Semitono")
        make_halftone = st.checkbox("Crear semitono", value=False)
        dot_size = st.slider("Tamano de punto", 4, 40, 8)
        angle = st.slider("Angulo", 0, 90, 15)
        invert_halftone = st.checkbox("Invertir semitono", value=False)

    return ProcessingOptions(
        mode_name=mode_name,
        mode=mode,
        use_ai=use_ai,
        remove_black=remove_black,
        clean_enabled=clean_enabled,
        trim=trim,
        alpha_cut=alpha_cut,
        despeckle_area=despeckle_area,
        edge_contract=edge_contract,
        black_threshold=black_threshold,
        protect_details=protect_details,
        max_ai_side=max_ai_side,
        upscale=upscale,
        dpi=int(dpi),
        width_cm=width_cm,
        height_cm=height_cm,
        make_halftone=make_halftone,
        dot_size=dot_size,
        angle=angle,
        invert_halftone=invert_halftone,
    )
