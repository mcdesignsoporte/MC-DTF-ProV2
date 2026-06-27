from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from core.constants import DEFAULT_DPI
from core.modes import MODES


@dataclass(frozen=True)
class ProcessingOptions:
    """All processing controls needed by the pipeline."""

    mode_name: str
    mode: dict[str, object]
    use_ai: bool
    remove_black: bool
    remove_color: bool
    clean_enabled: bool
    trim: bool
    alpha_cut: int
    despeckle_area: int
    edge_contract: int
    black_threshold: int
    color_tolerance: int
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


def render_mode_picker(selected_mode: str) -> tuple[str, dict[str, object]]:
    """Render the primary workflow choice."""
    st.subheader("Que quieres hacer?")
    labels = ["Automatico", "Fotografia", "Diseno DTF", "PNG Transparente", "Fondo de color", "Preparar DTF"]
    mapped = {"Diseno DTF": "Conservar Diseno"}
    current = selected_mode if selected_mode in labels else _label_from_mode(selected_mode)
    index = labels.index(current) if current in labels else 0
    selected = st.radio("Modo principal", labels, index=index, horizontal=True, key="selected_mode")
    mode = MODES[mapped.get(selected, selected)]
    st.caption(str(mode["description"]))
    return selected, mode


def _advanced_controls(mode: dict[str, object]) -> dict[str, object]:
    with st.expander("Configuracion avanzada", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        alpha_cut = col_a.slider("Corte alfa", 1, 254, int(mode["alpha_cut"]))
        black_threshold = col_b.slider("Umbral negro", 1, 80, int(mode["black_threshold"]))
        despeckle_area = col_c.slider("Basura menor a", 1, 500, int(mode["despeckle_area"]))
        edge_contract = col_a.slider("Contraer borde", 0, 4, int(mode["edge_contract"]))
        max_ai_side = col_b.slider("Tamano IA", 800, 2400, int(mode["max_ai_side"]), step=100)
        color_tolerance = col_c.slider("Tolerancia color", 8, 96, int(mode["color_tolerance"]))
        upscale = col_a.selectbox("Upscale", [1, 2, 3, 4], index=0, format_func=lambda x: "Original" if x == 1 else f"{x}x")
        protect_details = st.checkbox("Proteger letras, logos, humo, salpicaduras y contornos finos", value=bool(mode["protect_details"]))
    return {
        "alpha_cut": alpha_cut,
        "black_threshold": black_threshold,
        "despeckle_area": despeckle_area,
        "edge_contract": edge_contract,
        "max_ai_side": max_ai_side,
        "color_tolerance": color_tolerance,
        "upscale": upscale,
        "protect_details": protect_details,
    }


def render_sidebar(selected_mode: str) -> ProcessingOptions:
    """Render compact commercial controls with expert options hidden."""
    mode_name, mode = render_mode_picker(selected_mode)
    controls = _advanced_controls(mode)
    with st.sidebar:
        st.header("Salida")
        dpi = st.number_input("DPI", min_value=72, max_value=600, value=DEFAULT_DPI, step=1)
        width_cm = st.number_input("Ancho final cm", min_value=0.0, value=0.0, step=0.5)
        height_cm = st.number_input("Alto final cm", min_value=0.0, value=0.0, step=0.5)
        st.header("Semitono")
        make_halftone = st.checkbox("Crear semitono", value=False)
        dot_size = st.slider("Tamano de punto", 4, 40, 8)
        angle = st.slider("Angulo", 0, 90, 15)
        invert_halftone = st.checkbox("Invertir semitono", value=False)

    return ProcessingOptions(
        mode_name=mode_name,
        mode=mode,
        use_ai=bool(mode["use_ai"]),
        remove_black=bool(mode["remove_black"]),
        remove_color=bool(mode["remove_color"]),
        clean_enabled=bool(mode["clean_alpha"]),
        trim=bool(mode["trim"]),
        alpha_cut=int(controls["alpha_cut"]),
        despeckle_area=int(controls["despeckle_area"]),
        edge_contract=int(controls["edge_contract"]),
        black_threshold=int(controls["black_threshold"]),
        color_tolerance=int(controls["color_tolerance"]),
        protect_details=bool(controls["protect_details"]),
        max_ai_side=int(controls["max_ai_side"]),
        upscale=int(controls["upscale"]),
        dpi=int(dpi),
        width_cm=float(width_cm),
        height_cm=float(height_cm),
        make_halftone=make_halftone,
        dot_size=int(dot_size),
        angle=int(angle),
        invert_halftone=bool(invert_halftone),
    )


def _label_from_mode(mode_name: str) -> str:
    if mode_name in {"Quitar Fondo Negro", "Diseno Oscuro"}:
        return "Fondo de color"
    if mode_name == "Conservar Diseno":
        return "Diseno DTF"
    return mode_name
