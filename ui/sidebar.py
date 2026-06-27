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
    black_level: str
    color_tolerance: int
    protect_details: bool
    protect_white_details: bool
    white_protection_level: str
    fine_detail_level: str
    max_ai_side: int
    upscale: int
    dpi: int
    width_cm: float
    height_cm: float
    make_halftone: bool
    dot_size: int
    angle: int
    invert_halftone: bool


LABEL_TO_MODE = {
    "Automático": "Preparar DTF",
    "Fotografía": "Fotografia",
    "Diseño DTF": "Conservar Diseno",
    "PNG Transparente": "PNG Transparente",
    "Fondo de color": "Fondo de color",
    "Preparar DTF": "Preparar DTF",
}

MODE_TO_LABEL = {
    "Quitar Fondo Negro": "Fondo de color",
    "Diseno Oscuro": "Fondo de color",
    "Conservar Diseno": "Diseño DTF",
    "Fotografia": "Fotografía",
}


def _safe_mode_name(name: str | None) -> str:
    """Return a valid MODES key even if session state contains an old value."""
    if name in MODES:
        return str(name)

    if name in LABEL_TO_MODE and LABEL_TO_MODE[name] in MODES:
        return LABEL_TO_MODE[name]

    if name in MODE_TO_LABEL:
        label = MODE_TO_LABEL[name]
        mapped = LABEL_TO_MODE.get(label)
        if mapped in MODES:
            return mapped

    if "Preparar DTF" in MODES:
        return "Preparar DTF"

    return next(iter(MODES.keys()))


def _label_from_mode(mode_name: str) -> str:
    if mode_name in MODE_TO_LABEL:
        return MODE_TO_LABEL[mode_name]

    for label, key in LABEL_TO_MODE.items():
        if key == mode_name:
            return label

    return "Preparar DTF"


def render_mode_picker(selected_mode: str) -> tuple[str, dict[str, object]]:
    """Render the primary workflow choice without breaking Streamlit session state."""
    st.subheader("¿Qué quieres hacer?")

    labels = list(LABEL_TO_MODE.keys())
    safe_mode = _safe_mode_name(selected_mode)
    current_label = _label_from_mode(safe_mode)

    if current_label not in labels:
        current_label = "Preparar DTF"

    index = labels.index(current_label)

    selected_label = st.radio(
        "Modo principal",
        labels,
        index=index,
        horizontal=True,
        key="selected_mode_radio",
    )

    mode_key = LABEL_TO_MODE.get(selected_label, "Preparar DTF")
    mode_key = _safe_mode_name(mode_key)

    st.session_state["selected_mode"] = mode_key

    mode = MODES[mode_key]
    st.caption(str(mode.get("description", "")))

    return mode_key, mode


def _advanced_controls(mode: dict[str, object]) -> dict[str, object]:
    with st.expander("Configuración avanzada", expanded=False):
        col_a, col_b, col_c = st.columns(3)

        alpha_cut = col_a.slider(
            "Corte alfa",
            1,
            254,
            int(mode.get("alpha_cut", 70)),
        )

        black_threshold = col_b.slider(
            "Umbral negro",
            1,
            80,
            int(mode.get("black_threshold", 20)),
        )

        noise_level = col_c.selectbox(
            "Limpieza de basura",
            ["Muy suave", "Suave", "Normal", "Fuerte"],
            index=2,
        )

        edge_contract = col_a.slider(
            "Contraer borde",
            0,
            4,
            int(mode.get("edge_contract", 0)),
        )

        max_ai_side = col_b.slider(
            "Tamaño IA",
            800,
            2400,
            int(mode.get("max_ai_side", 1600)),
            step=100,
        )

        color_tolerance = col_c.slider(
            "Tolerancia color",
            8,
            96,
            int(mode.get("color_tolerance", 32)),
        )

        upscale = col_a.selectbox(
            "Upscale",
            [1, 2, 3, 4],
            index=0,
            format_func=lambda x: "Original" if x == 1 else f"{x}x",
        )

        black_level = col_b.selectbox(
            "Nivel fondo negro",
            ["Suave", "Normal", "Fuerte"],
            index=1,
        )

        protect_details = st.checkbox(
            "Proteger letras, logos, humo, salpicaduras y contornos finos",
            value=bool(mode.get("protect_details", True)),
        )
        protect_white_details = st.checkbox(
            "Proteger blancos importantes",
            value=bool(mode.get("protect_white_details", True)),
        )
        white_protection_level = st.selectbox(
            "Proteccion de blancos",
            ["Suave", "Normal", "Maxima"],
            index=1,
        )
        fine_detail_level = st.selectbox(
            "Proteccion de detalles finos",
            ["Suave", "Normal", "Maxima"],
            index=2,
        )

    return {
        "alpha_cut": alpha_cut,
        "black_threshold": black_threshold,
        "despeckle_area": _noise_area(noise_level),
        "edge_contract": edge_contract,
        "max_ai_side": max_ai_side,
        "color_tolerance": color_tolerance,
        "upscale": upscale,
        "black_level": black_level.lower(),
        "protect_details": protect_details,
        "protect_white_details": protect_white_details,
        "white_protection_level": white_protection_level.lower(),
        "fine_detail_level": fine_detail_level.lower(),
    }


def render_sidebar(selected_mode: str) -> ProcessingOptions:
    """Render compact commercial controls with expert options hidden."""
    mode_name, mode = render_mode_picker(selected_mode)
    controls = _advanced_controls(mode)

    with st.sidebar:
        st.header("Salida")

        dpi = st.number_input(
            "DPI",
            min_value=72,
            max_value=600,
            value=DEFAULT_DPI,
            step=1,
        )

        width_cm = st.number_input(
            "Ancho final cm",
            min_value=0.0,
            value=0.0,
            step=0.5,
        )

        height_cm = st.number_input(
            "Alto final cm",
            min_value=0.0,
            value=0.0,
            step=0.5,
        )

        st.header("Semitono")

        make_halftone = st.checkbox("Crear semitono", value=False)
        dot_size = st.slider("Tamaño de punto", 4, 40, 8)
        angle = st.slider("Ángulo", 0, 90, 15)
        invert_halftone = st.checkbox("Invertir semitono", value=False)

    return ProcessingOptions(
        mode_name=mode_name,
        mode=mode,
        use_ai=bool(mode.get("use_ai", False)),
        remove_black=bool(mode.get("remove_black", False)),
        remove_color=bool(mode.get("remove_color", False)),
        clean_enabled=bool(mode.get("clean_alpha", True)),
        trim=bool(mode.get("trim", True)),
        alpha_cut=int(controls["alpha_cut"]),
        despeckle_area=int(controls["despeckle_area"]),
        edge_contract=int(controls["edge_contract"]),
        black_threshold=int(controls["black_threshold"]),
        black_level=str(controls["black_level"]),
        color_tolerance=int(controls["color_tolerance"]),
        protect_details=bool(controls["protect_details"]),
        protect_white_details=bool(controls["protect_white_details"]),
        white_protection_level=str(controls["white_protection_level"]),
        fine_detail_level=str(controls["fine_detail_level"]),
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


def _noise_area(label: str) -> int:
    return {"Muy suave": 1, "Suave": 3, "Normal": 9, "Fuerte": 24}.get(label, 9)
