from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from core.modes import MODES


@dataclass(frozen=True)
class ProcessingOptions:
    """All processing controls needed by the pipeline."""

    mode_name: str
    mode: dict[str, object]
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


def render_mode_picker(selected_mode: str) -> tuple[str, dict[str, object]]:
    """Render the primary product decision instead of a wall of sliders."""
    st.subheader("What do you want to do?")
    labels = ["Photograph", "Transparent PNG", "Preserve artwork", "Remove black background", "Dark artwork", "Prepare for DTF"]
    selected = st.radio("Main action", labels, index=labels.index(selected_mode) if selected_mode in labels else 3, horizontal=True, key="selected_mode")
    mode = MODES[selected]
    st.caption(str(mode["description"]))
    return selected, mode


def _advanced_controls(mode: dict[str, object]) -> dict[str, object]:
    with st.expander("Advanced Settings", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        alpha_cut = col_a.slider("Alpha cutoff", 1, 254, int(mode["alpha_cut"]))
        black_threshold = col_b.slider("Black threshold", 1, 80, int(mode["black_threshold"]))
        despeckle_area = col_c.slider("Trash pixel area", 1, 500, int(mode["despeckle_area"]))
        edge_contract = col_a.slider("Edge contract", 0, 4, int(mode["edge_contract"]))
        max_ai_side = col_b.slider("AI inference size", 800, 2400, int(mode["max_ai_side"]), step=100)
        upscale = col_c.selectbox("Upscale", [1, 2, 3, 4], index=0, format_func=lambda x: "Original" if x == 1 else f"{x}x")
        protect_details = st.checkbox("Protect letters, logos, smoke, splashes, and thin contours", value=bool(mode["protect_details"]))
    return {
        "alpha_cut": alpha_cut,
        "black_threshold": black_threshold,
        "despeckle_area": despeckle_area,
        "edge_contract": edge_contract,
        "max_ai_side": max_ai_side,
        "upscale": upscale,
        "protect_details": protect_details,
    }


def render_sidebar(selected_mode: str) -> ProcessingOptions:
    """Render compact controls and keep expert options hidden by default."""
    mode_name, mode = render_mode_picker(selected_mode)
    controls = _advanced_controls(mode)
    with st.sidebar:
        st.header("Output")
        dpi = st.number_input("DPI", min_value=72, max_value=600, value=300, step=1)
        width_cm = st.number_input("Final width cm", min_value=0.0, value=0.0, step=0.5)
        height_cm = st.number_input("Final height cm", min_value=0.0, value=0.0, step=0.5)
        st.header("Halftone")
        make_halftone = st.checkbox("Create halftone", value=False)
        dot_size = st.slider("Dot size", 4, 40, 8)
        angle = st.slider("Angle", 0, 90, 15)
        invert_halftone = st.checkbox("Invert halftone", value=False)

    return ProcessingOptions(
        mode_name=mode_name,
        mode=mode,
        use_ai=bool(mode["use_ai"]),
        remove_black=bool(mode["remove_black"]),
        clean_enabled=bool(mode["clean_alpha"]),
        trim=bool(mode["trim"]),
        alpha_cut=int(controls["alpha_cut"]),
        despeckle_area=int(controls["despeckle_area"]),
        edge_contract=int(controls["edge_contract"]),
        black_threshold=int(controls["black_threshold"]),
        protect_details=bool(controls["protect_details"]),
        max_ai_side=int(controls["max_ai_side"]),
        upscale=int(controls["upscale"]),
        dpi=int(dpi),
        width_cm=float(width_cm),
        height_cm=float(height_cm),
        make_halftone=make_halftone,
        dot_size=dot_size,
        angle=angle,
        invert_halftone=invert_halftone,
    )
