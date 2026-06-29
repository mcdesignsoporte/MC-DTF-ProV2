from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from core.constants import DEFAULT_DPI
from core.manual_white_region import parse_seed_text
from core.modes import MODES
from core.white_complex import COMPLEX_WHITE_PRESETS, complex_white_preset


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
    safe_mode: bool
    enable_dtf_prepress: bool
    remove_white_halo: bool
    remove_black_halo: bool
    halo_strength: str
    expand_edge_px: int
    bleed_px: int
    create_cutline: bool
    min_printable_mm: float
    logo_detect_colors: bool
    logo_reduce_colors: bool
    logo_black_to_transparent: bool
    logo_white_to_transparent: bool
    logo_unify_colors: bool
    logo_separate_colors: bool
    logo_export_layers: bool
    logo_max_colors: int
    logo_color_tolerance: int
    complex_white_preset: str
    complex_white_tolerance: int
    complex_white_luminosity: int
    complex_white_saturation: int
    complex_white_preserve_internal: bool
    complex_white_halo_cleanup: bool
    complex_white_mask_offset: int
    complex_white_alpha_smoothing: int
    complex_white_export_debug: bool
    residue_refine_enabled: bool
    residue_luminosity: int
    residue_saturation: int
    residue_min_area: int
    residue_max_area: int
    residue_remove_connected: bool
    residue_remove_small: bool
    residue_preserve_internal: bool
    residue_manual_ids: tuple[int, ...]
    internal_residue_enabled: bool
    internal_residue_min_area: int
    internal_residue_max_area: int
    internal_residue_dark_sensitivity: int
    internal_residue_luminosity: int
    internal_residue_saturation: int
    internal_residue_auto_remove: bool
    internal_residue_manual_ids: tuple[int, ...]
    manual_white_enabled: bool
    manual_white_seeds: tuple[tuple[int, int], ...]
    manual_white_tolerance: int
    manual_white_luminosity: int
    manual_white_saturation: int
    manual_white_max_area: int
    manual_white_connectivity: int
    manual_white_action: str
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
    "Seguro": "Conservador Profesional",
    "Automático": "Automatico",
    "Fotografía": "Fotografia",
    "Diseño DTF": "Conservar Diseno",
    "PNG Transparente": "PNG Transparente",
    "Fondo negro": "Quitar Fondo Negro",
    "Diseño oscuro": "Diseno Oscuro",
    "Fondo de color": "Fondo de color",
    "Fondo blanco complejo": "Fondo blanco complejo",
    "Preparar DTF": "Preparar DTF",
}

MODE_TO_LABEL = {
    "Automatico": "Automático",
    "Conservador Profesional": "Seguro",
    "Fotografia": "Fotografía",
    "Conservar Diseno": "Diseño DTF",
    "PNG Transparente": "PNG Transparente",
    "Quitar Fondo Negro": "Fondo negro",
    "Diseno Oscuro": "Diseño oscuro",
    "Fondo de color": "Fondo de color",
    "Fondo blanco complejo": "Fondo blanco complejo",
    "Preparar DTF": "Preparar DTF",
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
        safe_mode = st.checkbox(
            "Modo seguro: no borrar arte",
            value=bool(mode.get("safe_mode", True)),
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
        st.subheader("Preparacion DTF")
        enable_dtf_prepress = st.checkbox("Preparar para DTF", value=True)
        remove_white_halo = st.checkbox("Eliminar halo blanco", value=True)
        remove_black_halo = st.checkbox("Eliminar halo negro", value=False)
        halo_strength = st.selectbox("Intensidad halo", ["Suave", "Normal", "Fuerte"], index=0)
        expand_edge_px = st.selectbox("Expandir colores", [0, 1, 2, 3], index=1, format_func=lambda value: f"{value} px")
        bleed_px = st.selectbox("Sangrado", [0, 1, 2, 3, 5], index=0, format_func=lambda value: f"{value} px")
        create_cutline = st.checkbox("Crear borde de corte", value=False)
        min_printable_mm = st.selectbox("Tamano minimo imprimible", [0.5, 1.0, 1.5, 2.0], index=1, format_func=lambda value: f"{value} mm")

        st.subheader("Herramientas para logos")
        logo_detect_colors = st.checkbox("Detectar colores", value=True)
        logo_reduce_colors = st.checkbox("Reducir colores", value=False)
        logo_black_to_transparent = st.checkbox("Negro a transparente", value=False)
        logo_white_to_transparent = st.checkbox("Blanco a transparente", value=False)
        logo_unify_colors = st.checkbox("Unificar colores similares", value=False)
        logo_separate_colors = st.checkbox("Separar por colores", value=False)
        logo_export_layers = st.checkbox("Exportar capas de color", value=False)
        logo_max_colors = st.selectbox("Maximo de colores", [2, 4, 6, 8, 12], index=3)
        logo_color_tolerance = st.slider("Tolerancia logos", 4, 80, 24)

        st.subheader("Fondo blanco complejo")
        complex_white_preset_name = st.selectbox("Preset", list(COMPLEX_WHITE_PRESETS.keys()), index=1)
        complex_preset = complex_white_preset(complex_white_preset_name)
        complex_white_tolerance = st.slider("Tolerancia de blanco", 20, 110, complex_preset.white_tolerance)
        complex_white_luminosity = st.slider("Umbral de luminosidad", 170, 255, complex_preset.luminosity_threshold)
        complex_white_saturation = st.slider("Umbral de saturacion", 0, 100, complex_preset.saturation_threshold)
        complex_white_preserve_internal = st.checkbox("Preservar blancos internos", value=complex_preset.preserve_internal_white)
        complex_white_halo_cleanup = st.checkbox("Limpieza de halo claro", value=complex_preset.halo_cleanup)
        mask_values = [-2, -1, 0, 1, 2]
        smooth_values = [0, 1, 2]
        complex_white_mask_offset = st.selectbox("Contraer/expandir mascara", mask_values, index=mask_values.index(complex_preset.mask_offset), format_func=lambda value: f"{value} px")
        complex_white_alpha_smoothing = st.selectbox("Suavizado de alpha", smooth_values, index=smooth_values.index(complex_preset.alpha_smoothing), format_func=lambda value: "No" if value == 0 else f"{value} px")
        complex_white_export_debug = st.checkbox("Exportar debug", value=False)

        st.subheader("Refinar residuos blancos")
        residue_refine_enabled = st.checkbox("Activar refinamiento", value=False)
        residue_luminosity = st.slider("Luminosidad residuo", 180, 255, 220)
        residue_saturation = st.slider("Saturacion residuo", 0, 120, 50)
        residue_min_area = st.number_input("Componente minimo px", min_value=1, max_value=5000, value=8, step=1)
        residue_max_area = st.number_input("Componente maximo px", min_value=10, max_value=250000, value=5000, step=50)
        residue_remove_connected = st.checkbox("Borrar conectados a transparencia", value=True)
        residue_remove_small = st.checkbox("Borrar componentes pequenos", value=False)
        residue_preserve_internal = st.checkbox("Preservar blancos internos", value=True, key="residue_preserve_internal")
        residue_manual_text = st.text_input("IDs manuales a borrar", value="", placeholder="Ejemplo: 3, 7, 12")

        st.subheader("Refinar residuos blancos internos")
        internal_residue_enabled = st.checkbox("Activar refinamiento interno", value=False)
        internal_residue_min_area = st.number_input("Residuo interno minimo px", min_value=1, max_value=2000, value=4, step=1)
        internal_residue_max_area = st.number_input("Residuo interno maximo px", min_value=10, max_value=50000, value=900, step=10)
        internal_residue_dark_sensitivity = st.slider("Sensibilidad de vecino oscuro", 10, 80, 34)
        internal_residue_luminosity = st.slider("Luminosidad interna minima", 170, 255, 218)
        internal_residue_saturation = st.slider("Saturacion interna maxima", 0, 120, 58)
        internal_residue_auto_remove = st.checkbox("Borrar automaticamente alta confianza", value=False)
        internal_residue_manual_text = st.text_input("IDs internos manuales a borrar", value="", placeholder="Ejemplo: 2, 5")

        st.subheader("Borrar zona blanca manual")
        manual_white_enabled = st.checkbox("Activar herramienta manual", value=False)
        manual_white_x = st.number_input("Coordenada X", min_value=0, value=0, step=1)
        manual_white_y = st.number_input("Coordenada Y", min_value=0, value=0, step=1)
        manual_white_seed_text = st.text_input("Varias semillas", value="", placeholder="x1,y1; x2,y2")
        manual_white_tolerance = st.slider("Tolerancia seleccion", 4, 120, 42)
        manual_white_luminosity = st.slider("Luminosidad manual minima", 120, 255, 190)
        manual_white_saturation = st.slider("Saturacion manual maxima", 0, 140, 90)
        manual_white_max_area = st.number_input("Area maxima permitida", min_value=10, max_value=500000, value=50000, step=100)
        manual_white_connectivity = st.selectbox("Conectividad", [4, 8], index=1)
        st.caption(f"Accion actual: {st.session_state.get('manual_white_action', 'preview')}")
        if st.button("Previsualizar region"):
            st.session_state["manual_white_action"] = "preview"
        if st.button("Borrar region seleccionada"):
            st.session_state["manual_white_action"] = "apply"
        if st.button("Restablecer borrado manual"):
            st.session_state["manual_white_action"] = "reset"

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
        "safe_mode": safe_mode,
        "protect_white_details": protect_white_details,
        "white_protection_level": white_protection_level.lower(),
        "fine_detail_level": fine_detail_level.lower(),
        "enable_dtf_prepress": enable_dtf_prepress,
        "remove_white_halo": remove_white_halo,
        "remove_black_halo": remove_black_halo,
        "halo_strength": halo_strength.lower(),
        "expand_edge_px": expand_edge_px,
        "bleed_px": bleed_px,
        "create_cutline": create_cutline,
        "min_printable_mm": min_printable_mm,
        "logo_detect_colors": logo_detect_colors,
        "logo_reduce_colors": logo_reduce_colors,
        "logo_black_to_transparent": logo_black_to_transparent,
        "logo_white_to_transparent": logo_white_to_transparent,
        "logo_unify_colors": logo_unify_colors,
        "logo_separate_colors": logo_separate_colors,
        "logo_export_layers": logo_export_layers,
        "logo_max_colors": logo_max_colors,
        "logo_color_tolerance": logo_color_tolerance,
        "complex_white_preset": complex_white_preset_name,
        "complex_white_tolerance": complex_white_tolerance,
        "complex_white_luminosity": complex_white_luminosity,
        "complex_white_saturation": complex_white_saturation,
        "complex_white_preserve_internal": complex_white_preserve_internal,
        "complex_white_halo_cleanup": complex_white_halo_cleanup,
        "complex_white_mask_offset": complex_white_mask_offset,
        "complex_white_alpha_smoothing": complex_white_alpha_smoothing,
        "complex_white_export_debug": complex_white_export_debug,
        "residue_refine_enabled": residue_refine_enabled,
        "residue_luminosity": residue_luminosity,
        "residue_saturation": residue_saturation,
        "residue_min_area": residue_min_area,
        "residue_max_area": residue_max_area,
        "residue_remove_connected": residue_remove_connected,
        "residue_remove_small": residue_remove_small,
        "residue_preserve_internal": residue_preserve_internal,
        "residue_manual_ids": _parse_component_ids(residue_manual_text),
        "internal_residue_enabled": internal_residue_enabled,
        "internal_residue_min_area": internal_residue_min_area,
        "internal_residue_max_area": internal_residue_max_area,
        "internal_residue_dark_sensitivity": internal_residue_dark_sensitivity,
        "internal_residue_luminosity": internal_residue_luminosity,
        "internal_residue_saturation": internal_residue_saturation,
        "internal_residue_auto_remove": internal_residue_auto_remove,
        "internal_residue_manual_ids": _parse_component_ids(internal_residue_manual_text),
        "manual_white_enabled": manual_white_enabled,
        "manual_white_seeds": _manual_seeds(int(manual_white_x), int(manual_white_y), manual_white_seed_text),
        "manual_white_tolerance": manual_white_tolerance,
        "manual_white_luminosity": manual_white_luminosity,
        "manual_white_saturation": manual_white_saturation,
        "manual_white_max_area": manual_white_max_area,
        "manual_white_connectivity": manual_white_connectivity,
        "manual_white_action": str(st.session_state.get("manual_white_action", "preview")),
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
        safe_mode=bool(controls["safe_mode"]),
        protect_white_details=bool(controls["protect_white_details"]),
        white_protection_level=str(controls["white_protection_level"]),
        fine_detail_level=str(controls["fine_detail_level"]),
        enable_dtf_prepress=bool(controls["enable_dtf_prepress"]),
        remove_white_halo=bool(controls["remove_white_halo"]),
        remove_black_halo=bool(controls["remove_black_halo"]),
        halo_strength=str(controls["halo_strength"]),
        expand_edge_px=int(controls["expand_edge_px"]),
        bleed_px=int(controls["bleed_px"]),
        create_cutline=bool(controls["create_cutline"]),
        min_printable_mm=float(controls["min_printable_mm"]),
        logo_detect_colors=bool(controls["logo_detect_colors"]),
        logo_reduce_colors=bool(controls["logo_reduce_colors"]),
        logo_black_to_transparent=bool(controls["logo_black_to_transparent"]),
        logo_white_to_transparent=bool(controls["logo_white_to_transparent"]),
        logo_unify_colors=bool(controls["logo_unify_colors"]),
        logo_separate_colors=bool(controls["logo_separate_colors"]),
        logo_export_layers=bool(controls["logo_export_layers"]),
        logo_max_colors=int(controls["logo_max_colors"]),
        logo_color_tolerance=int(controls["logo_color_tolerance"]),
        complex_white_preset=str(controls["complex_white_preset"]),
        complex_white_tolerance=int(controls["complex_white_tolerance"]),
        complex_white_luminosity=int(controls["complex_white_luminosity"]),
        complex_white_saturation=int(controls["complex_white_saturation"]),
        complex_white_preserve_internal=bool(controls["complex_white_preserve_internal"]),
        complex_white_halo_cleanup=bool(controls["complex_white_halo_cleanup"]),
        complex_white_mask_offset=int(controls["complex_white_mask_offset"]),
        complex_white_alpha_smoothing=int(controls["complex_white_alpha_smoothing"]),
        complex_white_export_debug=bool(controls["complex_white_export_debug"]),
        residue_refine_enabled=bool(controls["residue_refine_enabled"]),
        residue_luminosity=int(controls["residue_luminosity"]),
        residue_saturation=int(controls["residue_saturation"]),
        residue_min_area=int(controls["residue_min_area"]),
        residue_max_area=int(controls["residue_max_area"]),
        residue_remove_connected=bool(controls["residue_remove_connected"]),
        residue_remove_small=bool(controls["residue_remove_small"]),
        residue_preserve_internal=bool(controls["residue_preserve_internal"]),
        residue_manual_ids=tuple(controls["residue_manual_ids"]),
        internal_residue_enabled=bool(controls["internal_residue_enabled"]),
        internal_residue_min_area=int(controls["internal_residue_min_area"]),
        internal_residue_max_area=int(controls["internal_residue_max_area"]),
        internal_residue_dark_sensitivity=int(controls["internal_residue_dark_sensitivity"]),
        internal_residue_luminosity=int(controls["internal_residue_luminosity"]),
        internal_residue_saturation=int(controls["internal_residue_saturation"]),
        internal_residue_auto_remove=bool(controls["internal_residue_auto_remove"]),
        internal_residue_manual_ids=tuple(controls["internal_residue_manual_ids"]),
        manual_white_enabled=bool(controls["manual_white_enabled"]),
        manual_white_seeds=tuple(controls["manual_white_seeds"]),
        manual_white_tolerance=int(controls["manual_white_tolerance"]),
        manual_white_luminosity=int(controls["manual_white_luminosity"]),
        manual_white_saturation=int(controls["manual_white_saturation"]),
        manual_white_max_area=int(controls["manual_white_max_area"]),
        manual_white_connectivity=int(controls["manual_white_connectivity"]),
        manual_white_action=str(controls["manual_white_action"]),
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


def _parse_component_ids(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.replace(";", ",").split(","):
        text = part.strip()
        if text.isdigit():
            values.append(int(text))
    return tuple(sorted(set(values)))


def _manual_seeds(x: int, y: int, raw: str) -> tuple[tuple[int, int], ...]:
    seeds = list(parse_seed_text(raw))
    if not seeds:
        seeds.append((x, y))
    return tuple(dict.fromkeys(seeds))
