from __future__ import annotations

import streamlit as st

from core.module_registry import get_available_modules, professional_status
from core.version import VERSION


STATUS_LABELS = {
    "active": "Activos",
    "experimental": "Experimentales",
    "planned": "Planeados",
}


def render_professional_center() -> None:
    """Render the professional architecture and roadmap status."""
    with st.expander("Centro Profesional", expanded=False):
        status = professional_status()
        st.subheader("Estado del software")
        st.metric("Version", VERSION)
        st.metric("Motor", str(status["engine"]))
        st.metric("Modulos activos", str(status["active_modules"]))
        st.metric("Modulos experimentales", str(status["experimental_modules"]))
        st.metric("Modulos planeados", str(status["planned_modules"]))
        st.success("Modo seguro activo por defecto")

        sections = professional_center_sections()
        for state, title in STATUS_LABELS.items():
            rows = sections[state]
            st.subheader(title)
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.caption("Sin modulos en este estado.")

        st.subheader("Roadmap")
        st.caption("Fase 1 estable: motor no destructivo, QA visual y exportacion limpia. Fases 2-8 preparadas en arquitectura modular.")


def professional_center_sections() -> dict[str, list[dict[str, object]]]:
    """Return module rows grouped for the professional center."""
    sections: dict[str, list[dict[str, object]]] = {state: [] for state in STATUS_LABELS}
    for module in get_available_modules():
        if not bool(module.get("show_in_center", True)):
            continue

        state = str(module["status"])
        if state in sections:
            sections[state].append(_module_row(module))

    return sections


def _module_row(module: dict[str, object]) -> dict[str, object]:
    evidence = tuple(module.get("evidence", ()))
    return {
        "Nombre": module["name"],
        "Descripcion": module["description"],
        "Estado": module["status"],
        "Categoria": module["category"],
        "Usuario final": "Si" if bool(module.get("user_enabled", False)) else "No",
        "Evidencia": _evidence_note(evidence),
    }


def _evidence_note(evidence: tuple[object, ...]) -> str:
    if not evidence:
        return "Planeado"
    if len(evidence) == 1:
        return str(evidence[0])
    return f"{evidence[0]} (+{len(evidence) - 1})"
