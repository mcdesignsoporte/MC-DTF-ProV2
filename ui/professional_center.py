from __future__ import annotations

import streamlit as st

from core.capabilities import get_capabilities
from core.feature_flags import enabled_flags
from core.module_registry import get_active_modules, professional_status
from core.version import VERSION


def render_professional_center() -> None:
    """Render the professional architecture and roadmap status."""
    with st.expander("Centro Profesional", expanded=False):
        status = professional_status()
        st.subheader("Estado del software")
        st.metric("Version", VERSION)
        st.metric("Motor", str(status["engine"]))
        st.metric("Modulos activos", str(status["active_modules"]))
        st.success("Modo seguro activo por defecto")

        st.subheader("Funciones activas")
        for module in get_active_modules():
            st.write(f"OK {module['id']} | fase {module['phase']}")

        st.subheader("Capacidades")
        for capability in get_capabilities():
            st.write(f"{capability['name']} - {capability['status']} - fase {capability['phase']}")

        st.subheader("Feature flags")
        for flag, enabled in enabled_flags().items():
            st.write(f"{flag}: {'activo' if enabled else 'planeado'}")

        st.subheader("Roadmap")
        st.caption("Fase 1 estable: motor no destructivo, QA visual y exportacion limpia. Fases 2-8 preparadas en arquitectura modular.")
