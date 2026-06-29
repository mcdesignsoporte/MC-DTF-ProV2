from __future__ import annotations

import streamlit as st

from core.quality import QualityReport


def render_quality(report: QualityReport, status: str | None = None, reasons: list[object] | None = None) -> None:
    """Render DTF quality score and alpha distribution."""
    st.subheader("Calidad DTF")
    st.metric("Puntuacion", f"{report.stars} {report.score}%")
    st.caption(status or report.status)
    for reason in list(reasons or [])[:5]:
        st.warning(str(reason))
    for label, ok in report.checks.items():
        st.write(f"{'OK' if ok else '-'} {label}")
    st.subheader("Canal alfa")
    alpha = report.alpha
    st.metric("Transparente", f"{alpha.transparent_percent}%")
    st.metric("Solido", f"{alpha.solid_percent}%")
    st.metric("Semitransparente", f"{alpha.semitransparent_percent}%")
    if alpha.semitransparent_percent > 18:
        st.warning("Hay demasiados pixeles semitransparentes.")
