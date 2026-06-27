from __future__ import annotations

import streamlit as st

from core.quality import QualityReport


def render_quality(report: QualityReport) -> None:
    """Render DTF quality score and alpha distribution."""
    st.subheader("Calidad DTF")
    st.metric("Puntuacion", f"{report.stars} {report.score}%")
    st.caption(report.status)
    for label, ok in report.checks.items():
        st.write(f"{'✓' if ok else '•'} {label}")
    st.subheader("Canal alfa")
    a = report.alpha
    c1, c2, c3 = st.columns(3)
    c1.metric("Transparente", f"{a.transparent_percent}%")
    c2.metric("Solido", f"{a.solid_percent}%")
    c3.metric("Semitransparente", f"{a.semitransparent_percent}%")
    if a.semitransparent_percent > 18:
        st.warning("Hay demasiados pixeles semitransparentes.")
