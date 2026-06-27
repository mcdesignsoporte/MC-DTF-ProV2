from __future__ import annotations

import streamlit as st


def render_batch_table(rows: list[dict[str, object]]) -> None:
    """Render batch status without hiding failures."""
    if not rows:
        return
    st.subheader("Estado del lote")
    st.dataframe(rows, use_container_width=True, hide_index=True)
