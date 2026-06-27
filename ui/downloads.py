import streamlit as st
from PIL import Image


def render_downloads(result: Image.Image, png: bytes, pdf: bytes, zip_payload: bytes) -> None:
    """Render the production download panel."""
    st.metric("Tamano final", f"{result.width} x {result.height}px")
    st.download_button("Descargar ZIP completo", zip_payload, "mc_dtf_pro_v4.zip", "application/zip", use_container_width=True)
    st.download_button("Descargar PNG transparente", png, "mc_dtf_pro_v4.png", "image/png", use_container_width=True)
    st.download_button("Descargar PDF", pdf, "mc_dtf_pro_v4.pdf", "application/pdf", use_container_width=True)
