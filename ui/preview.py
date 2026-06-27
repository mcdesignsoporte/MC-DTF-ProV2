import streamlit as st
from PIL import Image


def render_input_summary(original: Image.Image, mode_name: str, mode: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(original, use_container_width=True)
        st.caption(f"{original.width} x {original.height}px")
    with right:
        st.subheader("Modo seleccionado")
        st.success(mode_name)
        if mode["key"] in ["black_bg", "dark_design"]:
            st.warning("Elimina negro puro y conserva letras, grises, cromados y efectos blancos.")
        elif mode["key"] == "preserve_full":
            st.info("Conserva el diseno completo sin eliminar fondo.")


def render_result_preview(preview_img: Image.Image) -> None:
    st.image(preview_img, use_container_width=True)
