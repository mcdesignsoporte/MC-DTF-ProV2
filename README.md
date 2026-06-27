# MC DTF Pro v2

Aplicación web en Streamlit para preparar imágenes para DTF.

## Funciones

- Subir imagen PNG/JPG/WEBP.
- Quitar fondo con IA usando rembg.
- Saltar IA si ya es PNG transparente.
- Limpieza de semitransparencias.
- Limpieza de píxeles basura.
- Reducción de halos por contracción de borde.
- Presets: Caricatura, General, Logo fuerte.
- Exportar PNG transparente.
- Exportar PDF a 300 DPI.
- Generar semitono básico.
- Vista previa sobre fondo transparente, negro, blanco y gris.

## Deploy en Streamlit Community Cloud

Main file path:

```text
streamlit_app.py
```

## Desarrollo local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Marca

Desarrollado para MC Creative Studio.
