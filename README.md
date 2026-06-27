# MC DTF Pro V4.0.0

Aplicacion Streamlit para preparar imagenes DTF con detector automatico, modos inteligentes, limpieza de fondo negro, proteccion basica de letras y exportacion lista para produccion.

## Funciones V4.0.0

- Interfaz profesional en Streamlit.
- Detector automatico del tipo de imagen.
- Modos inteligentes: Fotografia, PNG Transparente, Conservar diseno, Fondo negro y Preparar DTF.
- Eliminacion inteligente de fondo negro.
- Proteccion basica de letras, logos y detalles claros.
- Vista previa con fondos transparente, negro, blanco y gris.
- Exportacion PNG transparente, PDF y ZIP completo.
- Preparado para Streamlit Cloud y Python 3.12.

## Ejecutar local

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Estructura

```text
streamlit_app.py
requirements.txt
.streamlit/config.toml
core/
ui/
assets/
examples/
tests/
```

## Verificacion

```bash
python -m compileall streamlit_app.py core ui tests
python -m unittest tests.test_image_processing
python -m pip check
```

## Roadmap

- V4.1: procesamiento por lotes, comparador antes/despues y vista previa sobre playera, taza y sticker.
- V4.2: gang sheet automatico, calculo de costos, historial y optimizacion de velocidad.
