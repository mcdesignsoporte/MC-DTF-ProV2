# MC DTF Pro V4

Professional Streamlit software for preparing DTF artwork. The app is designed for Streamlit Cloud and Python 3.12, with a modular architecture and a production workflow for single images and batch processing.

## Features

- Smart image detection: photograph, transparent PNG, black background, white background, logo, design, dark artwork, and DTF artwork.
- Spanish commercial workflow modes: Fotografia, PNG Transparente, Conservar Diseno, Quitar Fondo Negro, Diseno Oscuro, and Preparar DTF.
- Automatic background removal for black, white, gray, green, blue, red, yellow, purple, dominant colors, mild gradients, and moderately noisy backgrounds.
- Detection metrics for resolution, transparency, background, black percentage, text, logo, noise, splash, and shadow signals.
- Confidence score for the recommended mode.
- Automatic recommendation for the safest processing mode.
- AI background removal only for photographs.
- OpenCV black background removal with protection for letters, logos, splashes, smoke, thin contours, outlines, and shadows.
- Visor profesional con auto zoom, imagen centrada y miniaturas optimizadas para archivos grandes.
- Fondos de vista comercial: Transparente, Playera negra, Playera blanca, Sudadera, Sticker, Tarro, Taza y Calcomania.
- Tabs de revision: Resultado final, Antes/Despues, Cambios de transparencia, Alpha y Original.
- Informacion tecnica de impresion: pixeles, centimetros, DPI, modo PNG, transparencia, peso y estado DTF.
- Calidad DTF con checks de fondo, bordes, resolucion, alfa limpio y tamano de impresion.
- Histograma alfa con porcentajes transparente, solido y semitransparente.
- Centro Profesional con capacidades, modulos activos, feature flags y roadmap.
- Arquitectura modular preparada para DTF, sublimacion, DTF UV, mockups, lote, vectorizacion, negocio y automatizacion.
- PNG, PDF, and ZIP export at 300 DPI with `original.png`, `procesado.png`, `procesado.pdf`, and `metadata.json`.
- Multiple-image batch processing with detector-selected mode per image.
- Advanced Settings hidden by default.
- Tests for alpha handling, black removal, export, and transparent PNG detection.

## Run Locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud

Use:

- Python: 3.12
- Main file: `streamlit_app.py`
- Requirements: `requirements.txt`

## Architecture

```text
streamlit_app.py
requirements.txt
README.md
CHANGELOG.md
MIGRATION.md
.streamlit/config.toml
assets/styles.css
core/
features/
services/
models/
ui/
tests/
docs/
```

## Core Modules

- `core/detector.py`: analyzes transparency, black percentage, white percentage, edge density, text, logo, noise, splash, shadow, estimated processing time, and resolution.
- `core/background.py`: gates rembg usage to photograph workflows only.
- `core/black_remove.py`: removes pure background black without AI.
- `core/background_remove.py`: removes dominant color, chroma, gradient-like, and OpenCV-segmented backgrounds.
- `core/detail_protect.py`: protects letters, logos, splashes, smoke, outlines, shadows, and thin contours.
- `core/export.py`: exports PNG, PDF, and ZIP with 300 DPI metadata.
- `core/logger.py`: centralized application logging.
- `core/pipeline.py`: shared single-image and batch processing pipeline.
- `core/presets.py`: exposes stable presets for future integrations.
- `core/preview.py`: creates fast auto-fit preview composites without changing export resolution.
- `core/quality.py`: evaluates DTF readiness and alpha-channel quality.
- `core/capabilities.py`: professional capability catalog.
- `core/module_registry.py`: active module registry and phase lookup.
- `core/feature_flags.py`: optional feature flags.
- `core/non_destructive.py`: stable API for the non-destructive engine.
- `core/quality_report.py`: production QA report.

## Verify

```bash
python -m compileall streamlit_app.py core ui tests
python -m unittest tests.test_image_processing
python -m pip check
```

## Quality Rules

- Exported images preserve original dimensions unless the user explicitly requests print sizing or upscaling.
- AI inference uses reduced resolution only during inference and restores alpha to the original image size.
- The app avoids exposing advanced sliders in the main workflow.
