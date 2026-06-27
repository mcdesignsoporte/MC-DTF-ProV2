# MC DTF Pro V4

Professional Streamlit software for preparing DTF artwork. The app is designed for Streamlit Cloud and Python 3.12, with a modular architecture and a production workflow for single images and batch processing.

## Features

- Smart image detection: photograph, transparent PNG, black background, white background, logo, design, dark artwork, and DTF artwork.
- Spanish commercial workflow modes: Fotografia, PNG Transparente, Conservar Diseno, Quitar Fondo Negro, Diseno Oscuro, and Preparar DTF.
- Detection metrics for resolution, transparency, background, black percentage, text, logo, noise, splash, and shadow signals.
- Automatic recommendation for the safest processing mode.
- AI background removal only for photographs.
- OpenCV black background removal with protection for letters, logos, splashes, smoke, thin contours, outlines, and shadows.
- Commercial previews: Transparent, Black shirt, White shirt, Sticker, Mug, Beer mug, and Hoodie.
- Before/After QA view and alpha-change inspection.
- PNG, PDF, and ZIP export at 300 DPI with PNG metadata.
- Multiple-image batch processing with status table, timing, detection, and batch ZIP export.
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
ui/
tests/
```

## Core Modules

- `core/detector.py`: analyzes transparency, black percentage, white percentage, edge density, text, logo, noise, splash, shadow, estimated processing time, and resolution.
- `core/background.py`: gates rembg usage to photograph workflows only.
- `core/black_remove.py`: removes pure background black without AI.
- `core/detail_protect.py`: protects letters, logos, splashes, smoke, outlines, shadows, and thin contours.
- `core/export.py`: exports PNG, PDF, and ZIP with 300 DPI metadata.
- `core/logger.py`: centralized application logging.
- `core/pipeline.py`: shared single-image and batch processing pipeline.
- `core/presets.py`: exposes stable presets for future integrations.
- `core/preview.py`: creates fast preview composites without changing export resolution.

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
