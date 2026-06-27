# Changelog

## V4.0.0

- Reworked the application into a modular `core/`, `ui/`, `assets/`, and `tests/` architecture.
- Added smart detection with recommended mode, AI decision, resolution, estimated time, black percentage, white percentage, transparency, and edge density.
- Restricted rembg usage to photograph workflows.
- Rewrote black background removal with OpenCV and detail protection.
- Added detail protection for letters, logos, white splashes, smoke, thin contours, outlines, and shadows.
- Added PNG, PDF, and ZIP exports with 300 DPI metadata.
- Rebuilt the interface around a primary "What do you want to do?" workflow.
- Hid advanced sliders inside Advanced Settings.
- Added commercial preview modes for shirts, stickers, mugs, beer mugs, and hoodies.
- Added multiple-image batch processing with ZIP download.
- Removed dead modules and empty nonfunctional files.
- Added verification tests for alpha behavior, black removal, export packaging, and transparent PNG detection.

## V4.0.1

- Added a shared `core/pipeline.py` processing path for single-image and batch jobs.
- Added Before/After QA preview.
- Added alpha-change inspection preview.
- Improved product previews with studio-style surfaces, garment shadows, mug details, and sticker surfaces.
- Added batch status table with file name, status, detected type, selected mode, processing time, and resolution/error.
- Added tests for the shared pipeline and before/after preview generation.

## V4.0.2

- Completed active `core/logger.py` and `core/presets.py` modules.
- Rebuilt modes in Spanish with stable internal mode keys.
- Expanded smart detection with text, logo, noise, splash, shadow, and background metrics.
- Refactored alpha cleanup into reusable `threshold_alpha`, `despeckle`, and `contract_edge` functions.
- Added validation and type hints to image loading/export helpers.
- Added logging to the production processing pipeline.
- Added tests for cleanup behavior and preset coverage.
