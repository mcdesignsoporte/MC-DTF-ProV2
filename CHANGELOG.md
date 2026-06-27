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
