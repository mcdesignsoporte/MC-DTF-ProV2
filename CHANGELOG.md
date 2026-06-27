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
- Added visual Antes/Despues QA preview.
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

## V4.1.0

- Added automatic background removal for arbitrary dominant colors.
- Added color/chroma removal for green, blue, red, yellow, purple, gray, black, white, and similar solid backgrounds.
- Added OpenCV segmentation fallback for non-uniform, textured, gradient, and moderately noisy backgrounds.
- Added dominant background color and background uniformity detection.
- Added Auto and Fondo de color workflow support.
- Preserved details by preventing the color remover from restoring pixels that match the detected background.
- Added tests for dominant color background removal.

## V4.1.1

- Replaced remaining English UI labels with Spanish labels.
- Pinned stable dependency versions in `requirements.txt`.
- Added detector confidence scoring.
- Added manual/recommended configuration actions after analysis.
- Added black background removal levels: Suave, Normal, and Fuerte.
- Updated batch processing so every image uses its detector-recommended mode.
- Updated ZIP exports to include `original.png`, `procesado.png`, `procesado.pdf`, and `metadata.json`.

## V4.2.0

- Added a professional auto-fit result viewer for large images.
- Kept exports at full resolution while using optimized thumbnails only for preview.
- Replaced result review radios with tabs: Resultado final, Antes / Despues, Cambios de transparencia, Alpha, and Original.
- Added preview backgrounds for Transparente, Playera negra, Playera blanca, Sudadera, Sticker, Tarro, Taza, and Calcomania.
- Added technical print information next to the result: pixels, centimeters, DPI, PNG mode, transparency, PNG size, and DTF status.
- Added DTF quality scoring with checks for background removal, clean edges, resolution, alpha cleanliness, and print size.
- Added alpha histogram metrics for transparent, solid, and semi-transparent pixels.
- Added tests for preview thumbnail fitting, fast checkerboard generation, alpha preview, alpha histogram, and quality scoring.

## V4.2.1

- Fixed Streamlit Cloud nested column rendering in the professional result viewer.
- Moved downloads into a result expander and simplified technical/quality metrics for Streamlit 1.41 compatibility.

## V4.3.0

- Added White Detail Protection Engine for letters, eyes, highlights, smoke, stars, outlines, splashes, and reflections.
- Added automatic white protection masks using OpenCV, NumPy, morphology, connected components, edges, texture, saturation, and context.
- Added advanced settings for important white protection and levels: Suave, Normal, and Maxima.
- Added automatic Maxima protection for logos, DTF designs, stickers, and vector-style artwork.
- Added Blancos protegidos QA view with green mask preview.
- Added white protection statistics: detected, protected, removed, and protected percentage.
- Added unit tests for white letters, eyes, reflections, white background removal, and protection speed.

## V4.4.0

- Added fine detail protection for thin lines, contours, splashes, decorative dots, shadows, letter texture, and internal artwork details.
- Replaced aggressive small-component cleanup with safe despeckle based on the main artwork mask, Canny edges, saturation, contrast, and connected components.
- Added protected-detail QA view with green mask overlay.
- Added statistics for fine details detected, details protected, noise removed, and components removed.
- Converted technical trash cleanup into safe levels: Muy suave, Suave, Normal, and Fuerte.
- Added tests for small splashes, thin lines, decorative dots, isolated noise, contours, and letter texture.

## V4.5.0

- Added professional non-destructive engine: protect artwork first, remove only confirmed background, restore protected pixels, then export.
- Added main artwork mask using alpha, Canny edges, saturation, local contrast, connected components, center distance, non-background color, and expanded bounding box.
- Added confirmed background removal for uniform low-detail regions connected to canvas edges and outside protected artwork.
- Added Conservador Profesional mode and default safe mode: no borrar arte.
- Added automatic art-loss risk detection and restoration metadata.
- Added QA tabs for Fondo eliminado, Arte protegido, Detalles protegidos, and Riesgo de perdida.
- Added tests for pink car body, wheels, white letters, nearby splashes, isolated noise, and edge-connected background.
