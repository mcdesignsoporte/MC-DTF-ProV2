# Migration Notes

## From Earlier V4 Builds

The app remains in the same repository and still starts from `streamlit_app.py`.

### Changed

- Mode keys changed to stable product keys:
  - `photograph`
  - `transparent_png`
  - `preserve_artwork`
  - `black_bg`
  - `dark_artwork`
  - `dtf_ready`
- Detection now returns:

```python
{
    "type": "...",
    "recommended_mode": "...",
    "use_ai": True,
}
```

Additional metrics are included for the UI.

In V4.0.2, the UI labels are Spanish commercial labels while the mode keys above remain stable for integrations.

Detection now also includes:

- `text_score`
- `logo_score`
- `noise_score`
- `shadow_score`
- `splash_score`
- `background`

### Removed

- Dead compatibility modules that were not used by the app.
- Placeholder README files inside `assets/` and `examples/`.

### Added

- `assets/styles.css`
- `core/detail_protect.py`
- `core/export.py`
- `core/logger.py`
- `core/presets.py`
- `core/background_remove.py`
- Batch processing from the main uploader.
- Modos de vista para superficies comerciales.

### V4.1.0

- `auto` mode can choose AI, color removal, black removal, alpha cleaning, or OpenCV segmentation.
- New stable mode key: `color_bg`.
- Detection payload includes:
  - `dominant_color`
  - `background_uniformity`

### V4.2.0

- The result viewer now uses fitted preview thumbnails for large images.
- Exported PNG, PDF, and ZIP files still preserve the processed image resolution.
- Los modos de vista se movieron a tabs profesionales.
- DTF quality data is available through `core.quality.evaluate_dtf_quality`.
- Alpha distribution is available through `core.quality.alpha_histogram`.

### Deployment

No repository replacement is required. Deploy the same repository to Streamlit Cloud using `streamlit_app.py`.
