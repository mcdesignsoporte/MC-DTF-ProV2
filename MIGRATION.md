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

### Removed

- Dead compatibility modules that were not used by the app.
- Placeholder README files inside `assets/` and `examples/`.

### Added

- `assets/styles.css`
- `core/detail_protect.py`
- `core/export.py`
- Batch processing from the main uploader.
- Preview modes for product surfaces.

### Deployment

No repository replacement is required. Deploy the same repository to Streamlit Cloud using `streamlit_app.py`.
