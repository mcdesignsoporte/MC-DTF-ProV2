from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from core.pipeline import PipelineSettings, process_artwork
from core.quality_report import quality_report
from ui.preview import render_result_workspace


class QualityReportCompatibilityTests(unittest.TestCase):
    def test_quality_report_accepts_2d_background_mask(self) -> None:
        original = Image.new("RGBA", (12, 10), (255, 255, 255, 255))
        result = Image.new("RGBA", (12, 10), (255, 255, 255, 0))
        mask = np.zeros((10, 12), dtype=bool)
        mask[:4, :] = True

        report = quality_report(original, result, background_mask=mask)

        self.assertGreater(report["removed_background_percent"], 0)

    def test_quality_report_accepts_none_background_mask(self) -> None:
        original = Image.new("RGBA", (12, 10), (255, 255, 255, 255))
        result = Image.new("RGBA", (12, 10), (255, 255, 255, 0))

        report = quality_report(original, result, background_mask=None)

        self.assertEqual(100.0, report["removed_background_percent"])

    def test_quality_report_normalizes_3d_background_mask(self) -> None:
        original = Image.new("RGBA", (12, 10), (255, 255, 255, 255))
        result = Image.new("RGBA", (12, 10), (255, 255, 255, 0))
        mask = np.zeros((10, 12, 4), dtype=np.uint8)
        mask[2:8, 2:10, 3] = 255

        report = quality_report(original, result, background_mask=mask)

        self.assertGreater(report["removed_background_percent"], 0)

    def test_quality_report_handles_trimmed_result_size(self) -> None:
        original = Image.new("RGBA", (40, 30), (255, 255, 255, 255))
        result = Image.new("RGBA", (20, 12), (255, 255, 255, 0))

        report = quality_report(original, result, background_mask=None)

        self.assertIn("score", report)
        self.assertGreaterEqual(report["removed_background_percent"], 0)

    def test_quality_report_with_complex_white_flow(self) -> None:
        original = _complex_white_artwork()
        payload = process_artwork(original, {"type": "Fondo blanco complejo"}, _complex_settings())

        report = quality_report(original, payload["image"], background_mask=None)

        self.assertIn("removed_background_percent", report)
        self.assertIsNotNone(payload["complex_white_debug"])

    def test_render_result_workspace_does_not_crash_for_complex_white(self) -> None:
        original = _complex_white_artwork()
        payload = process_artwork(original, {"type": "Fondo blanco complejo"}, _complex_settings())
        dummy = _DummyStreamlit()

        with patch("ui.preview.st", dummy), patch("ui.quality.st", dummy):
            render_result_workspace(
                original,
                payload["image"],
                300,
                payload["png"],
                complex_white_debug=payload["complex_white_debug"],
            )


class _DummyStreamlit:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index]

    def tabs(self, labels):
        return [self for _ in labels]

    def columns(self, spec, **_kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(count)]

    def color_picker(self, _label, value, **_kwargs):
        return value

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


def _complex_white_artwork() -> Image.Image:
    img = Image.new("RGBA", (64, 48), (255, 255, 255, 255))
    img.alpha_composite(Image.new("RGBA", (30, 24), (220, 40, 120, 255)), (17, 12))
    img.alpha_composite(Image.new("RGBA", (8, 6), (255, 255, 255, 255)), (28, 22))
    return img


def _complex_settings() -> PipelineSettings:
    return PipelineSettings(
        mode_key="complex_white_bg",
        use_ai=False,
        remove_black=False,
        remove_color=False,
        clean_enabled=True,
        trim=True,
        alpha_cut=45,
        despeckle_area=2,
        edge_contract=0,
        black_threshold=24,
        black_level="normal",
        color_tolerance=58,
        protect_details=True,
        protect_white_details=True,
        white_protection_level="maxima",
        fine_detail_level="maxima",
        safe_mode=True,
        enable_dtf_prepress=False,
        remove_white_halo=False,
        remove_black_halo=False,
        halo_strength="suave",
        expand_edge_px=0,
        bleed_px=0,
        create_cutline=False,
        min_printable_mm=1.0,
        logo_detect_colors=False,
        logo_reduce_colors=False,
        logo_black_to_transparent=False,
        logo_white_to_transparent=False,
        logo_unify_colors=False,
        logo_separate_colors=False,
        logo_export_layers=False,
        logo_max_colors=8,
        logo_color_tolerance=24,
        max_ai_side=1200,
        upscale=1,
        dpi=300,
        width_cm=0,
        height_cm=0,
        complex_white_preset="Balanceado",
        complex_white_tolerance=64,
        complex_white_luminosity=222,
        complex_white_saturation=48,
        complex_white_preserve_internal=True,
        complex_white_halo_cleanup=True,
        complex_white_mask_offset=0,
        complex_white_alpha_smoothing=1,
        complex_white_export_debug=False,
    )
