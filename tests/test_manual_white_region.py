from __future__ import annotations

import unittest
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from core.manual_white_region import (
    ManualWhiteRegionSettings,
    manual_white_region_report,
    parse_seed_text,
    preview_selected_region_overlay,
    remove_light_region_by_seed,
    select_light_region_by_seed,
)
from core.pipeline import PipelineSettings, process_artwork
from core.quality_report import quality_report


class ManualWhiteRegionTests(unittest.TestCase):
    def test_selects_connected_white_region_from_seed(self) -> None:
        selection = select_light_region_by_seed(_manual_artwork(), (42, 42), _manual_settings())

        self.assertEqual("aplicado", selection.decision)
        self.assertGreater(selection.area, 100)

    def test_removes_selected_white_region(self) -> None:
        img = _manual_artwork()

        result, selections = remove_light_region_by_seed(img, ((42, 42),), _manual_settings())

        self.assertEqual("aplicado", selections[0].decision)
        self.assertLess(_alpha_at(result, 42, 42), _alpha_at(img, 42, 42))

    def test_rejects_seed_on_saturated_pink_art(self) -> None:
        selection = select_light_region_by_seed(_manual_artwork(), (126, 70), _manual_settings())

        self.assertEqual("rechazado", selection.decision)
        self.assertEqual("la_region_no_parece_fondo_blanco", selection.reason)

    def test_rejects_region_exceeding_max_area(self) -> None:
        settings = replace(_manual_settings(), max_area=20)

        selection = select_light_region_by_seed(_manual_artwork(), (42, 42), settings)

        self.assertEqual("rechazado", selection.decision)
        self.assertEqual("area_demasiado_grande", selection.reason)

    def test_rejects_white_region_inside_colored_art_context(self) -> None:
        selection = select_light_region_by_seed(_pink_shine_artwork(), (96, 78), _manual_settings())

        self.assertEqual("rechazado", selection.decision)
        self.assertEqual("region_parece_parte_protegida_del_arte", selection.reason)

    def test_generates_overlay_without_error(self) -> None:
        img = _manual_artwork()
        selection = select_light_region_by_seed(img, (42, 42), _manual_settings())

        overlay = preview_selected_region_overlay(img, [selection])

        self.assertEqual(img.size, overlay.size)

    def test_applies_multiple_manual_seeds(self) -> None:
        img = _manual_artwork()

        result, selections = remove_light_region_by_seed(img, parse_seed_text("42,42; 82,102"), _manual_settings())

        self.assertEqual(2, manual_white_region_report(selections)["applied_count"])
        self.assertEqual(0, _alpha_at(result, 42, 42))
        self.assertEqual(0, _alpha_at(result, 82, 102))

    def test_pipeline_final_alpha_changes_after_manual_apply(self) -> None:
        settings = replace(
            _pipeline_settings(),
            manual_white_enabled=True,
            manual_white_seeds=((42, 42),),
            manual_white_action="apply",
        )

        payload = process_artwork(_manual_artwork(), {"type": "Fondo blanco complejo"}, settings)

        self.assertEqual(0, _alpha_at(payload["image"], 42, 42))
        self.assertIn("manual_white", payload["complex_white_debug"])

    def test_quality_report_uses_manual_updated_result(self) -> None:
        original = _manual_artwork()
        result, _ = remove_light_region_by_seed(original, ((42, 42),), _manual_settings())

        report = quality_report(original, result)

        self.assertGreater(report["removed_background_percent"], 0)

    def test_pipeline_exports_manual_debug_files(self) -> None:
        settings = replace(
            _pipeline_settings(),
            manual_white_enabled=True,
            manual_white_seeds=((42, 42),),
            manual_white_action="apply",
            complex_white_export_debug=True,
        )

        payload = process_artwork(_manual_artwork(), {"type": "Fondo blanco complejo"}, settings)

        with ZipFile(BytesIO(payload["zip"])) as zf:
            names = set(zf.namelist())
            self.assertIn("debug_manual_selected_region.png", names)
            self.assertIn("debug_manual_region_overlay.png", names)
            self.assertIn("debug_manual_region_removed.png", names)
            self.assertIn("debug_manual_region_report.json", names)

    def test_manual_tool_does_not_affect_other_modes(self) -> None:
        settings = replace(
            _pipeline_settings(),
            mode_key="black_bg",
            remove_black=True,
            manual_white_enabled=True,
            manual_white_seeds=((42, 42),),
            manual_white_action="apply",
        )

        payload = process_artwork(_manual_artwork(), {"type": "Fondo negro"}, settings)

        self.assertIsNone(payload["complex_white_debug"])


def _manual_settings() -> ManualWhiteRegionSettings:
    return ManualWhiteRegionSettings(
        tolerance=48,
        luminosity_min=190,
        saturation_max=90,
        max_area=5000,
        connectivity=8,
    )


def _manual_artwork() -> Image.Image:
    img = Image.new("RGBA", (180, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 24, 62, 62), fill=(242, 242, 242, 255))
    draw.rectangle((70, 90, 96, 118), fill=(238, 238, 238, 255))
    draw.rounded_rectangle((108, 42, 160, 106), radius=12, fill=(230, 60, 150, 255), outline=(15, 15, 15, 255), width=4)
    return img


def _pink_shine_artwork() -> Image.Image:
    img = Image.new("RGBA", (180, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((45, 35, 150, 112), radius=20, fill=(230, 60, 150, 255), outline=(15, 15, 15, 255), width=4)
    draw.ellipse((76, 58, 118, 96), fill=(255, 255, 255, 255))
    return img


def _alpha_at(img: Image.Image, x: int, y: int) -> int:
    return int(np.array(img.convert("RGBA"))[y, x, 3])


def _pipeline_settings() -> PipelineSettings:
    return PipelineSettings(
        mode_key="complex_white_bg",
        use_ai=False,
        remove_black=False,
        remove_color=False,
        clean_enabled=False,
        trim=False,
        alpha_cut=58,
        despeckle_area=2,
        edge_contract=0,
        black_threshold=24,
        black_level="normal",
        color_tolerance=44,
        protect_details=True,
        protect_white_details=False,
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
        complex_white_halo_cleanup=False,
        manual_white_tolerance=48,
        manual_white_luminosity=190,
        manual_white_saturation=90,
        manual_white_max_area=5000,
    )
