from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from core.production_dtf import clean_for_production, production_dtf_preset, production_dtf_preset_names
from core.quality_report import quality_gate
from core.residue_refine import detect_internal_light_residue_components
from ui.sidebar import ProcessingOptions
from core.pipeline import PipelineSettings, process_artwork


class ProductionDTFTests(unittest.TestCase):
    def test_production_presets_exist_and_are_valid(self) -> None:
        names = production_dtf_preset_names()
        self.assertEqual(["Conservador", "Balanceado", "Agresivo DTF", "Solo detectar"], names)
        for name in names:
            preset = production_dtf_preset(name)
            self.assertGreater(preset.complex_white.white_tolerance, 0)
            self.assertGreaterEqual(preset.residue.min_area, 1)
            self.assertGreaterEqual(preset.internal.min_area, 1)

    def test_clean_for_production_removes_high_confidence_residue(self) -> None:
        img = _black_line_residue_art()
        result = clean_for_production(img, "Agresivo DTF")

        self.assertLess(_opaque_white_pixels(result.image), _opaque_white_pixels(img))
        self.assertGreater(result.stats["production_removed_pixels"], 0)
        self.assertGreaterEqual(result.stats["residue_components_removed"], 1)

    def test_solo_detectar_does_not_modify_alpha(self) -> None:
        img = _black_line_residue_art()
        result = clean_for_production(img, "Solo detectar")

        self.assertEqual(_alpha_array(img).tolist(), _alpha_array(result.image).tolist())
        self.assertTrue(result.stats["detect_only"])

    def test_production_preserves_large_white_shine(self) -> None:
        img = _pink_letter_with_shine()
        result = clean_for_production(img, "Agresivo DTF")

        self.assertEqual(255, result.image.getpixel((150, 130))[3])
        self.assertEqual((255, 255, 255), result.image.getpixel((150, 130))[:3])

    def test_quality_gate_blocks_pending_residue(self) -> None:
        gate = quality_gate(
            dtf_score=92,
            internal_residue_stats={
                "internal_components_detected": 12,
                "internal_components_removed": 4,
                "internal_components_review": 2,
                "internal_protection_reasons": {"7": "area_grande"},
            },
        )

        self.assertNotEqual("Lista para imprimir", gate["status"])
        self.assertTrue(gate["blocked"])

    def test_pipeline_settings_and_options_have_production_fields(self) -> None:
        for name in ["production_dtf_enabled", "production_dtf_preset"]:
            self.assertIn(name, PipelineSettings.__dataclass_fields__)
        for name in ["production_dtf_preset", "production_dtf_action", "manual_white_quick_apply"]:
            self.assertIn(name, ProcessingOptions.__dataclass_fields__)

    def test_pipeline_production_cleanup_exports_debug(self) -> None:
        settings = replace(_pipeline_settings(), production_dtf_enabled=True, production_dtf_preset="Agresivo DTF", complex_white_export_debug=True)
        payload = process_artwork(_white_background_with_residue(), {"type": "Fondo blanco complejo"}, settings)

        debug = payload["complex_white_debug"]
        self.assertIn("production_dtf", debug)
        self.assertIn("production_residue", debug)
        self.assertIn("production_internal_residue", debug)
        self.assertLess(_opaque_white_pixels(payload["image"]), _opaque_white_pixels(_white_background_with_residue()))


def _alpha_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA").getchannel("A"))


def _opaque_white_pixels(img: Image.Image) -> int:
    arr = np.array(img.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    return int(np.count_nonzero((alpha > 220) & (rgb[:, :, 0] > 230) & (rgb[:, :, 1] > 230) & (rgb[:, :, 2] > 230)))


def _black_line_residue_art() -> Image.Image:
    img = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for idx in range(8):
        y = 32 + idx * 18
        draw.line((35, y, 225, y + 20), fill=(14, 14, 14, 255), width=9)
    for box in [(82, 60, 100, 76), (135, 100, 156, 119), (180, 140, 200, 160)]:
        draw.rectangle(box, fill=(248, 248, 248, 255))
    return img


def _pink_letter_with_shine() -> Image.Image:
    img = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((55, 55, 215, 170), radius=24, fill=(230, 55, 150, 255), outline=(15, 15, 15, 255), width=6)
    draw.ellipse((100, 92, 190, 150), fill=(255, 255, 255, 255))
    return img


def _white_background_with_residue() -> Image.Image:
    img = Image.new("RGB", (260, 220), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle((45, 40, 220, 180), fill=(15, 15, 15))
    draw.rectangle((105, 85, 125, 105), fill=(248, 248, 248))
    draw.rectangle((160, 125, 185, 145), fill=(248, 248, 248))
    return img.convert("RGBA")


def _pipeline_settings() -> PipelineSettings:
    return PipelineSettings(
        mode_key="complex_white_bg",
        use_ai=False,
        remove_black=False,
        remove_color=True,
        clean_enabled=False,
        trim=False,
        alpha_cut=70,
        despeckle_area=9,
        edge_contract=0,
        black_threshold=20,
        black_level="normal",
        color_tolerance=32,
        protect_details=True,
        protect_white_details=False,
        white_protection_level="normal",
        fine_detail_level="normal",
        safe_mode=False,
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
        max_ai_side=1600,
        upscale=1,
        dpi=300,
        width_cm=0.0,
        height_cm=0.0,
    )


if __name__ == "__main__":
    unittest.main()
