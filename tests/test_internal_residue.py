from __future__ import annotations

import unittest
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from core.auto_router import autopilot_quality_check
from core.pipeline import PipelineSettings, process_artwork
from core.residue_refine import (
    InternalLightResidueSettings,
    build_light_residue_overlay,
    detect_internal_light_residue_components,
    force_remove_internal_review_components,
    remove_selected_light_residue_components,
)
from ui.sidebar import ProcessingOptions


class InternalLightResidueTests(unittest.TestCase):
    def test_detects_trapped_white_between_black_lines(self) -> None:
        img = _black_line_residue_art()

        components = detect_internal_light_residue_components(img, _internal_settings())

        self.assertGreaterEqual(len(components), 2)
        self.assertTrue(any(item.suggested_action in {"borrar", "revisar"} for item in components))

    def test_halftone_white_trash_is_reduced(self) -> None:
        img = _halftone_residue_art()
        settings = replace(_internal_settings(), auto_remove_high_confidence=True)
        components = detect_internal_light_residue_components(img, settings)

        result = remove_selected_light_residue_components(img, components, settings)

        self.assertLess(_opaque_white_pixels(result), _opaque_white_pixels(img))

    def test_preserves_large_white_shine_on_pink_letter(self) -> None:
        img = _pink_letter_with_shine()
        settings = _internal_settings()
        components = detect_internal_light_residue_components(img, settings)

        result = remove_selected_light_residue_components(img, components, settings)

        self.assertEqual(255, result.getpixel((150, 130))[3])
        self.assertEqual((255, 255, 255), result.getpixel((150, 130))[:3])
        self.assertFalse(any(item.suggested_action == "borrar" for item in components))

    def test_preserves_small_eye_white_inside_character(self) -> None:
        img = _character_eye_art()
        settings = replace(_internal_settings(), auto_remove_high_confidence=True)
        components = detect_internal_light_residue_components(img, settings)

        result = remove_selected_light_residue_components(img, components, settings)

        self.assertEqual(255, result.getpixel((135, 120))[3])
        self.assertEqual((255, 255, 255), result.getpixel((135, 120))[:3])

    def test_removes_manual_internal_component_id(self) -> None:
        img = _black_line_residue_art()
        settings = _internal_settings()
        components = detect_internal_light_residue_components(img, settings)
        target = next(item for item in components if item.suggested_action in {"borrar", "revisar", "conservar"})

        manual = replace(settings, manual_remove_ids=(target.id,))
        result = remove_selected_light_residue_components(img, components, manual)

        x1, y1, x2, y2 = target.bbox
        self.assertEqual(0, result.getpixel(((x1 + x2) // 2, (y1 + y2) // 2))[3])

    def test_green_preview_overlay_generates_without_error(self) -> None:
        img = _black_line_residue_art()
        settings = _internal_settings()
        components = detect_internal_light_residue_components(img, settings)

        overlay = build_light_residue_overlay(img, components, settings)

        self.assertEqual(img.size, overlay.size)

    def test_autopilot_qa_detects_internal_residue_and_avoids_green(self) -> None:
        qa = autopilot_quality_check(_black_line_residue_art())

        self.assertNotEqual("green", qa["traffic_light"])
        self.assertGreater(qa["internal_light_residue_components"], 0)
        self.assertIn("Refinar residuos blancos internos", qa["suggested_next_step"])

    def test_pipeline_exports_internal_residue_debug_files(self) -> None:
        settings = replace(
            _pipeline_settings(),
            internal_residue_enabled=True,
            internal_residue_auto_remove=True,
            complex_white_export_debug=True,
        )

        payload = process_artwork(_white_background_with_internal_residue(), {"type": "Fondo blanco complejo"}, settings)

        self.assertIn("internal_residue", payload["complex_white_debug"])
        with ZipFile(BytesIO(payload["zip"])) as zf:
            names = set(zf.namelist())
            self.assertIn("debug_internal_light_residue_mask.png", names)
            self.assertIn("debug_internal_light_residue_overlay.png", names)
            self.assertIn("debug_preview_green.png", names)
            self.assertIn("debug_internal_residue_report.json", names)

    def test_debug_reports_when_internal_refinement_removes_no_pixels(self) -> None:
        settings = replace(_pipeline_settings(), internal_residue_enabled=True, internal_residue_auto_remove=True)

        payload = process_artwork(_pink_letter_with_shine(), {"type": "Fondo blanco complejo"}, settings)
        stats = payload["complex_white_debug"]["internal_residue"]["stats"]

        self.assertEqual(0, stats["internal_removed_pixels"])
        self.assertTrue(stats["internal_no_pixels_removed"])

    def test_final_preview_payload_uses_refined_result(self) -> None:
        settings = replace(_pipeline_settings(), internal_residue_enabled=True, internal_residue_auto_remove=True)

        payload = process_artwork(_black_line_residue_art(), {"type": "Fondo blanco complejo"}, settings)

        self.assertLess(_opaque_white_pixels(payload["image"]), _opaque_white_pixels(_black_line_residue_art()))
        self.assertGreater(payload["complex_white_debug"]["internal_residue"]["stats"]["internal_removed_pixels"], 0)

    def test_overlay_distinguishes_removed_review_and_protected_components(self) -> None:
        img = _mixed_status_art()
        settings = replace(_internal_settings(), auto_remove_high_confidence=True, dark_neighbor_threshold=50)
        components = detect_internal_light_residue_components(img, settings)
        components = [
            replace(components[0], suggested_action="borrar"),
            replace(components[1], suggested_action="revisar"),
            replace(components[2], suggested_action="conservar"),
        ]

        overlay = build_light_residue_overlay(img, components, settings)
        colors = {tuple(pixel) for pixel in np.array(overlay.convert("RGBA")).reshape(-1, 4)}

        self.assertTrue(any(red > 220 and green < 130 and blue < 130 and alpha > 0 for red, green, blue, alpha in colors))
        self.assertTrue(any(red > 220 and green > 180 and blue < 130 and alpha > 0 for red, green, blue, alpha in colors))
        self.assertTrue(any(red < 140 and green > 100 and blue > 220 and alpha > 0 for red, green, blue, alpha in colors))

    def test_forced_debug_removes_borrar_and_review_components(self) -> None:
        img = _mixed_status_art()
        settings = _internal_settings()
        components = detect_internal_light_residue_components(img, settings)

        forced = force_remove_internal_review_components(img, components, settings)

        self.assertLess(_opaque_white_pixels(forced), _opaque_white_pixels(img))

    def test_sidebar_internal_residue_fields_match_pipeline_settings(self) -> None:
        option_fields = ProcessingOptions.__dataclass_fields__
        pipeline_fields = PipelineSettings.__dataclass_fields__

        for name in [
            "internal_residue_enabled",
            "internal_residue_min_area",
            "internal_residue_max_area",
            "internal_residue_dark_sensitivity",
            "internal_residue_luminosity",
            "internal_residue_saturation",
            "internal_residue_auto_remove",
            "internal_residue_manual_ids",
        ]:
            self.assertIn(name, option_fields)
            self.assertIn(name, pipeline_fields)


def _internal_settings() -> InternalLightResidueSettings:
    return InternalLightResidueSettings(
        min_area=4,
        max_area=900,
        dark_neighbor_threshold=28,
        luminosity_threshold=218,
        saturation_threshold=58,
        auto_remove_high_confidence=False,
    )


def _black_line_residue_art() -> Image.Image:
    img = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for idx in range(10):
        y = 30 + idx * 16
        draw.line((30, y, 230, y + 18), fill=(15, 15, 15, 255), width=8)
    for box in [(92, 65, 106, 78), (142, 108, 158, 122), (186, 145, 198, 158)]:
        draw.rectangle(box, fill=(245, 245, 245, 255))
    return img


def _halftone_residue_art() -> Image.Image:
    img = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((35, 35, 225, 185), fill=(12, 12, 12, 255))
    for x in range(55, 210, 24):
        for y in range(55, 170, 24):
            draw.ellipse((x, y, x + 5, y + 5), fill=(248, 248, 248, 255))
    return img


def _pink_letter_with_shine() -> Image.Image:
    img = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((55, 55, 215, 170), radius=24, fill=(230, 55, 150, 255), outline=(15, 15, 15, 255), width=5)
    draw.ellipse((100, 92, 190, 150), fill=(255, 255, 255, 255))
    return img


def _character_eye_art() -> Image.Image:
    img = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((55, 35, 210, 190), fill=(230, 70, 145, 255), outline=(15, 15, 15, 255), width=6)
    draw.ellipse((118, 100, 154, 138), fill=(15, 15, 15, 255))
    draw.ellipse((128, 112, 143, 128), fill=(255, 255, 255, 255))
    return img


def _mixed_status_art() -> Image.Image:
    img = _black_line_residue_art()
    draw = ImageDraw.Draw(img)
    draw.rectangle((25, 175, 45, 190), fill=(250, 250, 250, 255))
    draw.rectangle((28, 178, 45, 182), fill=(15, 15, 15, 255))
    draw.rounded_rectangle((140, 160, 240, 210), radius=12, fill=(230, 55, 150, 255), outline=(15, 15, 15, 255), width=4)
    draw.ellipse((168, 174, 226, 202), fill=(255, 255, 255, 255))
    return img


def _white_background_with_internal_residue() -> Image.Image:
    img = Image.new("RGBA", (260, 220), (255, 255, 255, 255))
    img.alpha_composite(_black_line_residue_art())
    return img


def _opaque_white_pixels(img: Image.Image) -> int:
    count = 0
    for _, _, red, green, blue, alpha in _iter_pixels(img):
        if alpha > 220 and red > 235 and green > 235 and blue > 235:
            count += 1
    return count


def _iter_pixels(img: Image.Image):
    rgba = img.convert("RGBA")
    for y in range(rgba.height):
        for x in range(rgba.width):
            red, green, blue, alpha = rgba.getpixel((x, y))
            yield x, y, red, green, blue, alpha


def _pipeline_settings() -> PipelineSettings:
    return PipelineSettings(
        mode_key="complex_white_bg",
        use_ai=False,
        remove_black=False,
        remove_color=False,
        clean_enabled=True,
        trim=False,
        alpha_cut=58,
        despeckle_area=2,
        edge_contract=0,
        black_threshold=24,
        black_level="normal",
        color_tolerance=44,
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
        complex_white_export_debug=True,
        internal_residue_min_area=4,
        internal_residue_max_area=900,
        internal_residue_dark_sensitivity=28,
        internal_residue_luminosity=218,
        internal_residue_saturation=58,
    )
