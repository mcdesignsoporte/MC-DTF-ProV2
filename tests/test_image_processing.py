import unittest
from dataclasses import replace
from time import perf_counter
from io import BytesIO
from zipfile import ZipFile
import json

import numpy as np
from PIL import Image

from core.black_remove import remove_black_background
from core.background_remove import (
    cleanup_light_background_residue,
    cleanup_light_edge_matte,
    dominant_background_color,
    edge_light_residue_score,
    light_residue_percent,
    remove_dominant_background,
)
from core.background import apply_ai_alpha_to_original
from core.clean import clean_alpha, contract_edge, despeckle, protect_fine_details, safe_despeckle
from core.detector import detect
from core.halftone import make_halftone
from core.image_io import image_to_png_bytes, make_zip_bytes
from core.pipeline import PipelineSettings, process_artwork
from core.presets import available_mode_keys, preset_for_mode
from ui.detection import detection_value
from core.preview import alpha_preview, before_after_preview, checkerboard, composite_preview, preview_thumbnail
from core.quality import alpha_histogram, evaluate_dtf_quality
from core.export import build_export_package
from core.white_protection import build_protection_mask, protect_white_regions
from core.white_complex import ComplexWhiteSettings, compose_on_solid, remove_complex_white_background
from core.white_complex import COMPLEX_WHITE_PRESETS, complex_white_preset
from core.residue_refine import (
    ResidueRefineSettings,
    apply_residue_component_removal,
    detect_light_residue_components,
    overlay_residue_components,
    residue_component_report_json,
)
from core.artwork_mask import build_artwork_mask
from core.background_confirm import confirm_background_mask
from core.non_destructive_clean import non_destructive_clean, estimate_art_loss_risk, restore_artwork_pixels


class ImageProcessingTests(unittest.TestCase):
    def test_halftone_keeps_transparent_spaces_between_dots(self):
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 255))

        result = make_halftone(img, dot_size=8, angle=0)

        alpha = np.array(result.getchannel("A"))
        self.assertIn(0, alpha)
        self.assertIn(255, alpha)

    def test_ai_alpha_is_restored_to_original_size(self):
        original = Image.new("RGBA", (400, 200), (255, 0, 0, 255))
        ai_result = Image.new("RGBA", (160, 80), (255, 0, 0, 0))
        ai_result.putalpha(Image.new("L", ai_result.size, 128))

        result = apply_ai_alpha_to_original(original, ai_result)

        self.assertEqual((400, 200), result.size)
        self.assertEqual((255, 0, 0), result.getpixel((200, 100))[:3])
        self.assertEqual(128, result.getpixel((200, 100))[3])

    def test_preview_uses_viewport_for_large_images(self):
        img = Image.new("RGBA", (3543, 4724), (0, 0, 0, 0))

        preview = composite_preview(img, "Transparente", max_side=1000)

        self.assertEqual((1080, 760), preview.size)

    def test_preview_thumbnail_fits_viewport(self):
        img = Image.new("RGBA", (3543, 4724), (255, 0, 0, 255))

        thumb = preview_thumbnail(img, (800, 600), padding=30)

        self.assertLessEqual(thumb.width, 740)
        self.assertLessEqual(thumb.height, 540)

    def test_checkerboard_is_fast_rgba_canvas(self):
        board = checkerboard((320, 180), tile=16)

        self.assertEqual((320, 180), board.size)
        self.assertEqual("RGBA", board.mode)

    def test_alpha_preview_uses_fixed_viewport(self):
        img = Image.new("RGBA", (1200, 900), (255, 0, 0, 128))

        preview = alpha_preview(img)

        self.assertEqual((1080, 760), preview.size)

    def test_detector_recommends_dark_design_for_black_background(self):
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
        for x in range(35, 65):
            for y in range(35, 65):
                img.putpixel((x, y), (255, 255, 255, 255))

        result = detect(img)

        self.assertEqual("black_bg", result["recommended_mode"])
        self.assertEqual("negro", result["background"])
        self.assertIn("dominant_color", result)
        self.assertGreater(result["confidence"], 70)

    def test_black_removal_keeps_bright_letter_details(self):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        for x in range(10, 22):
            for y in range(8, 24):
                img.putpixel((x, y), (255, 255, 255, 255))

        result = remove_black_background(img, threshold=20, protect_details=True)

        self.assertEqual(0, result.getpixel((0, 0))[3])
        self.assertEqual(255, result.getpixel((16, 16))[3])

    def test_color_background_removal_removes_green_edges(self):
        img = Image.new("RGBA", (40, 40), (0, 180, 0, 255))
        for x in range(12, 28):
            for y in range(12, 28):
                img.putpixel((x, y), (255, 255, 255, 255))

        result = remove_dominant_background(img, tolerance=45)

        self.assertLess(abs(dominant_background_color(img)[1] - 180), 8)
        self.assertEqual(0, result.getpixel((0, 0))[3])
        self.assertGreater(result.getpixel((20, 20))[3], 180)

    def test_white_letters_remain_after_white_background_removal(self):
        img = Image.new("RGBA", (90, 50), (255, 255, 255, 255))
        for x in range(20, 70):
            for y in range(15, 35):
                img.putpixel((x, y), (220, 0, 0, 255))
        for x in range(30, 60):
            for y in range(22, 28):
                img.putpixel((x, y), (255, 255, 255, 255))

        cleaned = remove_dominant_background(img, tolerance=34, protect_details=True)
        protected, mask, stats = protect_white_regions(img, cleaned, level="maxima")

        self.assertEqual(0, protected.getpixel((0, 0))[3])
        self.assertEqual(255, protected.getpixel((42, 24))[3])
        self.assertTrue(mask[24, 42])
        self.assertGreater(stats.white_protected, 0)

    def test_white_eyes_and_reflections_remain(self):
        img = Image.new("RGBA", (80, 80), (255, 255, 255, 255))
        for x in range(18, 62):
            for y in range(18, 62):
                img.putpixel((x, y), (50, 130, 220, 255))
        for x in range(30, 38):
            for y in range(30, 38):
                img.putpixel((x, y), (255, 255, 255, 255))
        for x in range(48, 54):
            for y in range(24, 30):
                img.putpixel((x, y), (250, 250, 250, 255))

        cleaned = remove_dominant_background(img, tolerance=34, protect_details=True)
        protected, mask, stats = protect_white_regions(img, cleaned, level="maxima")

        self.assertEqual(255, protected.getpixel((34, 34))[3])
        self.assertEqual(255, protected.getpixel((50, 26))[3])
        self.assertTrue(mask[34, 34])
        self.assertFalse(stats.possible_detail_loss)

    def test_white_background_is_removed_but_internal_white_is_protected(self):
        img = Image.new("RGBA", (70, 70), (255, 255, 255, 255))
        for x in range(18, 52):
            for y in range(18, 52):
                img.putpixel((x, y), (0, 0, 0, 255))
        for x in range(30, 40):
            for y in range(30, 40):
                img.putpixel((x, y), (255, 255, 255, 255))

        mask = build_protection_mask(img, level="maxima")
        cleaned = remove_dominant_background(img, tolerance=34, protect_details=True)
        protected, _, _ = protect_white_regions(img, cleaned, level="maxima")

        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[35, 35])
        self.assertEqual(0, protected.getpixel((0, 0))[3])
        self.assertEqual(255, protected.getpixel((35, 35))[3])

    def test_white_protection_runs_quickly_on_medium_image(self):
        img = Image.new("RGBA", (600, 600), (255, 255, 255, 255))
        img.alpha_composite(Image.new("RGBA", (360, 360), (20, 80, 180, 255)), (120, 120))
        img.alpha_composite(Image.new("RGBA", (80, 40), (255, 255, 255, 255)), (260, 280))

        started = perf_counter()
        mask = build_protection_mask(img, level="maxima")
        elapsed = perf_counter() - started

        self.assertTrue(mask[300, 300])
        self.assertLess(elapsed, 1.0)

    def test_clean_alpha_removes_small_trash(self):
        img = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
        img.putpixel((1, 1), (255, 0, 0, 255))
        for x in range(5, 9):
            for y in range(5, 9):
                img.putpixel((x, y), (255, 0, 0, 255))

        result = clean_alpha(img, alpha_cut=70, despeckle_area=4)

        self.assertEqual(0, result.getpixel((1, 1))[3])
        self.assertEqual(255, result.getpixel((6, 6))[3])

    def test_safe_despeckle_keeps_small_splashes_near_design(self):
        img = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
        img.alpha_composite(Image.new("RGBA", (30, 30), (220, 30, 30, 255)), (25, 25))
        for point in [(20, 24), (21, 26), (58, 30)]:
            img.putpixel(point, (255, 210, 40, 255))
        arr = np.array(img)
        alpha, mask, stats = safe_despeckle(arr, arr[:, :, 3], min_area=8, level="maxima")

        self.assertEqual(255, alpha[24, 20])
        self.assertEqual(255, alpha[26, 21])
        self.assertTrue(mask[24, 20])
        self.assertEqual(0, stats.components_removed)

    def test_safe_despeckle_removes_isolated_noise_far_from_design(self):
        img = Image.new("RGBA", (90, 90), (0, 0, 0, 0))
        img.alpha_composite(Image.new("RGBA", (36, 36), (50, 100, 230, 255)), (28, 28))
        img.putpixel((2, 2), (80, 80, 80, 255))
        arr = np.array(img)
        alpha, _, stats = safe_despeckle(arr, arr[:, :, 3], min_area=8, level="normal")

        self.assertEqual(0, alpha[2, 2])
        self.assertGreaterEqual(stats.components_removed, 1)

    def test_fine_detail_mask_keeps_thin_lines_and_decorative_dots(self):
        img = Image.new("RGBA", (90, 90), (0, 0, 0, 0))
        for x in range(20, 70):
            img.putpixel((x, 45), (255, 255, 255, 255))
        img.alpha_composite(Image.new("RGBA", (24, 24), (0, 0, 0, 255)), (33, 33))
        img.putpixel((60, 35), (240, 240, 255, 255))
        arr = np.array(img)

        mask = protect_fine_details(arr, arr[:, :, 3], level="maxima")

        self.assertTrue(mask[45, 22])
        self.assertTrue(mask[35, 60])

    def test_safe_despeckle_keeps_letter_texture_and_contours(self):
        img = Image.new("RGBA", (100, 70), (0, 0, 0, 0))
        for x in range(20, 80):
            for y in range(25, 45):
                img.putpixel((x, y), (255, 255, 255, 255))
        for x in range(28, 72, 4):
            img.putpixel((x, 35), (30, 30, 30, 255))
        arr = np.array(img)
        alpha, mask, _ = safe_despeckle(arr, arr[:, :, 3], min_area=12, level="maxima")

        self.assertEqual(255, alpha[35, 28])
        self.assertTrue(mask[35, 28])

    def test_non_destructive_keeps_pink_car_body_and_removes_white_background(self):
        img = _synthetic_car()

        result = non_destructive_clean(img, min_area=6, background_tolerance=48, safe_mode=True)

        self.assertEqual(255, result.image.getpixel((48, 42))[3])
        self.assertEqual(255, result.image.getpixel((38, 56))[3])
        self.assertEqual(0, result.image.getpixel((2, 2))[3])
        self.assertTrue(result.artwork_mask[42, 48])
        self.assertGreater(result.stats["background_removed"], 0)

    def test_non_destructive_keeps_wheel_white_details(self):
        img = _synthetic_car()

        result = non_destructive_clean(img, min_area=8, background_tolerance=48, safe_mode=True)

        self.assertEqual(255, result.image.getpixel((35, 55))[3])
        self.assertEqual(255, result.image.getpixel((75, 55))[3])

    def test_background_confirm_only_removes_edge_connected_uniform_area(self):
        img = _synthetic_car()
        artwork = build_artwork_mask(img)
        background = confirm_background_mask(img, artwork.artwork_mask, artwork.main_bbox, tolerance=48)

        self.assertTrue(background[0, 0])
        self.assertFalse(background[42, 48])
        self.assertFalse(background[55, 35])

    def test_non_destructive_keeps_white_letters_on_black_background(self):
        img = Image.new("RGBA", (80, 50), (0, 0, 0, 255))
        for x in range(20, 60):
            for y in range(18, 32):
                img.putpixel((x, y), (255, 255, 255, 255))

        result = non_destructive_clean(img, min_area=6, background_tolerance=35, safe_mode=True)

        self.assertEqual(255, result.image.getpixel((30, 24))[3])

    def test_non_destructive_keeps_near_splashes_and_removes_far_noise(self):
        img = _synthetic_car()
        img.putpixel((82, 36), (255, 0, 160, 255))
        img.putpixel((4, 78), (80, 80, 80, 255))

        result = non_destructive_clean(img, min_area=8, background_tolerance=48, safe_mode=True)

        self.assertEqual(255, result.image.getpixel((82, 36))[3])
        self.assertEqual(0, result.image.getpixel((4, 78))[3])

    def test_art_loss_risk_detects_and_restores_artwork_pixels(self):
        img = _synthetic_car()
        artwork = build_artwork_mask(img)
        damaged = img.copy()
        damaged.putpixel((48, 42), (255, 80, 180, 0))

        risk = estimate_art_loss_risk(img, damaged, artwork.artwork_mask, artwork.main_bbox)
        restored, mask = restore_artwork_pixels(img, damaged, artwork.artwork_mask)

        self.assertTrue(risk["risk_detected"])
        self.assertTrue(mask[42, 48])
        self.assertEqual(255, restored.getpixel((48, 42))[3])

    def test_zip_contains_generated_assets(self):
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        png_payload = image_to_png_bytes(img)
        zip_payload = make_zip_bytes({"sample.png": png_payload})

        with ZipFile(BytesIO(zip_payload)) as zf:
            self.assertEqual(["sample.png"], zf.namelist())
            self.assertGreater(len(zf.read("sample.png")), 0)

    def test_export_preserves_dimensions_and_adds_assets(self):
        img = Image.new("RGBA", (18, 12), (0, 0, 0, 0))

        package = build_export_package(img, dpi=300, prefix="asset")

        self.assertIn("png", package)
        self.assertIn("pdf", package)
        self.assertIn("zip", package)
        reopened = Image.open(BytesIO(package["png"]))
        self.assertEqual((18, 12), reopened.size)

    def test_detector_recommends_transparent_png_without_ai(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
        img.alpha_composite(Image.new("RGBA", (12, 12), (255, 0, 0, 255)), (10, 10))

        result = detect(img)

        self.assertEqual("transparent_png", result["recommended_mode"])
        self.assertFalse(result["use_ai"])

    def test_detector_routes_uniform_light_background_to_color_removal(self):
        img = _light_background_artwork()

        result = detect(img)

        self.assertEqual("Fondo de color", result["type"])
        self.assertEqual("color_bg", result["recommended_mode"])
        self.assertNotEqual("professional_safe", result["recommended_mode"])
        self.assertEqual(0.0, result["transparency_percent"])

    def test_automatic_mode_removes_uniform_light_background(self):
        img = _light_background_artwork()
        detection = detect(img)

        result = process_artwork(img, detection, _auto_settings())["image"]
        alpha = np.array(result.getchannel("A"))

        self.assertGreater(np.mean(alpha < 250) * 100, 20)
        self.assertLess(alpha[0, 0], 10)

    def test_automatic_mode_keeps_internal_white_details(self):
        img = _light_background_artwork()
        detection = detect(img)

        result = process_artwork(img, detection, _auto_settings())["image"]

        self.assertEqual(255, result.getpixel((42, 34))[3])
        self.assertGreater(result.getpixel((42, 34))[0], 240)

    def test_light_residue_cleanup_removes_exterior_connected_plate(self):
        img = _residue_after_first_pass()
        before = light_residue_percent(img, background_color=(240, 240, 240))

        result = cleanup_light_background_residue(img, background_color=(240, 240, 240), tolerance=70)
        after = light_residue_percent(result, background_color=(240, 240, 240))

        self.assertGreater(before, 5)
        self.assertLess(after, 1)
        self.assertEqual(0, result.getpixel((8, 20))[3])

    def test_light_residue_cleanup_preserves_enclosed_white_artwork(self):
        img = _residue_after_first_pass()

        result = cleanup_light_background_residue(img, background_color=(240, 240, 240), tolerance=70)

        self.assertEqual(255, result.getpixel((45, 35))[3])
        self.assertEqual((255, 255, 255), result.getpixel((45, 35))[:3])

    def test_light_residue_cleanup_keeps_dark_outline_on_irregular_light_background(self):
        img = _outlined_artwork_with_residue()

        result = cleanup_light_background_residue(img, background_color=(240, 240, 240), tolerance=72)

        self.assertEqual(0, result.getpixel((4, 8))[3])
        self.assertEqual(255, result.getpixel((25, 20))[3])
        self.assertEqual((20, 20, 20), result.getpixel((25, 20))[:3])

    def test_automatic_mode_reduces_complex_light_background_residue(self):
        img = _complex_light_background_artwork()
        detection = detect(img)
        first_pass = remove_dominant_background(img, tolerance=44, protect_details=True)
        before = light_residue_percent(first_pass, background_color=(240, 240, 240), tolerance=70)

        result = process_artwork(img, detection, _auto_settings())["image"]
        after = light_residue_percent(result, background_color=(240, 240, 240), tolerance=70)
        alpha = np.array(result.getchannel("A"))

        self.assertGreater(np.mean(alpha < 250) * 100, 20)
        self.assertLess(after, before)
        self.assertEqual(255, result.getpixel((45, 35))[3])

    def test_light_edge_matte_cleanup_reduces_halo_on_black_preview(self):
        img = _white_matte_edge_image()
        before = edge_light_residue_score(img)

        result = cleanup_light_edge_matte(img, background_color=(240, 240, 240), tolerance=76)
        after = edge_light_residue_score(result)

        self.assertGreater(before, 10)
        self.assertLess(after, before * 0.45)

    def test_light_edge_matte_cleanup_preserves_internal_white_detail(self):
        img = _white_matte_edge_image()

        result = cleanup_light_edge_matte(img, background_color=(240, 240, 240), tolerance=76)

        self.assertEqual(255, result.getpixel((45, 35))[3])
        self.assertEqual((255, 255, 255), result.getpixel((45, 35))[:3])

    def test_light_edge_matte_cleanup_reduces_semitransparent_white_rgb(self):
        img = _white_matte_edge_image()

        result = cleanup_light_edge_matte(img, background_color=(240, 240, 240), tolerance=76)

        self.assertLess(result.getpixel((22, 28))[3], img.getpixel((22, 28))[3])

    def test_light_edge_matte_cleanup_does_not_degrade_clean_transparent_png(self):
        img = Image.new("RGBA", (42, 42), (0, 0, 0, 0))
        img.alpha_composite(Image.new("RGBA", (20, 20), (230, 30, 120, 255)), (11, 11))

        result = cleanup_light_edge_matte(img, background_color=(240, 240, 240), tolerance=76)

        self.assertTrue(np.array_equal(np.array(img), np.array(result)))

    def test_light_edge_matte_cleanup_does_not_alter_dark_artwork(self):
        img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        img.alpha_composite(Image.new("RGBA", (26, 26), (15, 15, 15, 255)), (11, 11))

        result = cleanup_light_edge_matte(img, background_color=(240, 240, 240), tolerance=76)

        self.assertTrue(np.array_equal(np.array(img), np.array(result)))

    def test_complex_white_flow_removes_exterior_background(self):
        img = _complex_character_artwork()

        result = remove_complex_white_background(img, _complex_settings())

        self.assertEqual(0, result.image.getpixel((0, 0))[3])
        self.assertGreater(result.stats["background_removed"], 1000)
        self.assertEqual(result.image.size, result.alpha_mask.size)
        self.assertEqual(result.image.size, result.background_mask.size)

    def test_complex_white_flow_preserves_internal_white_and_face_detail(self):
        img = _complex_character_artwork()

        result = remove_complex_white_background(img, _complex_settings()).image

        self.assertEqual(255, result.getpixel((50, 42))[3])
        self.assertEqual((255, 255, 255), result.getpixel((50, 42))[:3])
        self.assertEqual(255, result.getpixel((50, 29))[3])
        self.assertGreater(result.getpixel((50, 29))[0], 210)

    def test_complex_white_flow_reduces_halo_on_black_review(self):
        img = _complex_character_artwork()
        rough = remove_complex_white_background(img, _complex_settings(halo_cleanup=False)).image
        before = edge_light_residue_score(rough)

        result = remove_complex_white_background(img, _complex_settings()).image
        after = edge_light_residue_score(result)

        self.assertLess(after, before)
        self.assertEqual((0, 0, 0), compose_on_solid(result, (0, 0, 0)).getpixel((0, 0))[:3])

    def test_complex_white_flow_preserves_black_contours_and_fine_features(self):
        img = _complex_character_artwork()

        result = remove_complex_white_background(img, _complex_settings()).image

        self.assertEqual(255, result.getpixel((25, 32))[3])
        self.assertLess(result.getpixel((25, 32))[0], 40)
        self.assertEqual(255, result.getpixel((43, 37))[3])
        self.assertLess(result.getpixel((43, 37))[0], 40)

    def test_pipeline_complex_white_exports_debug_when_enabled(self):
        img = _complex_character_artwork()

        payload = process_artwork(img, {"type": "Fondo blanco complejo"}, _complex_pipeline_settings())

        self.assertIsNotNone(payload["complex_white_debug"])
        with ZipFile(BytesIO(payload["zip"])) as zf:
            names = set(zf.namelist())
            self.assertIn("debug_alpha_mask.png", names)
            self.assertIn("debug_preview_black.png", names)
            self.assertIn("debug_preview_red.png", names)

    def test_complex_white_presets_exist(self):
        self.assertEqual({"Conservador", "Balanceado", "Agresivo DTF"}, set(COMPLEX_WHITE_PRESETS))

    def test_complex_white_presets_generate_valid_settings(self):
        for name in COMPLEX_WHITE_PRESETS:
            settings = complex_white_preset(name)
            self.assertIsInstance(settings, ComplexWhiteSettings)
            self.assertGreaterEqual(settings.white_tolerance, 20)
            self.assertLessEqual(settings.white_tolerance, 110)
            self.assertGreaterEqual(settings.luminosity_threshold, 170)
            self.assertLessEqual(settings.luminosity_threshold, 255)
            self.assertGreaterEqual(settings.saturation_threshold, 0)
            self.assertLessEqual(settings.saturation_threshold, 100)
            self.assertIn(settings.mask_offset, {-2, -1, 0, 1, 2})
            self.assertIn(settings.alpha_smoothing, {0, 1, 2})

    def test_detects_white_residue_components_after_cutout(self):
        img = _residue_refine_artwork()

        components = detect_light_residue_components(img, _residue_settings())

        self.assertGreaterEqual(len(components), 3)
        self.assertTrue(any(item.classification == "borrar" for item in components))

    def test_residue_refine_preserves_internal_white_with_dark_contour(self):
        img = _residue_refine_artwork()
        settings = _residue_settings()
        components = detect_light_residue_components(img, settings)

        result = apply_residue_component_removal(img, components, settings)

        self.assertEqual(255, result.getpixel((45, 35))[3])
        self.assertEqual((255, 255, 255), result.getpixel((45, 35))[:3])

    def test_residue_refine_removes_component_connected_to_transparency(self):
        img = _residue_refine_artwork()
        settings = _residue_settings()
        components = detect_light_residue_components(img, settings)

        result = apply_residue_component_removal(img, components, settings)

        self.assertEqual(0, result.getpixel((17, 44))[3])

    def test_residue_refine_removes_manual_component_id(self):
        img = _residue_refine_artwork()
        settings = _residue_settings(remove_connected=False)
        components = detect_light_residue_components(img, settings)
        ambiguous = next(item for item in components if item.classification == "arte interno")

        manual = replace(settings, manual_remove_ids=(ambiguous.id,))
        result = apply_residue_component_removal(img, components, manual)

        x1, y1, x2, y2 = ambiguous.bbox
        self.assertEqual(0, result.getpixel(((x1 + x2) // 2, (y1 + y2) // 2))[3])

    def test_residue_refine_keeps_ambiguous_components_by_default(self):
        img = _residue_refine_artwork()
        settings = _residue_settings(remove_connected=False)
        components = detect_light_residue_components(img, settings)
        ambiguous = next(item for item in components if item.classification == "arte interno")

        result = apply_residue_component_removal(img, components, settings)

        x1, y1, x2, y2 = ambiguous.bbox
        self.assertEqual(255, result.getpixel(((x1 + x2) // 2, (y1 + y2) // 2))[3])

    def test_residue_overlay_and_report_generate_without_error(self):
        img = _residue_refine_artwork()
        settings = _residue_settings()
        components = detect_light_residue_components(img, settings)

        overlay = overlay_residue_components(img, components, settings)
        report = json.loads(residue_component_report_json(components).decode("utf-8"))

        self.assertEqual(img.size, overlay.size)
        self.assertEqual(len(components), len(report))

    def test_pipeline_complex_white_refinement_exports_debug_files(self):
        img = _complex_character_artwork_with_residue()
        settings = replace(
            _complex_pipeline_settings(),
            residue_refine_enabled=True,
            residue_luminosity=218,
            residue_saturation=60,
            residue_min_area=4,
            residue_max_area=2000,
            residue_remove_connected=True,
            residue_remove_small=False,
            residue_preserve_internal=True,
        )

        payload = process_artwork(img, {"type": "Fondo blanco complejo"}, settings)

        self.assertIsNotNone(payload["complex_white_debug"])
        self.assertIn("residue", payload["complex_white_debug"])
        with ZipFile(BytesIO(payload["zip"])) as zf:
            names = set(zf.namelist())
            self.assertIn("debug_residue_components.png", names)
            self.assertIn("debug_residue_overlay.png", names)
            self.assertIn("debug_refined_preview_black.png", names)
            self.assertIn("debug_refined_preview_red.png", names)
            self.assertIn("debug_component_report.json", names)

    def test_residue_refine_does_not_alter_clean_transparent_png(self):
        img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        img.alpha_composite(Image.new("RGBA", (20, 20), (220, 30, 80, 255)), (14, 14))
        settings = _residue_settings()

        components = detect_light_residue_components(img, settings)
        result = apply_residue_component_removal(img, components, settings)

        self.assertEqual([], components)
        self.assertTrue(np.array_equal(np.array(img), np.array(result)))

    def test_before_after_preview_combines_two_panels(self):
        before = Image.new("RGBA", (40, 30), (0, 0, 0, 255))
        after = Image.new("RGBA", (40, 30), (255, 0, 0, 128))

        result = before_after_preview(before, after, "Playera negra", max_side=120)

        self.assertGreater(result.width, before.width)
        self.assertGreaterEqual(result.height, before.height)

    def test_alpha_histogram_reports_distribution(self):
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        for x in range(5):
            for y in range(10):
                img.putpixel((x, y), (255, 0, 0, 0))
        for y in range(10):
            img.putpixel((9, y), (255, 0, 0, 120))

        stats = alpha_histogram(img)

        self.assertEqual(50.0, stats.transparent_percent)
        self.assertEqual(40.0, stats.solid_percent)
        self.assertEqual(10.0, stats.semitransparent_percent)

    def test_quality_score_has_dtf_checks(self):
        img = Image.new("RGBA", (1200, 1200), (255, 0, 0, 0))
        img.alpha_composite(Image.new("RGBA", (900, 900), (255, 0, 0, 255)), (150, 150))

        report = evaluate_dtf_quality(img, dpi=300)

        self.assertGreaterEqual(report.score, 60)
        self.assertIn("Sin fondo", report.checks)

    def test_pipeline_preserves_dimensions_without_resize_options(self):
        img = Image.new("RGBA", (28, 18), (255, 255, 255, 255))
        settings = PipelineSettings(
            mode_key="preserve_artwork",
            use_ai=False,
            remove_black=False,
            remove_color=False,
            clean_enabled=False,
            trim=False,
            alpha_cut=70,
            despeckle_area=1,
            edge_contract=0,
            black_threshold=24,
            black_level="normal",
            color_tolerance=42,
            protect_details=True,
            protect_white_details=True,
            white_protection_level="normal",
            fine_detail_level="maxima",
            safe_mode=True,
            enable_dtf_prepress=True,
            remove_white_halo=True,
            remove_black_halo=False,
            halo_strength="suave",
            expand_edge_px=1,
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
        )

        result = process_artwork(img, {"type": "dtf artwork", "use_ai": False}, settings)

        self.assertEqual((28, 18), result["image"].size)
        with ZipFile(BytesIO(result["zip"])) as zf:
            self.assertIn("alpha_quality.json", zf.namelist())
            self.assertIn("small_elements_report.json", zf.namelist())

    def test_presets_expose_required_modes(self):
        keys = available_mode_keys()

        self.assertIn("black_bg", keys)
        self.assertEqual("dtf_ready", preset_for_mode("dtf_ready")["key"])

    def test_detection_value_handles_legacy_payloads(self):
        self.assertEqual("-", detection_value({}, "dominant_color"))
        self.assertEqual(0, detection_value({"background_uniformity": None}, "background_uniformity", 0))

    def test_export_zip_contains_required_names_and_metadata(self):
        img = Image.new("RGBA", (3543, 4724), (255, 0, 0, 255))
        package = build_export_package(img, dpi=300, mode="dtf_ready", original=img, processing_seconds=1.2)

        with ZipFile(BytesIO(package["zip"])) as zf:
            names = set(zf.namelist())
            self.assertTrue({"original.png", "procesado.png", "procesado.pdf", "metadata.json"}.issubset(names))
            metadata = json.loads(zf.read("metadata.json"))
            self.assertEqual("dtf_ready", metadata["modo"])
            processed = Image.open(BytesIO(zf.read("procesado.png")))
            self.assertEqual((3543, 4724), processed.size)


if __name__ == "__main__":
    unittest.main()


def _synthetic_car() -> Image.Image:
    img = Image.new("RGBA", (100, 80), (255, 255, 255, 255))
    for x in range(20, 82):
        for y in range(34, 54):
            img.putpixel((x, y), (255, 80, 180, 255))
    for x in range(34, 68):
        for y in range(24, 38):
            img.putpixel((x, y), (255, 120, 200, 255))
    for x in range(40, 58):
        for y in range(27, 36):
            img.putpixel((x, y), (160, 220, 255, 255))
    for cx in [35, 75]:
        for x in range(cx - 8, cx + 9):
            for y in range(50, 67):
                if (x - cx) ** 2 + (y - 58) ** 2 <= 64:
                    img.putpixel((x, y), (25, 25, 25, 255))
                if (x - cx) ** 2 + (y - 58) ** 2 <= 16:
                    img.putpixel((x, y), (245, 245, 245, 255))
    for x in range(18, 84):
        img.putpixel((x, 33), (20, 20, 20, 255))
    return img


def _light_background_artwork() -> Image.Image:
    img = Image.new("RGBA", (90, 70), (240, 240, 240, 255))
    for x in range(24, 66):
        for y in range(20, 52):
            img.putpixel((x, y), (210, 30, 80, 255))
    for x in range(34, 54):
        for y in range(30, 39):
            img.putpixel((x, y), (255, 255, 255, 255))
    for x in range(22, 68):
        img.putpixel((x, 19), (25, 25, 25, 255))
    return img


def _residue_after_first_pass() -> Image.Image:
    img = Image.new("RGBA", (90, 70), (240, 240, 240, 0))
    for x in range(0, 28):
        for y in range(6, 62):
            img.putpixel((x, y), (240, 240, 240, 255))
    for x in range(25, 68):
        for y in range(18, 54):
            img.putpixel((x, y), (210, 30, 80, 255))
    for x in range(36, 55):
        for y in range(30, 40):
            img.putpixel((x, y), (255, 255, 255, 255))
    return img


def _outlined_artwork_with_residue() -> Image.Image:
    img = Image.new("RGBA", (70, 50), (240, 240, 240, 0))
    for x in range(0, 24):
        for y in range(0, 50):
            value = 236 + ((x + y) % 9)
            img.putpixel((x, y), (value, value, value, 255))
    for x in range(24, 50):
        img.putpixel((x, 20), (20, 20, 20, 255))
        img.putpixel((x, 36), (20, 20, 20, 255))
    for y in range(20, 37):
        img.putpixel((24, y), (20, 20, 20, 255))
        img.putpixel((50, y), (20, 20, 20, 255))
    for x in range(26, 49):
        for y in range(22, 35):
            img.putpixel((x, y), (230, 70, 150, 255))
    return img


def _complex_light_background_artwork() -> Image.Image:
    img = _light_background_artwork()
    for x in range(0, 30):
        for y in range(8, 64):
            value = 210 + ((x * 3 + y) % 9)
            img.putpixel((x, y), (value, value, value, 255))
    return img


def _white_matte_edge_image() -> Image.Image:
    img = Image.new("RGBA", (90, 70), (0, 0, 0, 0))
    for x in range(22, 68):
        for y in range(18, 54):
            img.putpixel((x, y), (240, 240, 240, 145))
    for x in range(25, 65):
        for y in range(21, 51):
            img.putpixel((x, y), (220, 40, 120, 255))
    for x in range(36, 55):
        for y in range(31, 40):
            img.putpixel((x, y), (255, 255, 255, 255))
    for x in range(24, 66):
        img.putpixel((x, 20), (20, 20, 20, 255))
    return img


def _complex_character_artwork() -> Image.Image:
    img = Image.new("RGBA", (96, 78), (255, 255, 255, 255))
    for x in range(24, 72):
        for y in range(18, 60):
            if ((x - 48) ** 2) / 900 + ((y - 39) ** 2) / 520 <= 1:
                img.putpixel((x, y), (230, 70, 145, 255))
    for x in range(27, 69):
        img.putpixel((x, 20), (20, 20, 20, 255))
        img.putpixel((x, 58), (20, 20, 20, 255))
    for y in range(24, 56):
        img.putpixel((25, y), (20, 20, 20, 255))
        img.putpixel((70, y), (20, 20, 20, 255))
    for x in range(41, 57):
        for y in range(28, 35):
            img.putpixel((x, y), (246, 205, 190, 255))
    for x in range(43, 58):
        img.putpixel((x, 37), (20, 20, 20, 255))
    for x in range(42, 59):
        for y in range(40, 47):
            img.putpixel((x, y), (255, 255, 255, 255))
    for x in range(22, 74):
        for y in [17, 61]:
            img.putpixel((x, y), (210, 210, 210, 180))
    for y in range(18, 61):
        for x in [22, 73]:
            img.putpixel((x, y), (210, 210, 210, 180))
    return img


def _complex_character_artwork_with_residue() -> Image.Image:
    img = _complex_character_artwork()
    for x in range(20, 29):
        for y in range(31, 50):
            img.putpixel((x, y), (242, 242, 242, 255))
    return img


def _residue_refine_artwork() -> Image.Image:
    img = _residue_after_first_pass()
    for x in range(39, 52):
        img.putpixel((x, 29), (20, 20, 20, 255))
        img.putpixel((x, 41), (20, 20, 20, 255))
    for y in range(29, 42):
        img.putpixel((38, y), (20, 20, 20, 255))
        img.putpixel((52, y), (20, 20, 20, 255))
    for x in range(40, 51):
        for y in range(31, 40):
            img.putpixel((x, y), (255, 255, 255, 255))
    for x in range(72, 78):
        for y in range(14, 20):
            img.putpixel((x, y), (246, 246, 246, 255))
    return img


def _residue_settings(remove_connected: bool = True) -> ResidueRefineSettings:
    return ResidueRefineSettings(
        luminosity_threshold=220,
        saturation_threshold=55,
        min_area=4,
        max_area=3000,
        remove_connected=remove_connected,
        remove_small=False,
        preserve_internal_white=True,
    )


def _complex_settings(halo_cleanup: bool = True) -> ComplexWhiteSettings:
    return ComplexWhiteSettings(
        white_tolerance=64,
        luminosity_threshold=222,
        saturation_threshold=48,
        preserve_internal_white=True,
        halo_cleanup=halo_cleanup,
        mask_offset=0,
        alpha_smoothing=1,
    )


def _auto_settings() -> PipelineSettings:
    return PipelineSettings(
        mode_key="auto",
        use_ai=False,
        remove_black=False,
        remove_color=True,
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
        complex_white_preset="Balanceado",
    )


def _complex_pipeline_settings() -> PipelineSettings:
    settings = _auto_settings()
    return PipelineSettings(
        mode_key="complex_white_bg",
        use_ai=False,
        remove_black=False,
        remove_color=False,
        clean_enabled=True,
        trim=False,
        alpha_cut=settings.alpha_cut,
        despeckle_area=settings.despeckle_area,
        edge_contract=settings.edge_contract,
        black_threshold=settings.black_threshold,
        black_level=settings.black_level,
        color_tolerance=settings.color_tolerance,
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
        complex_white_export_debug=True,
    )
