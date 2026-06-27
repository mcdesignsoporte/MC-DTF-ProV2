import unittest
from io import BytesIO
from zipfile import ZipFile
import json

import numpy as np
from PIL import Image

from core.black_remove import remove_black_background
from core.background_remove import dominant_background_color, remove_dominant_background
from core.background import apply_ai_alpha_to_original
from core.clean import clean_alpha, contract_edge, despeckle
from core.detector import detect
from core.halftone import make_halftone
from core.image_io import image_to_png_bytes, make_zip_bytes
from core.pipeline import PipelineSettings, process_artwork
from core.presets import available_mode_keys, preset_for_mode
from ui.detection import detection_value
from core.preview import alpha_preview, before_after_preview, checkerboard, composite_preview, preview_thumbnail
from core.quality import alpha_histogram, evaluate_dtf_quality
from core.export import build_export_package


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

    def test_clean_alpha_removes_small_trash(self):
        img = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
        img.putpixel((1, 1), (255, 0, 0, 255))
        for x in range(5, 9):
            for y in range(5, 9):
                img.putpixel((x, y), (255, 0, 0, 255))

        result = clean_alpha(img, alpha_cut=70, despeckle_area=4)

        self.assertEqual(0, result.getpixel((1, 1))[3])
        self.assertEqual(255, result.getpixel((6, 6))[3])

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
            max_ai_side=1200,
            upscale=1,
            dpi=300,
            width_cm=0,
            height_cm=0,
        )

        result = process_artwork(img, {"type": "dtf artwork", "use_ai": False}, settings)

        self.assertEqual((28, 18), result["image"].size)

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
