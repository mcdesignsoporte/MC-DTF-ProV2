import unittest
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image

from core.black_remove import remove_black_background
from core.background import apply_ai_alpha_to_original
from core.detector import detect
from core.halftone import make_halftone
from core.image_io import image_to_png_bytes, make_zip_bytes
from core.preview import composite_preview
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

    def test_preview_is_capped_for_large_images(self):
        img = Image.new("RGBA", (4000, 2000), (0, 0, 0, 0))

        preview = composite_preview(img, "Transparente", max_side=1000)

        self.assertEqual((1000, 500), preview.size)

    def test_detector_recommends_dark_design_for_black_background(self):
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
        for x in range(35, 65):
            for y in range(35, 65):
                img.putpixel((x, y), (255, 255, 255, 255))

        result = detect(img)

        self.assertEqual("black_bg", result["recommended_mode"])

    def test_black_removal_keeps_bright_letter_details(self):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        for x in range(10, 22):
            for y in range(8, 24):
                img.putpixel((x, y), (255, 255, 255, 255))

        result = remove_black_background(img, threshold=20, protect_details=True)

        self.assertEqual(0, result.getpixel((0, 0))[3])
        self.assertEqual(255, result.getpixel((16, 16))[3])

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


if __name__ == "__main__":
    unittest.main()
