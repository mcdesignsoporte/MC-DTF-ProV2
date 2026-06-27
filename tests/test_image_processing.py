import unittest

import numpy as np
from PIL import Image

from core.background import apply_ai_alpha_to_original
from core.halftone import make_halftone
from core.preview import composite_preview


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


if __name__ == "__main__":
    unittest.main()
