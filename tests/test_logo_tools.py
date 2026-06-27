import unittest

import numpy as np
from PIL import Image

from core.logo_tools import (
    black_to_transparent,
    detect_dominant_colors,
    export_color_layers,
    reduce_colors,
    separate_colors,
    unify_similar_colors,
    white_to_transparent,
)


class LogoToolsTests(unittest.TestCase):
    def test_detect_dominant_colors(self):
        img = _logo_image()

        colors = detect_dominant_colors(img, max_colors=4)

        self.assertGreaterEqual(len(colors), 3)
        self.assertIn("hex", colors[0])

    def test_reduce_colors(self):
        img = _logo_image()
        noisy = img.copy()
        noisy.putpixel((5, 5), (250, 2, 1, 255))

        result = reduce_colors(noisy, max_colors=3)
        opaque = np.array(result)[np.array(result.getchannel("A")) > 20][:, :3]

        self.assertLessEqual(len(np.unique(opaque, axis=0)), 3)

    def test_black_to_transparent(self):
        img = _logo_image()

        result = black_to_transparent(img, tolerance=8)

        self.assertEqual(0, result.getpixel((20, 20))[3])
        self.assertEqual(255, result.getpixel((5, 5))[3])

    def test_white_to_transparent(self):
        img = _logo_image()

        result = white_to_transparent(img, tolerance=8)

        self.assertEqual(0, result.getpixel((35, 35))[3])
        self.assertEqual(255, result.getpixel((5, 5))[3])

    def test_unify_similar_colors(self):
        img = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        for x in range(10):
            img.putpixel((x, 5), (250, 0, 0, 255))
        for x in range(10, 20):
            img.putpixel((x, 5), (240, 5, 5, 255))

        result = unify_similar_colors(img, tolerance=64)
        opaque = np.array(result)[np.array(result.getchannel("A")) > 20][:, :3]

        self.assertLessEqual(len(np.unique(opaque, axis=0)), 4)

    def test_separate_colors(self):
        img = _logo_image()

        layers = separate_colors(img, max_colors=4)

        self.assertGreaterEqual(len(layers), 3)
        self.assertTrue(all(layer["image"].mode == "RGBA" for layer in layers))

    def test_export_color_layers(self):
        img = _logo_image()

        exported = export_color_layers(img, max_colors=4)

        self.assertIn("palette", exported)
        self.assertIn("files", exported)
        self.assertTrue(any(name.startswith("layer_01") for name in exported["files"]))


def _logo_image() -> Image.Image:
    img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    img.alpha_composite(Image.new("RGBA", (16, 16), (255, 0, 0, 255)), (0, 0))
    img.alpha_composite(Image.new("RGBA", (16, 16), (0, 0, 255, 255)), (20, 0))
    img.alpha_composite(Image.new("RGBA", (16, 16), (0, 0, 0, 255)), (12, 12))
    img.alpha_composite(Image.new("RGBA", (8, 8), (255, 255, 255, 255)), (32, 32))
    return img


if __name__ == "__main__":
    unittest.main()
