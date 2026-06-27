import unittest

import numpy as np
from PIL import Image

from core.dtf_prepress import (
    DTFPrepressSettings,
    analyze_alpha_quality,
    create_bleed,
    create_cutline_mask,
    detect_black_halo,
    detect_too_small_elements,
    detect_white_halo,
    expand_edge_colors,
    prepare_dtf,
    remove_black_halo_safe,
    remove_white_halo_safe,
)


class DTFPrepressTests(unittest.TestCase):
    def test_analyze_alpha_quality_reports_distribution(self):
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        for x in range(5):
            for y in range(10):
                img.putpixel((x, y), (255, 0, 0, 0))
        img.putpixel((9, 9), (255, 0, 0, 128))

        result = analyze_alpha_quality(img)

        self.assertGreater(result["transparent_percent"], 40)
        self.assertGreater(result["semi_transparent_percent"], 0)

    def test_detect_white_halo_only_on_edge(self):
        img = _halo_image((250, 250, 250, 255))
        img.putpixel((20, 20), (255, 255, 255, 255))

        mask = detect_white_halo(img)

        self.assertTrue(mask[10, 20])
        self.assertFalse(mask[20, 20])

    def test_detect_black_halo_only_on_edge(self):
        img = _halo_image((10, 10, 10, 255))
        img.putpixel((20, 20), (0, 0, 0, 255))

        mask = detect_black_halo(img)

        self.assertTrue(mask[10, 20])
        self.assertFalse(mask[20, 20])

    def test_remove_halo_does_not_modify_internal_white_or_black(self):
        white = _halo_image((250, 250, 250, 255))
        white.putpixel((20, 20), (255, 255, 255, 255))
        black = _halo_image((10, 10, 10, 255))
        black.putpixel((20, 20), (0, 0, 0, 255))

        fixed_white = remove_white_halo_safe(white, "fuerte")
        fixed_black = remove_black_halo_safe(black, "fuerte")

        self.assertEqual((255, 255, 255), fixed_white.getpixel((20, 20))[:3])
        self.assertEqual((0, 0, 0), fixed_black.getpixel((20, 20))[:3])

    def test_expand_edge_colors_preserves_alpha(self):
        img = Image.new("RGBA", (8, 8), (255, 255, 255, 0))
        img.putpixel((4, 4), (255, 0, 0, 255))
        before_alpha = np.array(img.getchannel("A"))

        result = expand_edge_colors(img, 1)

        self.assertTrue(np.array_equal(before_alpha, np.array(result.getchannel("A"))))

    def test_create_bleed_expands_alpha(self):
        img = Image.new("RGBA", (12, 12), (255, 255, 255, 0))
        img.putpixel((6, 6), (255, 0, 0, 255))

        result, bleed = create_bleed(img, 2)

        self.assertGreater(np.count_nonzero(np.array(result.getchannel("A"))), 1)
        self.assertTrue(np.any(bleed))

    def test_create_cutline_mask(self):
        img = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        img.alpha_composite(Image.new("RGBA", (6, 6), (255, 0, 0, 255)), (7, 7))

        mask = create_cutline_mask(img, 2)

        self.assertTrue(np.any(mask))
        self.assertFalse(mask[10, 10])

    def test_detect_too_small_elements(self):
        img = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
        img.putpixel((3, 3), (255, 0, 0, 255))
        img.alpha_composite(Image.new("RGBA", (16, 16), (255, 0, 0, 255)), (12, 12))

        report = detect_too_small_elements(img, dpi=300, min_mm=1.0)

        self.assertGreaterEqual(report["count"], 1)
        self.assertTrue(report["warning"])

    def test_prepare_dtf_returns_artifacts(self):
        img = _halo_image((250, 250, 250, 255))

        result = prepare_dtf(img, DTFPrepressSettings(create_cutline=True, bleed_px=1, expand_edge_px=1))

        self.assertEqual(img.size, result.image.size)
        self.assertTrue(result.metadata["cutline_ready"])
        self.assertIn("transparent_percent", result.alpha_quality)


def _halo_image(halo_color: tuple[int, int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
    img.alpha_composite(Image.new("RGBA", (18, 18), (255, 0, 0, 255)), (11, 11))
    for x in range(10, 30):
        img.putpixel((x, 10), halo_color)
        img.putpixel((x, 29), halo_color)
        img.putpixel((10, x), halo_color)
        img.putpixel((29, x), halo_color)
    return img


if __name__ == "__main__":
    unittest.main()
