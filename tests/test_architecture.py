import unittest

import numpy as np
from PIL import Image

from core.capabilities import get_capabilities, get_capability
from core.feature_flags import enabled_flags, is_enabled
from core.module_registry import get_active_modules, get_available_modules, get_modules_by_phase, is_feature_enabled
from core.non_destructive import build_protected_art_mask, restore_protected_pixels, safe_background_removal
from core.quality_report import quality_report


class ArchitectureTests(unittest.TestCase):
    def test_capabilities_catalog_has_twenty_items(self):
        capabilities = get_capabilities()

        self.assertEqual(len(get_available_modules()), len(capabilities))
        self.assertEqual("background_removal", capabilities[0]["id"])
        self.assertEqual("active", get_capability("exporting")["status"])

    def test_feature_flags_default_values(self):
        flags = enabled_flags()

        self.assertTrue(flags["ENABLE_MOCKUPS"])
        self.assertTrue(flags["ENABLE_BATCH"])
        self.assertFalse(is_enabled("ENABLE_API"))

    def test_module_registry_exposes_active_and_phase_modules(self):
        modules = get_available_modules()
        active = get_active_modules()
        phase_one = get_modules_by_phase(1)

        self.assertGreaterEqual(len(modules), 20)
        self.assertGreater(len(active), 0)
        self.assertTrue(any(module["id"] == "cleanup" for module in phase_one))
        self.assertTrue(is_feature_enabled("batch"))
        self.assertTrue(is_feature_enabled("dtf_prepress"))
        self.assertFalse(is_feature_enabled("business"))

    def test_non_destructive_wrapper_removes_background_and_protects_art(self):
        img = Image.new("RGBA", (40, 40), (255, 255, 255, 255))
        img.alpha_composite(Image.new("RGBA", (16, 16), (255, 0, 0, 255)), (12, 12))

        mask = build_protected_art_mask(img)
        result, background = safe_background_removal(img, tolerance=48)

        self.assertTrue(mask[18, 18])
        self.assertTrue(background[0, 0])
        self.assertEqual(0, result.getpixel((0, 0))[3])
        self.assertEqual(255, result.getpixel((18, 18))[3])

    def test_restore_protected_pixels(self):
        original = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
        result = Image.new("RGBA", (20, 20), (255, 0, 0, 0))
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True

        restored, restored_mask = restore_protected_pixels(original, result, mask)

        self.assertTrue(restored_mask[10, 10])
        self.assertEqual(255, restored.getpixel((10, 10))[3])

    def test_quality_report_returns_score_and_risk(self):
        original = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
        result = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        art = np.zeros((20, 20), dtype=bool)
        art[8:12, 8:12] = True
        bg = np.ones((20, 20), dtype=bool)

        report = quality_report(original, result, art, bg, {"risk_detected": True})

        self.assertEqual("alto", report["risk_level"])
        self.assertGreater(report["removed_background_percent"], 0)
        self.assertIn("warnings", report)


if __name__ == "__main__":
    unittest.main()
