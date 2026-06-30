from __future__ import annotations

import unittest

from ui.manual_white import parse_manual_seed_inputs, scale_click_to_image_coordinates
from ui.sidebar import _manual_seeds


class ManualWhiteUITests(unittest.TestCase):
    def test_click_origin_is_not_used_as_seed(self) -> None:
        result = scale_click_to_image_coordinates(0, 0, (100, 100), (1000, 1000))

        self.assertIsNone(result.seed)
        self.assertEqual("clic_no_confirmado", result.reason)
        self.assertEqual((), _manual_seeds(None, None, "", click_data={"x": 0, "y": 0}, display_size=(100, 100), image_size=(1000, 1000)))

    def test_valid_click_scales_to_real_image_coordinates(self) -> None:
        result = scale_click_to_image_coordinates(25, 50, (100, 100), (1000, 500))

        self.assertEqual((250, 250), result.seed)
        self.assertEqual(((250, 250),), _manual_seeds(None, None, "", click_data={"x": 25, "y": 50}, display_size=(100, 100), image_size=(1000, 500)))

    def test_manual_seed_requires_explicit_non_origin_values(self) -> None:
        self.assertEqual("coordenadas_vacias", parse_manual_seed_inputs("", "", (100, 100)).reason)
        self.assertEqual("origen_requiere_confirmacion", parse_manual_seed_inputs("0", "0", (100, 100)).reason)
        self.assertEqual((12, 34), parse_manual_seed_inputs("12", "34", (100, 100)).seed)
        self.assertEqual((), _manual_seeds(0, 0, "", include_manual=True))
        self.assertEqual(((12, 34),), _manual_seeds(12, 34, "", include_manual=True))

    def test_selected_origin_seed_is_rejected(self) -> None:
        self.assertEqual((), _manual_seeds(None, None, "", selected_seed=(0, 0)))


if __name__ == "__main__":
    unittest.main()
