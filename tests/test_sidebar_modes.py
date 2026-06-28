import unittest

from core.modes import MODES
from ui.sidebar import LABEL_TO_MODE, MODE_TO_LABEL, _label_from_mode, _safe_mode_name


EXPECTED_LABEL_TO_MODE = {
    "Seguro": "Conservador Profesional",
    "Automático": "Automatico",
    "Fotografía": "Fotografia",
    "Diseño DTF": "Conservar Diseno",
    "PNG Transparente": "PNG Transparente",
    "Fondo negro": "Quitar Fondo Negro",
    "Diseño oscuro": "Diseno Oscuro",
    "Fondo de color": "Fondo de color",
    "Fondo blanco complejo": "Fondo blanco complejo",
    "Preparar DTF": "Preparar DTF",
}

EXPECTED_MODE_TO_LABEL = {
    "Automatico": "Automático",
    "Conservador Profesional": "Seguro",
    "Fotografia": "Fotografía",
    "Conservar Diseno": "Diseño DTF",
    "PNG Transparente": "PNG Transparente",
    "Quitar Fondo Negro": "Fondo negro",
    "Diseno Oscuro": "Diseño oscuro",
    "Fondo de color": "Fondo de color",
    "Fondo blanco complejo": "Fondo blanco complejo",
    "Preparar DTF": "Preparar DTF",
}


class SidebarModeMappingTests(unittest.TestCase):
    def test_visible_labels_map_to_expected_modes(self) -> None:
        self.assertEqual(EXPECTED_LABEL_TO_MODE, LABEL_TO_MODE)
        for mode_name in LABEL_TO_MODE.values():
            self.assertIn(mode_name, MODES)

    def test_internal_modes_round_trip_to_visible_labels(self) -> None:
        self.assertEqual(EXPECTED_MODE_TO_LABEL, MODE_TO_LABEL)
        for mode_name, label in MODE_TO_LABEL.items():
            self.assertEqual(label, _label_from_mode(mode_name))
            self.assertEqual(mode_name, _safe_mode_name(label))

    def test_important_modes_are_visible_and_unique(self) -> None:
        important_modes = {
            "Automatico",
            "Conservador Profesional",
            "Fotografia",
            "Conservar Diseno",
            "PNG Transparente",
            "Quitar Fondo Negro",
            "Diseno Oscuro",
            "Fondo de color",
            "Fondo blanco complejo",
            "Preparar DTF",
        }

        self.assertEqual(important_modes, set(LABEL_TO_MODE.values()))
        self.assertEqual(len(LABEL_TO_MODE), len(set(LABEL_TO_MODE.values())))
        self.assertEqual(len(MODE_TO_LABEL), len(set(MODE_TO_LABEL.values())))

    def test_specific_mode_regressions(self) -> None:
        self.assertEqual("Automatico", _safe_mode_name("Automático"))
        self.assertEqual("Quitar Fondo Negro", _safe_mode_name("Fondo negro"))
        self.assertEqual("Diseno Oscuro", _safe_mode_name("Diseño oscuro"))


if __name__ == "__main__":
    unittest.main()
