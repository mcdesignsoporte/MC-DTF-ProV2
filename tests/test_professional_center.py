from __future__ import annotations

import inspect
import unittest

from core.module_registry import get_available_modules
from ui import professional_center
from ui.professional_center import professional_center_sections


class ProfessionalCenterTests(unittest.TestCase):
    def test_professional_center_uses_module_registry_sections(self) -> None:
        sections = professional_center_sections()
        expected_states = {"active", "experimental", "planned"}

        self.assertEqual(expected_states, set(sections))
        for state in expected_states:
            self.assertGreater(len(sections[state]), 0)

    def test_professional_center_rows_cover_visible_registry_modules(self) -> None:
        visible_modules = [
            module
            for module in get_available_modules()
            if bool(module.get("show_in_center", True))
        ]
        row_count = sum(len(rows) for rows in professional_center_sections().values())

        self.assertEqual(len(visible_modules), row_count)

    def test_professional_center_does_not_render_duplicate_capability_list(self) -> None:
        source = inspect.getsource(professional_center)

        self.assertIn("get_available_modules", source)
        self.assertNotIn("get_capabilities", source)
        self.assertNotIn("get_active_modules", source)

    def test_planned_modules_are_not_user_enabled_in_center(self) -> None:
        planned_rows = professional_center_sections()["planned"]

        self.assertGreater(len(planned_rows), 0)
        for row in planned_rows:
            self.assertEqual("No", row["Usuario final"])

    def test_rows_expose_required_professional_center_fields(self) -> None:
        required = {"Nombre", "Descripcion", "Estado", "Categoria", "Usuario final", "Evidencia"}

        for rows in professional_center_sections().values():
            for row in rows:
                self.assertEqual(required, set(row))


if __name__ == "__main__":
    unittest.main()
