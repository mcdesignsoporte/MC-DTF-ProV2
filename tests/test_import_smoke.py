from __future__ import annotations

import importlib
import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_core_and_ui_modules_import(self) -> None:
        modules = (
            "core.pipeline",
            "core.modes",
            "core.module_registry",
            "ui.sidebar",
        )

        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))


if __name__ == "__main__":
    unittest.main()
