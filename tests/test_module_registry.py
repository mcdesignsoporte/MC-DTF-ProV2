from __future__ import annotations

import importlib
import unittest

from core.module_registry import get_active_modules, get_available_modules, is_feature_enabled


VALID_STATUSES = {"active", "planned", "experimental"}


class ModuleRegistryTests(unittest.TestCase):
    def test_registry_loads_modules(self) -> None:
        modules = get_available_modules()

        self.assertGreaterEqual(len(modules), 20)
        self.assertEqual(len(modules), len({module["id"] for module in modules}))

    def test_each_module_has_required_metadata(self) -> None:
        for module in get_available_modules():
            with self.subTest(module=module["id"]):
                self.assertIsInstance(module["id"], str)
                self.assertIsInstance(module["name"], str)
                self.assertIsInstance(module["description"], str)
                self.assertIsInstance(module["category"], str)
                self.assertIn(module["status"], VALID_STATUSES)
                self.assertTrue(module["name"])
                self.assertTrue(module["description"])
                self.assertTrue(module["category"])

    def test_active_modules_have_functionality_evidence(self) -> None:
        for module in get_active_modules():
            with self.subTest(module=module["id"]):
                evidence = tuple(module.get("evidence", ()))
                self.assertGreater(len(evidence), 0)
                for dotted_path in evidence:
                    self.assertTrue(_dotted_path_exists(str(dotted_path)), dotted_path)

    def test_placeholder_modules_are_not_active(self) -> None:
        placeholder_ids = {
            module["id"]
            for module in get_available_modules()
            if not tuple(module.get("evidence", ()))
        }

        self.assertIn("vectorization", placeholder_ids)
        self.assertIn("sublimation", placeholder_ids)
        self.assertIn("dtf_uv", placeholder_ids)
        for module_id in placeholder_ids:
            self.assertFalse(is_feature_enabled(str(module_id)), module_id)

    def test_status_matches_feature_enabled_contract(self) -> None:
        for module in get_available_modules():
            expected = module["status"] == "active"
            self.assertEqual(expected, is_feature_enabled(str(module["id"])))


def _dotted_path_exists(dotted_path: str) -> bool:
    parts = dotted_path.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        target = module
        for attr in parts[index:]:
            if not hasattr(target, attr):
                return False
            target = getattr(target, attr)
        return True
    return False
