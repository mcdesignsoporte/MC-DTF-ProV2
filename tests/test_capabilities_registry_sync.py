from __future__ import annotations

import unittest

from core.capabilities import get_capabilities, get_capability
from core.module_registry import get_available_modules, is_feature_enabled


class CapabilitiesRegistrySyncTests(unittest.TestCase):
    def test_capabilities_are_derived_from_registered_modules(self) -> None:
        modules = {module["id"]: module for module in get_available_modules()}
        capabilities = {capability["id"]: capability for capability in get_capabilities()}

        self.assertEqual(set(modules), set(capabilities))

    def test_capability_statuses_match_registry_statuses(self) -> None:
        modules = {module["id"]: module for module in get_available_modules()}

        for capability in get_capabilities():
            with self.subTest(capability=capability["id"]):
                module = modules[capability["id"]]
                self.assertEqual(module["status"], capability["status"])
                self.assertEqual(module["name"], capability["name"])
                self.assertEqual(module["description"], capability["description"])

    def test_no_contradictory_feature_enabled_state(self) -> None:
        for module in get_available_modules():
            with self.subTest(module=module["id"]):
                expected = module["status"] == "active" and bool(module["user_enabled"])
                self.assertEqual(expected, is_feature_enabled(str(module["id"])))

    def test_planned_modules_are_not_user_enabled(self) -> None:
        planned = [module for module in get_available_modules() if module["status"] == "planned"]

        self.assertGreater(len(planned), 0)
        for module in planned:
            self.assertFalse(module["user_enabled"])
            self.assertFalse(is_feature_enabled(str(module["id"])))

    def test_experimental_modules_are_not_stable_capabilities(self) -> None:
        experimental = [module for module in get_available_modules() if module["status"] == "experimental"]

        self.assertGreater(len(experimental), 0)
        for module in experimental:
            capability = get_capability(str(module["id"]))
            self.assertIsNotNone(capability)
            self.assertEqual("experimental", capability["status"])
            self.assertFalse(is_feature_enabled(str(module["id"])))


if __name__ == "__main__":
    unittest.main()
