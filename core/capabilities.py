"""Compatibility accessors for the module registry capability catalog."""

from __future__ import annotations

from core.module_registry import get_available_modules


def get_capabilities() -> list[dict[str, object]]:
    """Return capability-shaped dictionaries from the module registry."""
    return [_module_to_capability(module) for module in get_available_modules()]


def get_capability(capability_id: str) -> dict[str, object] | None:
    """Return one capability by id."""
    for capability in get_capabilities():
        if capability["id"] == capability_id:
            return capability
    return None


def _module_to_capability(module: dict[str, object]) -> dict[str, object]:
    return {
        "id": module["id"],
        "name": module["name"],
        "description": module["description"],
        "status": module["status"],
        "priority": module["priority"],
        "phase": module["phase"],
        "category": module["category"],
        "dependencies": list(module.get("dependencies", ())),
        "available_actions": list(module.get("available_actions", ())),
        "show_in_center": bool(module.get("show_in_center", True)),
        "user_enabled": bool(module.get("user_enabled", False)),
    }
