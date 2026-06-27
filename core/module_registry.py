"""Module registry for professional MC DTF Pro features."""

from __future__ import annotations

from core.capabilities import get_capabilities
from core.feature_flags import enabled_flags


MODULES: tuple[dict[str, object], ...] = (
    {"id": "background_removal", "package": "features.background_removal", "phase": 1, "active": True},
    {"id": "cleanup", "package": "features.cleanup", "phase": 1, "active": True},
    {"id": "dtf", "package": "features.dtf", "phase": 2, "active": True},
    {"id": "dtf_prepress", "package": "features.dtf_prepress", "phase": 2, "active": True},
    {"id": "vectorization", "package": "features.vectorization", "phase": 3, "active": False},
    {"id": "upscale", "package": "features.upscale", "phase": 3, "active": False},
    {"id": "image_correction", "package": "features.image_correction", "phase": 4, "active": False},
    {"id": "object_removal", "package": "features.object_removal", "phase": 5, "active": False},
    {"id": "logos", "package": "features.logos", "phase": 3, "active": True},
    {"id": "sublimation", "package": "features.sublimation", "phase": 4, "active": False},
    {"id": "dtf_uv", "package": "features.dtf_uv", "phase": 4, "active": False},
    {"id": "design_tools", "package": "features.design_tools", "phase": 5, "active": False},
    {"id": "mockups", "package": "features.mockups", "phase": 5, "active": True},
    {"id": "print_layout", "package": "features.print_layout", "phase": 6, "active": False},
    {"id": "exporting", "package": "features.exporting", "phase": 7, "active": True},
    {"id": "batch", "package": "features.batch", "phase": 6, "active": True},
    {"id": "ai_tools", "package": "features.ai_tools", "phase": 3, "active": True},
    {"id": "measurement", "package": "features.measurement", "phase": 2, "active": True},
    {"id": "compatibility", "package": "features.compatibility", "phase": 1, "active": True},
    {"id": "business", "package": "features.business", "phase": 8, "active": False},
    {"id": "automation", "package": "features.automation", "phase": 8, "active": False},
)


def get_available_modules() -> list[dict[str, object]]:
    """Return every registered module."""
    return [dict(module) for module in MODULES]


def get_active_modules() -> list[dict[str, object]]:
    """Return active modules."""
    return [dict(module) for module in MODULES if bool(module["active"])]


def get_modules_by_phase(phase: int) -> list[dict[str, object]]:
    """Return modules for a roadmap phase."""
    return [dict(module) for module in MODULES if int(module["phase"]) == phase]


def is_feature_enabled(feature_id: str) -> bool:
    """Return whether a module or feature flag is enabled."""
    for module in MODULES:
        if module["id"] == feature_id:
            return bool(module["active"])
    flag_key = f"ENABLE_{feature_id.upper()}"
    return bool(enabled_flags().get(flag_key, False))


def professional_status() -> dict[str, object]:
    """Return a summary for the professional center."""
    capabilities = get_capabilities()
    return {
        "capabilities": len(capabilities),
        "active_modules": len(get_active_modules()),
        "planned_modules": len(MODULES) - len(get_active_modules()),
        "engine": "No destructivo",
    }
