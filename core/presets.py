from __future__ import annotations

from core.modes import MODES


def preset_for_mode(mode_key: str) -> dict[str, object]:
    """Return the processing preset for a mode key."""
    for mode in MODES.values():
        if mode["key"] == mode_key:
            return dict(mode)
    return dict(MODES["Preparar DTF"])


def available_mode_keys() -> list[str]:
    """Return stable mode keys for integrations and tests."""
    return [str(mode["key"]) for mode in MODES.values()]
