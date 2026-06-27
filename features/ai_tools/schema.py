from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the ai tools module."""

    id: str = "ai_tools"
    status: str = "planned"
    phase: int = 1

