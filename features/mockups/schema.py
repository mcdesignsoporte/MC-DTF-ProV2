from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the mockups module."""

    id: str = "mockups"
    status: str = "planned"
    phase: int = 1

