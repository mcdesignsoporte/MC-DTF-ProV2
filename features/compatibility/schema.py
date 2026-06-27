from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the compatibility module."""

    id: str = "compatibility"
    status: str = "planned"
    phase: int = 1

