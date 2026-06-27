from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the object removal module."""

    id: str = "object_removal"
    status: str = "planned"
    phase: int = 1

