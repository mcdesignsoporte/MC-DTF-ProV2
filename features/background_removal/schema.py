from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the background removal module."""

    id: str = "background_removal"
    status: str = "planned"
    phase: int = 1

