from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the image correction module."""

    id: str = "image_correction"
    status: str = "planned"
    phase: int = 1

