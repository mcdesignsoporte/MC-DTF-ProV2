from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the upscale module."""

    id: str = "upscale"
    status: str = "planned"
    phase: int = 1

