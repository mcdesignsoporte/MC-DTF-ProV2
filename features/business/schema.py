from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the business module."""

    id: str = "business"
    status: str = "planned"
    phase: int = 1

