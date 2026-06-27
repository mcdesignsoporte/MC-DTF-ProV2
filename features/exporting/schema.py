from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the exporting module."""

    id: str = "exporting"
    status: str = "planned"
    phase: int = 1

