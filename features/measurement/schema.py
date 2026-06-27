from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the measurement module."""

    id: str = "measurement"
    status: str = "planned"
    phase: int = 1

