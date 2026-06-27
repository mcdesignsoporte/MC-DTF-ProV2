from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the automation module."""

    id: str = "automation"
    status: str = "planned"
    phase: int = 1

