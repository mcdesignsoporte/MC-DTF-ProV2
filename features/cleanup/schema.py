from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the cleanup module."""

    id: str = "cleanup"
    status: str = "planned"
    phase: int = 1

