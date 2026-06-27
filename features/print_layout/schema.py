from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the print layout module."""

    id: str = "print_layout"
    status: str = "planned"
    phase: int = 1

