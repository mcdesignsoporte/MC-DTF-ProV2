from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the dtf uv module."""

    id: str = "dtf_uv"
    status: str = "planned"
    phase: int = 1

