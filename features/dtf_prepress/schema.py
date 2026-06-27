from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata contract for the dtf prepress module."""

    id: str = "dtf_prepress"
    status: str = "active"
    phase: int = 2
