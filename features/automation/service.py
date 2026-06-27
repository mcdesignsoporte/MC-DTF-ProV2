from __future__ import annotations

from .schema import FeatureSchema


class FeatureService:
    """Service boundary for the automation module."""

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self.schema = schema or FeatureSchema()

    def metadata(self) -> dict[str, object]:
        """Return module metadata without activating unfinished features."""
        return {
            "id": self.schema.id,
            "status": self.schema.status,
            "phase": self.schema.phase,
        }

