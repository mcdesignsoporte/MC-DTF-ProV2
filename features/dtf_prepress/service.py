from __future__ import annotations

from PIL import Image

from core.dtf_prepress import DTFPrepressSettings, prepare_dtf

from .schema import FeatureSchema


class FeatureService:
    """Service boundary for the dtf prepress module."""

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self.schema = schema or FeatureSchema()

    def metadata(self) -> dict[str, object]:
        """Return module metadata."""
        return {"id": self.schema.id, "status": self.schema.status, "phase": self.schema.phase}

    def prepare(self, image: Image.Image, settings: DTFPrepressSettings | None = None):
        """Run DTF prepress with explicit settings."""
        return prepare_dtf(image, settings or DTFPrepressSettings())
