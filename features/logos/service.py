from __future__ import annotations

from PIL import Image

from core.logo_tools import detect_dominant_colors, export_color_layers, reduce_colors

from .schema import FeatureSchema


class FeatureService:
    """Service boundary for professional logo tools."""

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self.schema = schema or FeatureSchema()

    def metadata(self) -> dict[str, object]:
        """Return module metadata."""
        return {"id": self.schema.id, "status": self.schema.status, "phase": self.schema.phase}

    def palette(self, image: Image.Image, max_colors: int = 8) -> list[dict[str, object]]:
        """Return dominant logo colors."""
        return detect_dominant_colors(image, max_colors=max_colors)

    def reduce(self, image: Image.Image, max_colors: int = 8) -> Image.Image:
        """Reduce image palette."""
        return reduce_colors(image, max_colors=max_colors)

    def layers(self, image: Image.Image, max_colors: int = 8) -> dict[str, object]:
        """Export logo layers by color."""
        return export_color_layers(image, max_colors=max_colors)
