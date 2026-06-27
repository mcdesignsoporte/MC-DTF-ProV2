"""Professional capability catalog for MC DTF Pro."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Capability:
    """Structured capability definition."""

    id: str
    name: str
    description: str
    status: str
    priority: int
    phase: int
    dependencies: tuple[str, ...]
    available_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a serializable capability payload."""
        data = asdict(self)
        data["dependencies"] = list(self.dependencies)
        data["available_actions"] = list(self.available_actions)
        return data


CAPABILITIES: tuple[Capability, ...] = (
    Capability("background_removal", "Eliminacion inteligente de fondo", "Quita solo fondo confirmado y conserva arte.", "active", 1, 1, (), ("procesar", "qa")),
    Capability("cleanup", "Limpieza automatica", "Limpia alfa y ruido aislado con motor seguro.", "active", 1, 1, ("background_removal",), ("limpiar", "qa")),
    Capability("dtf", "Preparacion para impresion DTF", "Prepara PNG/PDF/ZIP a 300 DPI.", "active", 1, 2, ("cleanup",), ("exportar",)),
    Capability("vectorization", "Vectorizacion", "Preparacion futura para SVG y trazados.", "planned", 3, 3, ("cleanup",), ()),
    Capability("upscale", "Escalado con IA", "Base futura para ampliacion inteligente.", "planned", 3, 3, ("dtf",), ()),
    Capability("image_correction", "Correccion de imagen", "Ajuste futuro de color, brillo y contraste.", "planned", 2, 4, ("dtf",), ()),
    Capability("object_removal", "Eliminacion de objetos", "Herramientas futuras para borrar objetos.", "planned", 4, 5, ("ai_tools",), ()),
    Capability("logos", "Herramientas para logos", "Limpieza y preparacion de logos.", "experimental", 2, 3, ("background_removal",), ("proteger",)),
    Capability("sublimation", "Preparacion para sublimacion", "Plantillas y espejo automatico futuros.", "planned", 3, 4, ("image_correction",), ()),
    Capability("dtf_uv", "Preparacion DTF UV", "Capa blanca y simulacion futura.", "planned", 3, 4, ("dtf",), ()),
    Capability("design_tools", "Diseno", "Texto, contornos, sombras y capas futuras.", "planned", 4, 5, ("mockups",), ()),
    Capability("mockups", "Mockups", "Previews comerciales en prendas y objetos.", "active", 2, 5, ("dtf",), ("previsualizar",)),
    Capability("print_layout", "Organizacion de impresion", "Gang sheet y nesting futuros.", "planned", 2, 6, ("batch",), ()),
    Capability("exporting", "Exportacion", "PNG, PDF y ZIP estables; formatos extra futuros.", "stable", 1, 7, ("dtf",), ("descargar",)),
    Capability("batch", "Procesamiento por lotes", "Procesa multiples imagenes con detector.", "active", 2, 6, ("background_removal",), ("procesar lote",)),
    Capability("ai_tools", "IA integrada", "Uso controlado de IA solo cuando corresponde.", "experimental", 3, 3, ("background_removal",), ("rembg",)),
    Capability("measurement", "Medicion para impresion", "Tamanos, DPI y centimetros.", "active", 2, 2, ("exporting",), ("medir",)),
    Capability("compatibility", "Compatibilidad", "Windows, Streamlit Cloud y Python 3.12.", "stable", 1, 1, (), ("validar",)),
    Capability("business", "Funciones empresariales", "Cotizacion, clientes y costos futuros.", "planned", 5, 8, ("automation",), ()),
    Capability("automation", "Automatizacion", "API, plugins y flujos futuros.", "planned", 5, 8, ("business",), ()),
)


def get_capabilities() -> list[dict[str, object]]:
    """Return all capabilities as dictionaries."""
    return [capability.to_dict() for capability in CAPABILITIES]


def get_capability(capability_id: str) -> dict[str, object] | None:
    """Return one capability by id."""
    for capability in CAPABILITIES:
        if capability.id == capability_id:
            return capability.to_dict()
    return None
