"""Module registry for professional MC DTF Pro features."""

from __future__ import annotations

from core.capabilities import get_capabilities
from core.feature_flags import enabled_flags


MODULES: tuple[dict[str, object], ...] = (
    {
        "id": "background_removal",
        "name": "Eliminacion de fondo",
        "package": "features.background_removal",
        "phase": 1,
        "status": "active",
        "active": True,
        "description": "Flujos reales de IA, color dominante y OpenCV desde core.",
        "evidence": ("core.background.remove_background_ai", "core.background_remove.remove_dominant_background"),
    },
    {
        "id": "cleanup",
        "name": "Limpieza segura",
        "package": "features.cleanup",
        "phase": 1,
        "status": "active",
        "active": True,
        "description": "Limpieza alfa, despeckle seguro y proteccion de detalles desde core.",
        "evidence": ("core.clean.clean_alpha", "core.clean.safe_despeckle"),
    },
    {
        "id": "dtf",
        "name": "Preparacion DTF",
        "package": "features.dtf",
        "phase": 2,
        "status": "active",
        "active": True,
        "description": "Pipeline principal con modos, exportacion y preparacion de arte.",
        "evidence": ("core.pipeline.process_artwork",),
    },
    {
        "id": "dtf_prepress",
        "name": "Preprensa DTF",
        "package": "features.dtf_prepress",
        "phase": 2,
        "status": "active",
        "active": True,
        "description": "Correccion de halos, sangrado, cutline y reporte alfa.",
        "evidence": ("features.dtf_prepress.service.FeatureService.prepare", "core.dtf_prepress.prepare_dtf"),
    },
    {
        "id": "vectorization",
        "name": "Vectorizacion",
        "package": "features.vectorization",
        "phase": 3,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "upscale",
        "name": "Upscale",
        "package": "features.upscale",
        "phase": 3,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "image_correction",
        "name": "Correccion de imagen",
        "package": "features.image_correction",
        "phase": 4,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "object_removal",
        "name": "Eliminacion de objetos",
        "package": "features.object_removal",
        "phase": 5,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "logos",
        "name": "Herramientas para logos",
        "package": "features.logos",
        "phase": 3,
        "status": "experimental",
        "active": False,
        "description": "Servicio parcial con paleta, reduccion y capas; aun no estable.",
        "evidence": ("features.logos.service.FeatureService.palette", "core.logo_tools.export_color_layers"),
    },
    {
        "id": "sublimation",
        "name": "Sublimacion",
        "package": "features.sublimation",
        "phase": 4,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "dtf_uv",
        "name": "DTF UV",
        "package": "features.dtf_uv",
        "phase": 4,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "design_tools",
        "name": "Herramientas de diseno",
        "package": "features.design_tools",
        "phase": 5,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "mockups",
        "name": "Mockups",
        "package": "features.mockups",
        "phase": 5,
        "status": "experimental",
        "active": False,
        "description": "Vista previa disponible en UI; servicio modular aun es solo metadata.",
        "evidence": ("ui.preview.render_result_workspace",),
    },
    {
        "id": "print_layout",
        "name": "Layout de impresion",
        "package": "features.print_layout",
        "phase": 6,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "exporting",
        "name": "Exportacion",
        "package": "features.exporting",
        "phase": 7,
        "status": "active",
        "active": True,
        "description": "Exportacion PNG, PDF y ZIP con metadata desde core.",
        "evidence": ("core.export.build_export_package", "core.export.png_bytes"),
    },
    {
        "id": "batch",
        "name": "Procesamiento por lotes",
        "package": "features.batch",
        "phase": 6,
        "status": "active",
        "active": True,
        "description": "Flujo de lote conectado en Streamlit y tabla de estado en UI.",
        "evidence": ("ui.batch.render_batch_table",),
    },
    {
        "id": "ai_tools",
        "name": "Herramientas IA",
        "package": "features.ai_tools",
        "phase": 3,
        "status": "experimental",
        "active": False,
        "description": "Rembg esta integrado en core, pero el servicio modular aun es metadata.",
        "evidence": ("core.background.get_rembg_session", "core.background.should_use_ai"),
    },
    {
        "id": "measurement",
        "name": "Medicion",
        "package": "features.measurement",
        "phase": 2,
        "status": "active",
        "active": True,
        "description": "Informacion tecnica de DPI, centimetros, peso y transparencia en UI.",
        "evidence": ("ui.preview.render_technical_info",),
    },
    {
        "id": "compatibility",
        "name": "Compatibilidad",
        "package": "features.compatibility",
        "phase": 1,
        "status": "active",
        "active": True,
        "description": "Validaciones de arquitectura, imports y configuracion de plataforma.",
        "evidence": ("core.capabilities.get_capabilities", "core.feature_flags.enabled_flags"),
    },
    {
        "id": "business",
        "name": "Negocio",
        "package": "features.business",
        "phase": 8,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
    {
        "id": "automation",
        "name": "Automatizacion",
        "package": "features.automation",
        "phase": 8,
        "status": "planned",
        "active": False,
        "description": "Modulo futuro; actualmente solo frontera y metadata.",
        "evidence": (),
    },
)


def get_available_modules() -> list[dict[str, object]]:
    """Return every registered module."""
    return [dict(module) for module in MODULES]


def get_active_modules() -> list[dict[str, object]]:
    """Return active modules."""
    return [dict(module) for module in MODULES if module["status"] == "active"]


def get_modules_by_phase(phase: int) -> list[dict[str, object]]:
    """Return modules for a roadmap phase."""
    return [dict(module) for module in MODULES if int(module["phase"]) == phase]


def is_feature_enabled(feature_id: str) -> bool:
    """Return whether a module or feature flag is enabled."""
    for module in MODULES:
        if module["id"] == feature_id:
            return module["status"] == "active"
    flag_key = f"ENABLE_{feature_id.upper()}"
    return bool(enabled_flags().get(flag_key, False))


def professional_status() -> dict[str, object]:
    """Return a summary for the professional center."""
    capabilities = get_capabilities()
    return {
        "capabilities": len(capabilities),
        "active_modules": len(get_active_modules()),
        "planned_modules": sum(1 for module in MODULES if module["status"] == "planned"),
        "experimental_modules": sum(1 for module in MODULES if module["status"] == "experimental"),
        "engine": "No destructivo",
    }
