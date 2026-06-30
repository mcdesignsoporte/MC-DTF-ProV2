from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from PIL import Image

from core.residue_refine import (
    InternalLightResidueSettings,
    ResidueRefineSettings,
    apply_residue_component_removal,
    build_internal_residue_debug,
    build_residue_debug,
    detect_internal_light_residue_components,
    detect_light_residue_components,
    remove_selected_light_residue_components,
)
from core.white_complex import ComplexWhiteSettings, compose_on_solid, remove_complex_white_background


@dataclass(frozen=True)
class ProductionDTFPreset:
    """One-click production cleanup settings for complex white DTF artwork."""

    name: str
    complex_white: ComplexWhiteSettings
    residue: ResidueRefineSettings
    internal: InternalLightResidueSettings
    detect_only: bool = False


@dataclass(frozen=True)
class ProductionDTFResult:
    """Production cleanup result plus QA/debug information."""

    image: Image.Image
    stats: dict[str, object]
    debug: dict[str, object]


PRODUCTION_DTF_PRESETS: dict[str, ProductionDTFPreset] = {
    "Conservador": ProductionDTFPreset(
        name="Conservador",
        complex_white=ComplexWhiteSettings(
            white_tolerance=52,
            luminosity_threshold=232,
            saturation_threshold=34,
            preserve_internal_white=True,
            halo_cleanup=True,
            mask_offset=-1,
            alpha_smoothing=1,
        ),
        residue=ResidueRefineSettings(
            luminosity_threshold=226,
            saturation_threshold=38,
            min_area=5,
            max_area=3500,
            remove_connected=True,
            remove_small=False,
            preserve_internal_white=True,
        ),
        internal=InternalLightResidueSettings(
            min_area=4,
            max_area=1200,
            dark_neighbor_threshold=48,
            luminosity_threshold=224,
            saturation_threshold=42,
            auto_remove_high_confidence=True,
        ),
    ),
    "Balanceado": ProductionDTFPreset(
        name="Balanceado",
        complex_white=ComplexWhiteSettings(
            white_tolerance=68,
            luminosity_threshold=220,
            saturation_threshold=52,
            preserve_internal_white=True,
            halo_cleanup=True,
            mask_offset=0,
            alpha_smoothing=1,
        ),
        residue=ResidueRefineSettings(
            luminosity_threshold=218,
            saturation_threshold=58,
            min_area=4,
            max_area=12000,
            remove_connected=True,
            remove_small=True,
            preserve_internal_white=True,
        ),
        internal=InternalLightResidueSettings(
            min_area=3,
            max_area=5000,
            dark_neighbor_threshold=40,
            luminosity_threshold=214,
            saturation_threshold=64,
            auto_remove_high_confidence=True,
        ),
    ),
    "Agresivo DTF": ProductionDTFPreset(
        name="Agresivo DTF",
        complex_white=ComplexWhiteSettings(
            white_tolerance=86,
            luminosity_threshold=204,
            saturation_threshold=78,
            preserve_internal_white=True,
            halo_cleanup=True,
            mask_offset=1,
            alpha_smoothing=2,
        ),
        residue=ResidueRefineSettings(
            luminosity_threshold=205,
            saturation_threshold=82,
            min_area=3,
            max_area=30000,
            remove_connected=True,
            remove_small=True,
            preserve_internal_white=True,
        ),
        internal=InternalLightResidueSettings(
            min_area=3,
            max_area=15000,
            dark_neighbor_threshold=28,
            luminosity_threshold=198,
            saturation_threshold=88,
            auto_remove_high_confidence=True,
        ),
    ),
    "Solo detectar": ProductionDTFPreset(
        name="Solo detectar",
        complex_white=ComplexWhiteSettings(),
        residue=ResidueRefineSettings(
            luminosity_threshold=218,
            saturation_threshold=58,
            min_area=4,
            max_area=30000,
            remove_connected=True,
            remove_small=False,
            preserve_internal_white=True,
        ),
        internal=InternalLightResidueSettings(
            min_area=3,
            max_area=15000,
            dark_neighbor_threshold=30,
            luminosity_threshold=205,
            saturation_threshold=80,
            auto_remove_high_confidence=False,
        ),
        detect_only=True,
    ),
}


def production_dtf_preset(name: str) -> ProductionDTFPreset:
    """Return a production preset, falling back to Balanceado."""
    return PRODUCTION_DTF_PRESETS.get(name, PRODUCTION_DTF_PRESETS["Balanceado"])


def production_dtf_preset_names() -> list[str]:
    """Return the stable UI order for production presets."""
    return list(PRODUCTION_DTF_PRESETS.keys())


def clean_for_production(image: Image.Image, preset_name: str = "Balanceado") -> ProductionDTFResult:
    """Apply one-click safe cleanup for complex white DTF artwork.

    High-confidence exterior and internal residues are removed. Protected and
    ambiguous components are left for review/manual cleanup.
    """
    preset = production_dtf_preset(preset_name)
    source = image.convert("RGBA")
    work = source
    debug: dict[str, object] = {}
    stats: dict[str, object] = {
        "preset": preset.name,
        "detect_only": preset.detect_only,
    }

    # If the image is still mostly opaque on a white background, first run the
    # complex-white cutter. If it is already transparent, this step is harmless
    # but preserves existing alpha by removing only detected background.
    complex_result = remove_complex_white_background(work, preset.complex_white)
    work = complex_result.image if not preset.detect_only else work
    debug["complex_white"] = {
        "stats": complex_result.stats,
        "previews": {
            "preview_green": compose_on_solid(complex_result.image, (0, 150, 80)),
            "preview_black": compose_on_solid(complex_result.image, (0, 0, 0)),
        },
    }

    residue_source = work
    residue_components = detect_light_residue_components(residue_source, preset.residue)
    residue_refined = residue_source if preset.detect_only else apply_residue_component_removal(residue_source, residue_components, preset.residue)
    debug["residue"] = build_residue_debug(residue_source, residue_refined, residue_components, preset.residue)
    work = residue_refined

    internal_source = work
    internal_components = detect_internal_light_residue_components(internal_source, preset.internal)
    internal_refined = internal_source if preset.detect_only else remove_selected_light_residue_components(internal_source, internal_components, preset.internal)
    debug["internal_residue"] = build_internal_residue_debug(internal_source, internal_refined, internal_components, preset.internal)
    work = internal_refined

    stats.update(_production_stats(source, work, residue_components, internal_components, preset))
    debug["production"] = {
        "stats": stats,
        "previews": {
            "preview_green": compose_on_solid(work, (0, 150, 80)),
            "preview_black": compose_on_solid(work, (0, 0, 0)),
        },
        "report_json": json.dumps(stats, indent=2, ensure_ascii=False).encode("utf-8"),
    }
    return ProductionDTFResult(image=work, stats=stats, debug=debug)


def suggested_internal_component_ids(image: Image.Image, preset_name: str = "Balanceado") -> tuple[int, ...]:
    """Return IDs that the selected production preset would auto-remove."""
    preset = production_dtf_preset(preset_name)
    components = detect_internal_light_residue_components(image, preset.internal)
    return tuple(item.id for item in components if item.suggested_action == "borrar")


def _production_stats(source: Image.Image, result: Image.Image, residue_components, internal_components, preset: ProductionDTFPreset) -> dict[str, object]:
    src_alpha = np.array(source.convert("RGBA").getchannel("A"))
    dst_alpha = np.array(result.convert("RGBA").getchannel("A"))
    removed_pixels = int(np.count_nonzero((src_alpha > 20) & (dst_alpha <= 20)))
    residue_removed = sum(1 for item in residue_components if item.classification == "borrar") if not preset.detect_only else 0
    internal_removed = sum(1 for item in internal_components if item.suggested_action == "borrar") if not preset.detect_only else 0
    pending_review = sum(1 for item in internal_components if item.suggested_action == "revisar")
    protected = sum(1 for item in internal_components if item.suggested_action == "conservar")
    return {
        "production_removed_pixels": removed_pixels,
        "production_alpha_change_percent": round(removed_pixels / max(1, src_alpha.size) * 100, 4),
        "residue_components_detected": len(residue_components),
        "residue_components_removed": residue_removed,
        "internal_components_detected": len(internal_components),
        "internal_components_removed": internal_removed,
        "internal_components_pending_review": pending_review,
        "internal_components_protected": protected,
        "production_pending_components": pending_review + protected,
        "production_status": _production_status(pending_review, protected, removed_pixels, preset),
    }


def _production_status(review: int, protected: int, removed_pixels: int, preset: ProductionDTFPreset) -> str:
    if preset.detect_only:
        return "Solo detectar"
    if review > 0:
        return "Revisión obligatoria"
    if protected > 0:
        return "Revisión recomendada"
    if removed_pixels <= 0:
        return "Revisión recomendada"
    return "Lista para revisar/exportar"
