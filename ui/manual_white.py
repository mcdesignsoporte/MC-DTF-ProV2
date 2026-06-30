from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedValidationResult:
    seed: tuple[int, int] | None
    reason: str | None = None


def scale_click_to_image_coordinates(
    visual_x: int | float,
    visual_y: int | float,
    display_size: tuple[int, int],
    image_size: tuple[int, int],
    *,
    reject_origin: bool = True,
) -> SeedValidationResult:
    """Convert a click from displayed image coordinates to real image coordinates.

    The Streamlit click component may return an initial (0, 0)-like value on some
    clients before the user actually clicks. For this tool, silently accepting that
    value creates a destructive false seed, so it is rejected by default.
    """
    display_w, display_h = int(display_size[0]), int(display_size[1])
    image_w, image_h = int(image_size[0]), int(image_size[1])

    if display_w <= 0 or display_h <= 0 or image_w <= 0 or image_h <= 0:
        return SeedValidationResult(None, "tamano_invalido")

    try:
        vx = int(round(float(visual_x)))
        vy = int(round(float(visual_y)))
    except (TypeError, ValueError):
        return SeedValidationResult(None, "clic_invalido")

    if reject_origin and vx == 0 and vy == 0:
        return SeedValidationResult(None, "clic_no_confirmado")

    if not (0 <= vx < display_w and 0 <= vy < display_h):
        return SeedValidationResult(None, "clic_fuera_de_imagen")

    real_x = int(round(vx * image_w / display_w))
    real_y = int(round(vy * image_h / display_h))
    real_x = max(0, min(image_w - 1, real_x))
    real_y = max(0, min(image_h - 1, real_y))

    if reject_origin and real_x == 0 and real_y == 0:
        return SeedValidationResult(None, "clic_no_confirmado")

    return SeedValidationResult((real_x, real_y), None)


def parse_manual_seed_inputs(
    x_text: str,
    y_text: str,
    image_size: tuple[int, int] | None,
    *,
    allow_origin: bool = False,
) -> SeedValidationResult:
    """Parse manual X/Y fields without treating empty/default (0,0) as a seed."""
    x_raw = str(x_text or "").strip()
    y_raw = str(y_text or "").strip()

    if not x_raw or not y_raw:
        return SeedValidationResult(None, "coordenadas_vacias")

    try:
        x = int(x_raw)
        y = int(y_raw)
    except ValueError:
        return SeedValidationResult(None, "coordenadas_invalidas")

    if x == 0 and y == 0 and not allow_origin:
        return SeedValidationResult(None, "origen_requiere_confirmacion")

    if image_size is not None:
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            return SeedValidationResult(None, "tamano_invalido")
        if not (0 <= x < width and 0 <= y < height):
            return SeedValidationResult(None, "coordenadas_fuera_de_imagen")

    return SeedValidationResult((x, y), None)
