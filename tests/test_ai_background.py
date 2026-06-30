from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from core.ai_background import (
    ai_background_report,
    build_ai_foreground_mask,
    compose_ai_preview_on_color,
    remove_background_ai,
)
from core.auto_router import autopilot_route
from core.modes import MODES
from core.pipeline import PipelineSettings, process_artwork


class AIBackgroundTests(unittest.TestCase):
    def test_remove_background_ai_returns_rgba_with_alpha(self) -> None:
        result = remove_background_ai(_simple_artwork())

        self.assertEqual("RGBA", result.mode)
        alpha = np.array(result.getchannel("A"))
        self.assertGreater(np.count_nonzero(alpha < 250), 0)
        self.assertGreater(np.count_nonzero(alpha > 5), 0)

    def test_foreground_mask_and_previews_generate(self) -> None:
        result = remove_background_ai(_simple_artwork())
        mask = build_ai_foreground_mask(result)
        green = compose_ai_preview_on_color(result, "green")
        black = compose_ai_preview_on_color(result, "black")

        self.assertEqual("L", mask.mode)
        self.assertEqual("RGBA", green.mode)
        self.assertEqual("RGBA", black.mode)

    def test_ai_background_report_has_review_metrics(self) -> None:
        original = _simple_artwork()
        result = remove_background_ai(original)
        report = ai_background_report(original, result)

        self.assertGreaterEqual(report["transparent_percent"], 0)
        self.assertGreater(report["changed_alpha_percent"], 0)
        self.assertEqual("rembg", report["engine"])

    def test_mode_exists(self) -> None:
        self.assertIn("Recorte IA", MODES)
        self.assertEqual("ai_background", MODES["Recorte IA"]["key"])

    def test_pipeline_processes_ai_background_mode(self) -> None:
        payload = process_artwork(_simple_artwork(), {}, _ai_settings())
        alpha = np.array(payload["image"].getchannel("A"))

        self.assertIn("png", payload)
        self.assertIn("ai_background_debug", payload)
        self.assertIsNotNone(payload["ai_background_debug"])
        self.assertGreater(np.count_nonzero(alpha < 250), 0)

    def test_autopilot_recommends_ai_for_high_risk_white_art(self) -> None:
        decision = autopilot_route(_high_risk_white_art())

        self.assertEqual("white_background_complex", decision["case_type"])
        self.assertEqual("ai_background", decision["recommended_mode"])
        self.assertTrue(decision["needs_manual_review"])
        self.assertEqual("red", decision["traffic_light"])


def _simple_artwork() -> Image.Image:
    img = Image.new("RGBA", (160, 140), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((40, 30, 125, 110), radius=18, fill=(230, 60, 140, 255), outline=(15, 15, 15, 255), width=4)
    draw.rectangle((68, 60, 96, 78), fill=(255, 255, 255, 255), outline=(20, 20, 20, 255), width=2)
    return img


def _high_risk_white_art() -> Image.Image:
    img = Image.new("RGBA", (700, 700), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    draw.text((190, 70), "PINK", fill=(20, 20, 20, 255))
    draw.rounded_rectangle((160, 140, 540, 540), radius=80, fill=(230, 70, 145, 255), outline=(10, 10, 10, 255), width=9)
    draw.rectangle((270, 280, 430, 360), fill=(255, 255, 255, 255), outline=(10, 10, 10, 255), width=5)
    for idx in range(120):
        x = 60 + (idx * 23) % 580
        y = 100 + (idx * 41) % 520
        color = (245, 245, 245, 255) if idx % 3 == 0 else (220, 45 + idx % 120, 150, 255)
        draw.ellipse((x, y, x + 4 + idx % 8, y + 4 + idx % 8), fill=color)
    for idx in range(48):
        x = 80 + (idx * 27) % 560
        y = 120 + (idx * 31) % 500
        draw.line((x, y, x + 120, y + 30), fill=(20, 20, 20, 255), width=3)
    return img


def _ai_settings() -> PipelineSettings:
    return PipelineSettings(
        mode_key="ai_background",
        use_ai=True,
        remove_black=False,
        remove_color=False,
        clean_enabled=True,
        trim=False,
        alpha_cut=35,
        despeckle_area=2,
        edge_contract=0,
        black_threshold=24,
        black_level="normal",
        color_tolerance=42,
        protect_details=True,
        protect_white_details=False,
        white_protection_level="maxima",
        fine_detail_level="maxima",
        safe_mode=False,
        enable_dtf_prepress=False,
        remove_white_halo=False,
        remove_black_halo=False,
        halo_strength="suave",
        expand_edge_px=0,
        bleed_px=0,
        create_cutline=False,
        min_printable_mm=1.0,
        logo_detect_colors=False,
        logo_reduce_colors=False,
        logo_black_to_transparent=False,
        logo_white_to_transparent=False,
        logo_unify_colors=False,
        logo_separate_colors=False,
        logo_export_layers=False,
        logo_max_colors=8,
        logo_color_tolerance=24,
        max_ai_side=1200,
        upscale=1,
        dpi=300,
        width_cm=0,
        height_cm=0,
    )


if __name__ == "__main__":
    unittest.main()
