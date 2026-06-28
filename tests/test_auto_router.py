from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from core.auto_router import autopilot_quality_check, autopilot_route
from core.pipeline import PipelineSettings, process_artwork


class AutoPilotRouterTests(unittest.TestCase):
    def test_png_transparent_routes_to_transparent_png(self) -> None:
        decision = autopilot_route(_transparent_png())

        self.assertEqual("png_transparent", decision["case_type"])
        self.assertEqual("transparent_png", decision["recommended_mode"])

    def test_white_background_simple_routes_to_color_bg(self) -> None:
        decision = autopilot_route(_simple_white_background())

        self.assertEqual("white_background_simple", decision["case_type"])
        self.assertEqual("color_bg", decision["recommended_mode"])

    def test_white_complex_art_routes_to_complex_white_bg(self) -> None:
        decision = autopilot_route(_complex_white_artwork())

        self.assertEqual("white_background_complex", decision["case_type"])
        self.assertEqual("complex_white_bg", decision["recommended_mode"])
        self.assertEqual("red", decision["traffic_light"])

    def test_pink_panther_detector_payload_routes_to_complex_white(self) -> None:
        decision = autopilot_route(Image.new("RGBA", (1200, 1600), (240, 240, 240, 255)), _pink_panther_detection())

        self.assertEqual("white_background_complex", decision["case_type"])
        self.assertEqual("complex_white_bg", decision["recommended_mode"])
        self.assertEqual("red", decision["traffic_light"])
        self.assertTrue(decision["needs_manual_review"])
        self.assertIn("Fondo blanco complejo", str(decision["suggested_next_step"]))

    def test_near_white_complex_synthetic_routes_to_complex_white(self) -> None:
        decision = autopilot_route(_near_white_complex_artwork())

        self.assertEqual("white_background_complex", decision["case_type"])
        self.assertEqual("complex_white_bg", decision["recommended_mode"])
        self.assertEqual("red", decision["traffic_light"])

    def test_black_background_routes_to_black_bg(self) -> None:
        decision = autopilot_route(_black_background_artwork())

        self.assertEqual("black_background", decision["case_type"])
        self.assertEqual("black_bg", decision["recommended_mode"])

    def test_ambiguous_case_is_not_green(self) -> None:
        decision = autopilot_route(_ambiguous_artwork())

        self.assertNotEqual("green", decision["traffic_light"])
        self.assertTrue(decision["needs_manual_review"])

    def test_high_risk_art_requires_review(self) -> None:
        decision = autopilot_route(_complex_white_artwork())

        self.assertEqual("alto", decision["risk_level"])
        self.assertTrue(decision["needs_manual_review"])

    def test_quality_check_detects_light_halo_on_black_review(self) -> None:
        qa = autopilot_quality_check(_halo_artwork())

        self.assertEqual("red", qa["traffic_light"])
        self.assertGreater(qa["halo_light_score"], 0)

    def test_auto_pipeline_uses_autopilot_for_complex_white(self) -> None:
        payload = process_artwork(_complex_white_artwork(), {}, _auto_settings())

        self.assertEqual("complex_white_bg", payload["autopilot"]["recommended_mode"])
        self.assertIsNotNone(payload["complex_white_debug"])


def _transparent_png() -> Image.Image:
    img = Image.new("RGBA", (620, 620), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((180, 180, 440, 440), fill=(230, 40, 130, 255))
    return img


def _simple_white_background() -> Image.Image:
    img = Image.new("RGBA", (620, 620), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((200, 180, 430, 430), fill=(230, 40, 130, 255))
    return img


def _complex_white_artwork() -> Image.Image:
    img = Image.new("RGBA", (640, 640), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 130, 490, 500), fill=(230, 70, 145, 255), outline=(15, 15, 15, 255), width=8)
    draw.rectangle((250, 270, 390, 340), fill=(255, 255, 255, 255), outline=(15, 15, 15, 255), width=5)
    for idx in range(36):
        x = 90 + (idx * 13) % 460
        y = 80 + (idx * 29) % 470
        color = (40 + idx * 5 % 200, 30 + idx * 9 % 180, 220 - idx * 3 % 170, 255)
        draw.line((x, y, min(620, x + 70), min(620, y + 30)), fill=color, width=3)
    return img


def _near_white_complex_artwork() -> Image.Image:
    img = Image.new("RGBA", (700, 700), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((170, 140, 530, 540), radius=90, fill=(230, 70, 145, 255), outline=(12, 12, 12, 255), width=9)
    draw.rectangle((270, 285, 430, 355), fill=(255, 255, 255, 255), outline=(12, 12, 12, 255), width=5)
    draw.text((235, 90), "PINK", fill=(20, 20, 20, 255))
    for idx in range(70):
        x = 70 + (idx * 19) % 560
        y = 80 + (idx * 37) % 540
        radius = 2 + idx % 7
        color = (245, 245, 245, 255) if idx % 3 == 0 else (220, 50 + idx % 120, 150, 255)
        draw.ellipse((x, y, x + radius, y + radius), fill=color)
    for idx in range(28):
        x = 110 + (idx * 23) % 480
        y = 120 + (idx * 31) % 500
        draw.line((x, y, x + 95, y + 25), fill=(25, 25, 25, 255), width=2)
    return img


def _black_background_artwork() -> Image.Image:
    img = Image.new("RGBA", (620, 620), (0, 0, 0, 255))
    ImageDraw.Draw(img).rectangle((190, 190, 430, 430), fill=(240, 80, 140, 255))
    return img


def _pink_panther_detection() -> dict[str, object]:
    return {
        "type": "Fondo de color",
        "recommended_mode": "color_bg",
        "use_ai": False,
        "dominant_color": "#f0f0f0",
        "white_percent": 50.85,
        "black_percent": 13.68,
        "background_uniformity": 81.64,
        "edge_density": 9.92,
        "color_count": 157.0,
        "logo_score": 39.85,
        "text_score": 100.0,
        "splash_score": 52.43,
        "noise_score": 16.61,
        "transparency_percent": 0.0,
        "confidence": 90.82,
        "background": "color dominante",
    }


def _ambiguous_artwork() -> Image.Image:
    img = Image.new("RGBA", (620, 620), (210, 210, 210, 255))
    draw = ImageDraw.Draw(img)
    for idx in range(42):
        x = 60 + (idx * 17) % 500
        y = 60 + (idx * 31) % 500
        draw.arc((x, y, x + 90, y + 70), 0, 270, fill=(20, 20, 20, 255), width=2)
    return img


def _halo_artwork() -> Image.Image:
    img = Image.new("RGBA", (620, 620), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((170, 170, 450, 450), fill=(245, 245, 245, 150))
    draw.ellipse((190, 190, 430, 430), fill=(230, 40, 120, 255))
    return img


def _auto_settings() -> PipelineSettings:
    return PipelineSettings(
        mode_key="auto",
        use_ai=False,
        remove_black=False,
        remove_color=True,
        clean_enabled=True,
        trim=False,
        alpha_cut=58,
        despeckle_area=2,
        edge_contract=0,
        black_threshold=24,
        black_level="normal",
        color_tolerance=44,
        protect_details=True,
        protect_white_details=True,
        white_protection_level="maxima",
        fine_detail_level="maxima",
        safe_mode=True,
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
