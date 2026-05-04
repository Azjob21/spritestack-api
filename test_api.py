"""
Test suite for the SpriteStack Studio AI API
=============================================
Run with:  pytest tests/test_api.py -v

Covers all four endpoints with:
  - Happy-path checks (correct structure, types, value ranges)
  - Edge-case inputs (empty prompt, blank audio, mismatched frame sizes)
  - Fake-plugin specific paths (parse-scene no-objects, tween confidence modes)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import pytest

# Make the parent directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


# ===========================================================================
# /health
# ===========================================================================

class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "plugins" in body

    def test_health_all_endpoints_present(self):
        r = client.get("/health")
        plugins = r.json()["plugins"]
        for key in ("parse-scene", "transcribe", "generate-sprite", "tween-frames"):
            assert key in plugins

    def test_health_generate_sprite_mode_tracks_local_model(self):
        r = client.get("/health")
        plugins = r.json()["plugins"]
        model_dir = Path(__file__).resolve().parent / "models" / "pixel-art-model"
        real_generation_enabled = os.getenv("SPRITESTACK_REAL_GENERATION", "1").strip().lower() not in {
            "0",
            "false",
            "off",
        }
        force_real_generation = os.getenv("SPRITESTACK_FORCE_REAL_GENERATION", "0").strip().lower() in {
            "1",
            "true",
            "on",
        }
        python_supports_local_model = sys.version_info < (3, 12)
        expected_mode = (
            "real"
            if (
                model_dir.is_dir()
                and real_generation_enabled
                and (python_supports_local_model or force_real_generation)
            )
            else "fake"
        )
        assert plugins["generate-sprite"]["mode"] == expected_mode


# ===========================================================================
# /parse-scene
# ===========================================================================

class TestParseScene:
    def _post(self, prompt: str) -> dict:
        return client.post("/parse-scene", json={"prompt": prompt}).json()

    def test_returns_objects_list(self):
        body = self._post("tree on the left, rock on the right")
        assert "objects" in body
        assert isinstance(body["objects"], list)

    def test_object_fields(self):
        body = self._post("knight in the center")
        for obj in body["objects"]:
            assert "name" in obj
            assert "type" in obj
            assert "x" in obj
            assert "y" in obj

    def test_object_type_is_valid(self):
        body = self._post("tree on the left, castle in the background")
        valid_types = {"stack", "sprite", "texture"}
        for obj in body["objects"]:
            assert obj["type"] in valid_types

    def test_xy_in_unit_range(self):
        body = self._post("house on the right, river at the bottom")
        for obj in body["objects"]:
            assert 0.0 <= obj["x"] <= 1.0
            assert 0.0 <= obj["y"] <= 1.0

    def test_empty_scene_prompt(self):
        body = self._post("empty scene")
        assert body["objects"] == []

    def test_blank_prompt_returns_422(self):
        r = client.post("/parse-scene", json={"prompt": "   "})
        assert r.status_code == 422

    def test_missing_prompt_returns_422(self):
        r = client.post("/parse-scene", json={})
        assert r.status_code == 422

    def test_unknown_objects_fallback_to_sprite(self):
        body = self._post("xyzzy in the middle")
        assert len(body["objects"]) >= 1
        assert body["objects"][0]["name"] == "Sprite"

    def test_multiple_objects_parsed(self):
        body = self._post("tree on the left, rock on the right, cloud at the top")
        assert len(body["objects"]) >= 2

    def test_position_keywords_honoured(self):
        body = self._post("tree on the left")
        tree = next((o for o in body["objects"] if o["name"] == "Tree"), None)
        assert tree is not None
        assert tree["x"] < 0.5  # left side

    def test_right_position_keyword(self):
        body = self._post("castle on the right")
        castle = next((o for o in body["objects"] if o["name"] == "Castle"), None)
        assert castle is not None
        assert castle["x"] > 0.5

    def test_deterministic_for_same_prompt(self):
        body_a = self._post("knight and tree")
        body_b = self._post("knight and tree")
        assert body_a == body_b


# ===========================================================================
# /transcribe
# ===========================================================================

class TestTranscribe:
    def _post_wav(self, content: bytes = b"\x00" * 256, filename: str = "test.wav"):
        return client.post(
            "/transcribe",
            files={"file": (filename, content, "audio/wav")},
        )

    def test_returns_text(self):
        r = self._post_wav()
        assert r.status_code == 200
        body = r.json()
        assert "text" in body
        assert isinstance(body["text"], str)
        assert len(body["text"]) > 0

    def test_text_is_non_empty_string(self):
        r = self._post_wav(b"\xFF" * 512)
        assert r.json()["text"].strip() != ""

    def test_empty_file_returns_400(self):
        r = self._post_wav(content=b"")
        assert r.status_code == 400

    def test_deterministic_for_same_content(self):
        content = b"\xAB\xCD" * 128
        body_a = self._post_wav(content, "voice.wav").json()
        body_b = self._post_wav(content, "voice.wav").json()
        assert body_a["text"] == body_b["text"]

    def test_different_filenames_may_differ(self):
        content = b"\x01" * 64
        # Just ensure both calls succeed — output may or may not differ
        r1 = self._post_wav(content, "a.wav")
        r2 = self._post_wav(content, "b.wav")
        assert r1.status_code == 200
        assert r2.status_code == 200


# ===========================================================================
# /generate-sprite
# ===========================================================================

class TestGenerateSprite:
    def _post(self, prompt: str, width: int = 16, height: int = 16) -> dict:
        return client.post(
            "/generate-sprite",
            json={"prompt": prompt, "width": width, "height": height},
        ).json()

    def test_returns_hex_field(self):
        body = self._post("knight")
        assert "hex" in body

    def test_hex_correct_length(self):
        w, h = 16, 16
        body = self._post("tree", width=w, height=h)
        expected_len = w * h * 8  # 4 bytes × 2 hex chars per byte
        assert len(body["hex"]) == expected_len

    def test_hex_correct_length_32x32(self):
        w, h = 32, 32
        body = self._post("castle", width=w, height=h)
        assert len(body["hex"]) == w * h * 8

    def test_returns_width_height(self):
        body = self._post("rock", width=24, height=24)
        assert body["width"] == 24
        assert body["height"] == 24

    def test_hex_is_valid_hexadecimal(self):
        body = self._post("flower")
        hex_str = body["hex"]
        try:
            int(hex_str, 16)
        except ValueError:
            pytest.fail("hex field contains non-hexadecimal characters")

    def test_different_prompts_produce_different_images(self):
        body_a = self._post("tree")
        body_b = self._post("dragon")
        assert body_a["hex"] != body_b["hex"]

    def test_same_prompt_is_deterministic(self):
        body_a = self._post("hero")
        body_b = self._post("hero")
        assert body_a["hex"] == body_b["hex"]

    def test_empty_prompt_returns_422(self):
        r = client.post("/generate-sprite", json={"prompt": ""})
        assert r.status_code == 422

    def test_missing_prompt_returns_422(self):
        r = client.post("/generate-sprite", json={"width": 16, "height": 16})
        assert r.status_code == 422


# ===========================================================================
# /tween-frames
# ===========================================================================

def _make_frame(w: int = 8, h: int = 8, fill: tuple = (255, 0, 0, 255)) -> dict:
    """Build a hex-encoded solid-colour frame."""
    r, g, b, a = fill
    pixel = f"{r:02X}{g:02X}{b:02X}{a:02X}"
    return {"hex": pixel * (w * h), "width": w, "height": h}


class TestTweenFrames:
    BASE_URL = "/tween-frames"

    def _post(self, payload: dict) -> dict:
        return client.post(self.BASE_URL, json=payload).json()

    def _happy_payload(self, test_mode: str | None = None) -> dict:
        p = {
            "current_frame": _make_frame(8, 8, (255, 0, 0, 255)),
            "next_frame":    _make_frame(8, 8, (0, 0, 255, 255)),
            "num_intermediate": 1,
        }
        if test_mode:
            p["test_mode"] = test_mode
        return p

    def test_returns_frames_list(self):
        body = self._post(self._happy_payload())
        assert "frames" in body
        assert isinstance(body["frames"], list)

    def test_returns_confidence(self):
        body = self._post(self._happy_payload())
        assert "confidence" in body
        assert isinstance(body["confidence"], float)

    def test_confidence_in_range(self):
        body = self._post(self._happy_payload())
        assert 0.0 <= body["confidence"] <= 1.0

    def test_frame_has_hex_width_height(self):
        body = self._post(self._happy_payload())
        frame = body["frames"][0]
        assert "hex" in frame
        assert "width" in frame
        assert "height" in frame

    def test_output_frame_correct_pixel_count(self):
        w, h = 8, 8
        body = self._post(self._happy_payload())
        frame = body["frames"][0]
        assert len(frame["hex"]) == w * h * 8

    def test_midpoint_is_blend_of_inputs(self):
        # Red (255,0,0) blended with Blue (0,0,255) at t=0.5 → (127,0,127)
        body = self._post(self._happy_payload())
        frame = body["frames"][0]
        first_pixel = frame["hex"][:8]
        r = int(first_pixel[0:2], 16)
        g = int(first_pixel[2:4], 16)
        b = int(first_pixel[4:6], 16)
        assert 120 <= r <= 135
        assert g == 0
        assert 120 <= b <= 135

    def test_high_confidence_mode(self):
        body = self._post(self._happy_payload("high_confidence"))
        assert body["confidence"] >= 0.90

    def test_low_confidence_mode(self):
        body = self._post(self._happy_payload("low_confidence"))
        assert body["confidence"] < 0.60

    def test_mismatched_frame_sizes_low_confidence(self):
        payload = {
            "current_frame": _make_frame(8, 8),
            "next_frame":    _make_frame(16, 16),
            "num_intermediate": 1,
        }
        body = self._post(payload)
        # Should degrade gracefully, not crash
        assert "confidence" in body
        assert body["confidence"] < 0.5

    def test_malformed_hex_returns_zero_confidence(self):
        payload = {
            "current_frame": {"hex": "ZZZZZZZZ", "width": 1, "height": 1},
            "next_frame":    {"hex": "ZZZZZZZZ", "width": 1, "height": 1},
            "num_intermediate": 1,
        }
        body = self._post(payload)
        assert body["confidence"] == 0.0

    def test_missing_current_frame_returns_422(self):
        r = client.post(self.BASE_URL, json={"next_frame": _make_frame()})
        assert r.status_code == 422

    def test_missing_next_frame_returns_422(self):
        r = client.post(self.BASE_URL, json={"current_frame": _make_frame()})
        assert r.status_code == 422

    def test_non_json_body_returns_400(self):
        r = client.post(self.BASE_URL, content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400
