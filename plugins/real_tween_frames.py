"""
Real plugin: /tween-frames
Uses the local Pixel Art RIFE RGBA interpolation package.

Install: pip install torch numpy Pillow
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "pixel_rife_rgba_integration_package"
MODEL_DIR = PACKAGE_DIR / "model"
WEIGHTS_PATH = MODEL_DIR / "rife_rgba_best.pth"

_engine = None


async def _run_fake(data: dict) -> dict:
    try:
        from plugins.fake_tween_frames import run as fake_run
    except ModuleNotFoundError:
        from api.plugins.fake_tween_frames import run as fake_run
    return await fake_run(data)


def _get_engine():
    global _engine
    if _engine is None:
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"RIFE RGBA weights not found: {WEIGHTS_PATH}")
        if str(MODEL_DIR) not in sys.path:
            sys.path.insert(0, str(MODEL_DIR))
        try:
            from pixel_tween_engine import PixelTweenEngine
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Pixel RIFE RGBA dependencies are missing. "
                "Install them with: pip install torch numpy Pillow"
            ) from exc
        _engine = PixelTweenEngine(weights_path=WEIGHTS_PATH, enforce_size=64)
    return _engine


def _decode_hex_frame(frame: dict) -> tuple[int, int, np.ndarray] | None:
    try:
        w = int(frame.get("width") or frame.get("w") or 0)
        h = int(frame.get("height") or frame.get("h") or 0)
        hex_str = str(frame.get("hex") or frame.get("pixel_hex") or "")
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0 or not hex_str:
        return None

    hex_clean = "".join(ch for ch in hex_str if ch in "0123456789abcdefABCDEF")
    expected_chars = w * h * 8
    if len(hex_clean) != expected_chars:
        return None

    raw = bytes.fromhex(hex_clean)
    rgba = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4)).copy()
    return w, h, rgba


def _resize_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(rgba, "RGBA")
    img = img.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8).copy()


def _encode_hex_frame(w: int, h: int, rgba: np.ndarray) -> dict:
    return {
        "hex": rgba.astype(np.uint8, copy=False).tobytes().hex().upper(),
        "width": w,
        "height": h,
    }


async def run(data: dict) -> dict:
    current_raw = data.get("current_frame") or {}
    next_raw = data.get("next_frame") or {}
    if data.get("test_mode"):
        return await _run_fake(data)

    current_decoded = _decode_hex_frame(current_raw)
    next_decoded = _decode_hex_frame(next_raw)
    if current_decoded is None or next_decoded is None:
        return {
            "frames": [],
            "confidence": 0.0,
            "model": "Pixel_RIFE_RGBA_v1",
            "error": "Could not decode one or both input frames.",
        }

    w, h, cur_rgba = current_decoded
    w2, h2, next_rgba = next_decoded
    if (w, h) != (w2, h2):
        return {
            "frames": [_encode_hex_frame(w, h, cur_rgba)],
            "confidence": 0.10,
            "model": "Pixel_RIFE_RGBA_v1",
            "warning": "Frame dimensions differ; returning current frame unchanged.",
        }
    if (w, h) != (64, 64):
        return await _run_fake(data)

    try:
        engine = _get_engine()
        pred = engine.predict(cur_rgba, next_rgba)
    except Exception:
        log.exception("Pixel RIFE RGBA tween failed")
        raise

    return {
        "frames": [_encode_hex_frame(w, h, pred)],
        "confidence": 0.95,
        "model": "Pixel_RIFE_RGBA_v1",
    }
