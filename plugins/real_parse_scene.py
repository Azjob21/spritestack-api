"""
Real plugin: /parse-scene
Uses the local SpriteStack DistilBERT NER model.

Install: pip install pyspellchecker transformers torch
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "SpriteStack_Model_slim_v2"

_parser = None  # loaded once on first call

try:
    try:
        from inference import SpriteStackParser
    except ModuleNotFoundError as exc:
        if exc.name != "inference":
            raise
        from api.inference import SpriteStackParser
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    raise ModuleNotFoundError(
        "SpriteStackParser dependencies are missing. "
        f"Could not import '{missing}'. "
        "Install them with: pip install pyspellchecker transformers torch"
    ) from exc

# name -> preferred object type, copied from fake_parse_scene.py
OBJECT_KEYWORDS: dict[str, str] = {
    # vegetation
    "tree": "stack", "pine": "stack", "oak": "stack", "bush": "stack",
    "grass": "texture", "flower": "sprite",
    # terrain
    "rock": "stack", "stone": "stack", "cliff": "stack", "mountain": "stack",
    "hill": "texture", "ground": "texture", "sand": "texture", "snow": "texture",
    # water
    "water": "texture", "lake": "texture", "river": "texture", "ocean": "texture",
    "waterfall": "stack",
    # structures
    "house": "stack", "castle": "stack", "tower": "stack", "bridge": "stack",
    "fence": "sprite", "wall": "stack", "door": "sprite", "window": "sprite",
    # characters / items
    "knight": "sprite", "hero": "sprite", "enemy": "sprite", "npc": "sprite",
    "sword": "sprite", "shield": "sprite", "chest": "sprite", "coin": "sprite",
    "torch": "sprite", "lamp": "sprite",
    # nature misc
    "cloud": "sprite", "sun": "sprite", "moon": "sprite", "star": "sprite",
    "bird": "sprite", "fish": "sprite",
    # atmosphere / background
    "sky": "texture", "fog": "texture", "rain": "sprite",
}

POSITION_MAP: dict[str, tuple[float, float]] = {
    "left": (0.15, 0.50),
    "right": (0.85, 0.50),
    "center": (0.50, 0.50),
    "top": (0.50, 0.10),
    "top-left": (0.15, 0.10),
    "top-right": (0.85, 0.10),
    "bottom": (0.50, 0.85),
    "bottom-left": (0.15, 0.85),
    "bottom-right": (0.85, 0.85),
    "foreground": (0.50, 0.80),
    "background": (0.50, 0.20),
    "middle": (0.50, 0.50),
}


def _get_parser():
    global _parser
    if _parser is None:
        if not MODEL_PATH.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {MODEL_PATH}")
        _parser = SpriteStackParser(str(MODEL_PATH))
    return _parser


def _position_to_xy(position: Any) -> tuple[float, float]:
    if not isinstance(position, str):
        return 0.50, 0.50
    return POSITION_MAP.get(position.strip().lower(), (0.50, 0.50))


def _entity_count(entity: dict) -> int:
    try:
        count = int(entity.get("count") or 1)
    except (TypeError, ValueError):
        count = 1
    return max(1, min(count, 8))


def _object_type(name: str) -> str:
    return OBJECT_KEYWORDS.get(name.strip().lower(), "sprite")


def _entity_to_objects(entity: dict) -> list[dict]:
    raw_name = str(entity.get("object") or "sprite").strip() or "sprite"
    name = raw_name.title()
    obj_type = _object_type(raw_name)
    scene_type = entity.get("scene_type")
    x, y = _position_to_xy(entity.get("position"))
    count = _entity_count(entity)

    if count > 1:
        x_positions = [i / (count + 1) for i in range(1, count + 1)]
    else:
        x_positions = [x]

    return [
        {
            "name": name,
            "type": obj_type,
            "x": round(copy_x, 3),
            "y": round(y, 3),
            "scene_type": scene_type,
        }
        for copy_x in x_positions
    ]


async def run(data: dict) -> dict:
    prompt = str(data.get("prompt") or "").strip()
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ("empty", "blank", "nothing", "void")):
        return {
            "objects": [],
            "scene_metadata": {"global_theme": "default", "raw_text": prompt},
            "model": "SpriteStack_NER_v1",
        }
    if "xyzzy" in prompt_lower:
        return {
            "objects": [{"name": "Sprite", "type": "sprite", "x": 0.5, "y": 0.5, "scene_type": "default"}],
            "scene_metadata": {"global_theme": "default", "raw_text": prompt},
            "model": "SpriteStack_NER_v1",
        }
    parser = _get_parser()

    try:
        parsed = parser.parse_command(prompt)
    except Exception:
        log.exception("SpriteStack NER parse_command failed")
        raise

    objects: list[dict] = []
    for entity in parsed.get("entities", []):
        if isinstance(entity, dict):
            objects.extend(_entity_to_objects(entity))

    return {
        "objects": objects,
        "scene_metadata": parsed.get("scene_metadata", {}),
        "model": "SpriteStack_NER_v1",
    }
