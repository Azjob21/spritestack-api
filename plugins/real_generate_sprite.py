"""
Real plugin: /generate-sprite
Uses PublicPrompts/All-In-One-Pixel-Model (SD 1.5)
Install: pip install diffusers transformers accelerate torch Pillow
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image

_pipe = None  # loaded once on first call

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pixel-art-model"

def _get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import StableDiffusionPipeline
        import torch
        if not MODEL_PATH.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {MODEL_PATH}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        _pipe  = StableDiffusionPipeline.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=dtype,
            local_files_only=True,   # never tries to hit the internet
        ).to(device)
    return _pipe


def _image_to_hex(img: Image.Image, width: int, height: int) -> str:
    img = img.convert("RGBA")
    
    # if target is tiny (≤32px), step down in halves for cleaner pixels
    if width <= 32:
        intermediate = 256
        img = img.resize((intermediate, intermediate), Image.LANCZOS)
        img = img.resize((width, height), Image.NEAREST)
    else:
        img = img.resize((width, height), Image.LANCZOS)
    
    pixels = list(img.getdata())
    return "".join(f"{r:02X}{g:02X}{b:02X}{a:02X}" for r, g, b, a in pixels)


async def run(data: dict) -> dict:
    import torch

    prompt: str = (data.get("prompt") or "sprite").strip()
    width:  int = max(1, int(data.get("width")  or 16))
    height: int = max(1, int(data.get("height") or 16))

    # generate at 512x512 — SD 1.5 was trained at this resolution
    # downscaling to 16/32 happens after
    GEN_SIZE = 512

    full_prompt = (
        f"pixelsprite, {prompt}, "
        "pixel art, 16-bit, clean sprite, simple background, "
        "centered character, flat colors, sharp edges, game asset"
    )
    negative = (
        "blurry, noisy, photorealistic, 3d render, watermark, "
        "text, signature, gradient background, complex background, "
        "jpeg artifacts, deformed, extra limbs, ugly, low quality"
    )

    pipe = _get_pipe()
    result = pipe(
        full_prompt,
        negative_prompt=negative,
        width=GEN_SIZE,
        height=GEN_SIZE,
        num_inference_steps=40,   # was 25, needs more steps
        guidance_scale=9.0,       # was 7.5, higher = follows prompt more
        generator=torch.manual_seed(42),  # reproducible
    )
    image = result.images[0]

    hex_str = _image_to_hex(image, width, height)
    return {
        "hex":    hex_str,
        "width":  width,
        "height": height,
        "prompt": prompt,
        "model":  "All-In-One-Pixel-Model",
    }
