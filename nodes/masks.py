"""Gausify Orbit subject-mask nodes: solid-colour key, RMBG-1.4 and SAM 2.1
(the masking methods of web/src/pipeline/masks.ts, makeMasks). Each returns one
MASK per frame (white = subject) and a JSON stats string."""

from __future__ import annotations

import json
import logging
import os
import re

import numpy as np

from .common import CATEGORY, Progress, images_to_u8, interrupted, raise_if_interrupted, run_cancellable

log = logging.getLogger("gausify.nodes")

RMBG_LICENSE_ERROR = (
    "RMBG-1.4 (BRIA AI) is licensed for non-commercial use only. Enable "
    "'accept_noncommercial_license' to confirm you accept the BRIA RMBG-1.4 non-commercial license "
    "(https://huggingface.co/briaai/RMBG-1.4) (or set GAUSIFY_ACCEPT_RMBG_LICENSE=1 for the whole "
    "ComfyUI instance), or use the Solid Colour or SAM 2.1 mask nodes."
)


def rmbg_license_accepted_by_env() -> bool:
    """GAUSIFY_ACCEPT_RMBG_LICENSE=1/true/yes: the operator accepts the RMBG-1.4 license instance-wide."""
    return os.environ.get("GAUSIFY_ACCEPT_RMBG_LICENSE", "").strip().lower() in ("1", "true", "yes")


_HEX = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


# ---------------------------------------------------------------- parsing


def parse_color(s: str) -> tuple[int, int, int]:
    """'#rrggbb', '#rgb' or 'r,g,b' (0-255) -> (r, g, b)."""
    t = (s or "").strip()
    m = _HEX.match(t)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    parts = [p for p in re.split(r"[,\s]+", t) if p]
    if len(parts) == 3:
        try:
            v = [int(p) for p in parts]
        except ValueError:
            v = []
        if len(v) == 3 and all(0 <= c <= 255 for c in v):
            return v[0], v[1], v[2]
    raise ValueError(f"color {s!r}: expected '#rrggbb' or 'r,g,b' (0-255)")


def _floats(text: str, what: str, n: int) -> list[float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != n:
        raise ValueError(f"{what} {text!r}: expected {n} comma-separated numbers")
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{what} {text!r}: not numbers") from None


def _unit(v: float, what: str, text: str) -> float:
    if not 0.0 <= v <= 1.0 or v != v:
        raise ValueError(f"{what} {text!r}: coordinates are normalized and must be between 0 and 1")
    return v


def parse_box(s: str) -> list[float] | None:
    """'x0,y0,x1,y1' (normalized 0-1) -> [x0, y0, x1, y1]; '' -> None."""
    if not (s or "").strip():
        return None
    v = _floats(s, "box", 4)
    for c in v:
        _unit(c, "box", s)
    if v[0] >= v[2] or v[1] >= v[3]:
        raise ValueError(f"box {s!r}: needs x0 < x1 and y0 < y1")
    return v


def parse_points(s: str) -> list[dict]:
    """'x,y,label;x,y,label' (normalized; label 1 = subject, 0 = background)."""
    out = []
    for item in (s or "").split(";"):
        if not item.strip():
            continue
        x, y, label = _floats(item, "point", 3)
        if label not in (0.0, 1.0):
            raise ValueError(f"point {item.strip()!r}: label must be 1 (subject) or 0 (background)")
        out.append({"x": _unit(x, "point", item.strip()), "y": _unit(y, "point", item.strip()), "label": int(label)})
    return out


def sam2_prompt(box: str, points: str) -> dict | None:
    b, p = parse_box(box), parse_points(points)
    return {"box": b, "points": p} if (b or p) else None


# ---------------------------------------------------------------- helpers


def _result(masks: np.ndarray, stats: dict):
    import torch

    return (torch.from_numpy(masks.astype(np.float32) / 255.0), json.dumps(stats))


def _stats(count: int, coverage: float, no_subject: int, backend: str, key=None) -> dict:
    return {"count": count, "meanCoverage": coverage, "noSubject": no_subject, "backend": backend,
            "keyColor": list(key) if key is not None else None}


# ---------------------------------------------------------------- nodes


class _MaskBase:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "stats")


class GausifyMaskChroma(_MaskBase):
    FUNCTION = "mask"
    DESCRIPTION = ("Subject mask by chroma key (gausify-orbit 'Solid colour'): distance to the background colour "
                   "with soft edges and flood fill from the borders. Use with a green or blue screen; enable "
                   "despill on the Train node to remove screen spill from the subject.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "auto_color": ("BOOLEAN", {"default": True, "tooltip": "Detect the background colour from the frame borders."}),
            "color": ("STRING", {"default": "#00ff00", "tooltip": "Background colour (#rrggbb) when auto_color is off."}),
            "tolerance": ("INT", {"default": 12, "min": 0, "max": 100}),
            "softness": ("INT", {"default": 6, "min": 0, "max": 100}),
            "contiguous": ("BOOLEAN", {"default": True,
                                       "tooltip": "Only background connected to the borders is removed (holes in the subject stay)."}),
        }}

    def mask(self, images, auto_color, color, tolerance, softness, contiguous):
        from gausify.masking.chroma import border_key, chroma_mask

        manual = parse_color(color)
        frames = images_to_u8(images)
        n = len(frames)
        progress = Progress()
        masks = np.zeros(frames.shape[:3], np.uint8)
        keys, coverage = [], 0.0
        for i in range(n):
            if interrupted():
                raise_if_interrupted()
            key = border_key(frames[i]) if auto_color else manual
            keys.append(key)
            masks[i] = chroma_mask(frames[i], key, tolerance, softness, contiguous)
            coverage += float(masks[i].sum(dtype=np.float64)) / (255.0 * masks[i].size)
            progress((i + 1) / n, f"solid colour {i + 1}/{n}")
        key_color = tuple(int(sorted(k[c] for k in keys)[len(keys) >> 1]) for c in range(3))
        if auto_color:
            log.info("gausify mask: detected background color: rgb(%d, %d, %d)", *key_color)
        return _result(masks, _stats(n, coverage / n, 0, "Solid color", key_color))


class GausifyMaskRMBG(_MaskBase):
    FUNCTION = "mask"
    DESCRIPTION = ("Subject mask with RMBG-1.4 (BRIA AI; non-commercial license), ONNX Runtime on CUDA. "
                   "The 176 MB model is downloaded on first use.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "accept_noncommercial_license": ("BOOLEAN", {"default": False,
                                             "tooltip": "RMBG-1.4 is licensed by BRIA AI for non-commercial use only. "
                                                        "GAUSIFY_ACCEPT_RMBG_LICENSE=1 accepts it for the whole instance."}),
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
        }}

    def mask(self, images, accept_noncommercial_license, device):
        if not (accept_noncommercial_license or rmbg_license_accepted_by_env()):
            raise ValueError(RMBG_LICENSE_ERROR)
        from gausify.masking.rmbg import rmbg_masks

        masks, stats = run_cancellable(rmbg_masks, images_to_u8(images), Progress(), interrupted, device=device)
        raise_if_interrupted()
        return _result(masks, stats)


class GausifyMaskSAM2(_MaskBase):
    FUNCTION = "mask"
    DESCRIPTION = ("Subject mask with SAM 2.1 tiny video tracking (Meta, Apache-2.0): mark the subject with a "
                   "box and/or points on one frame (default: the object at the centre) and it is tracked "
                   "across the others. Models are downloaded on first use.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "prompt_frame": ("INT", {"default": 0, "min": 0, "max": 100000,
                                     "tooltip": "Index of the frame the box/points refer to."}),
            "box": ("STRING", {"default": "", "tooltip": "'x0,y0,x1,y1' normalized 0-1 (e.g. 0.25,0.1,0.75,0.9)."}),
            "points": ("STRING", {"default": "", "tooltip": "'x,y,label;...' normalized 0-1; label 1 = subject, "
                                  "0 = background (e.g. 0.5,0.5,1;0.1,0.1,0). Empty box and points: the object at the centre."}),
            "variant": (["fp32", "fp16"], {"default": "fp32"}),
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
        }}

    def mask(self, images, prompt_frame, box, points, variant, device):
        from gausify.masking.sam2 import sam2_masks

        prompt = sam2_prompt(box, points)
        frames = images_to_u8(images)
        n = len(frames)
        start = int(prompt_frame)
        if not 0 <= start < n:
            raise ValueError(f"prompt_frame {start} is outside the {n} frames (0..{n - 1})")
        log.info("gausify mask: SAM 2.1 %s from frame %d", "with prompt" if prompt else "on the centre object", start)
        masks, stats = run_cancellable(
            sam2_masks, frames, prompt, start, variant, Progress(), interrupted, device=device,
        )
        raise_if_interrupted()
        return _result(masks, stats)
