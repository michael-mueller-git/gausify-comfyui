"""Pure parts of SAM 2.1 video tracking (no ONNX Runtime): constants, memory-bank
parameters, layout conversions, mask post-processing and the prompt.

Port of gausify-orbit `web/src/pipeline/sam2-math.ts`. Reference values come from
Transformers 4.57 (facebook/sam2.1-hiera-tiny).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gausify.alpha import _js_round

SIZE = 1024  # encoder input
LOW = 256  # decoder low-res masks
TOKENS = 64 * 64  # positions in the lowest-resolution feature map
HIDDEN = 256
MEM = 64
NUM_MASKMEM = 7  # the current frame + 6 memories
MAX_POINTERS = 16
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

PARAMS_FILE = Path(__file__).with_name("assets") / "sam2-params.bin"


@dataclass
class Params:
    no_memory: np.ndarray  # (256,)
    tpos: np.ndarray  # (7, 64)
    proj_w: np.ndarray  # (64, 256)
    proj_b: np.ndarray  # (64,)


def parse_params(buf: bytes) -> Params:
    """sam2-params.bin: no_memory_embedding (256), memory_temporal_positional_encoding
    (7 x 64), weight (64 x 256) and bias (64) of the pointer temporal projection, f32 LE."""
    a = np.frombuffer(buf, dtype="<f4").astype(np.float32)
    sizes = [HIDDEN, NUM_MASKMEM * MEM, MEM * HIDDEN, MEM]
    if a.size != sum(sizes):
        raise ValueError("SAM 2.1: parameters have unexpected size")
    o = np.cumsum([0] + sizes)
    return Params(
        no_memory=a[o[0] : o[1]],
        tpos=a[o[1] : o[2]].reshape(NUM_MASKMEM, MEM),
        proj_w=a[o[2] : o[3]].reshape(MEM, HIDDEN),
        proj_b=a[o[3] : o[4]],
    )


def load_params() -> Params:
    return parse_params(PARAMS_FILE.read_bytes())


def round_bf16(a: np.ndarray) -> np.ndarray:
    """Rounds float32 values to bf16, nearest even (Transformers stores the memories
    that way). Returns a new float32 array."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).reshape(np.shape(a))


def pointer_pos_enc(t: float, p: Params) -> np.ndarray:
    """Pointer temporal encoding: 1D sine of dim 256 -> Linear(256, 64). float32 (64,)."""
    half = HIDDEN // 2
    i = np.arange(half)
    e = t / 10000.0 ** ((2 * (i // 2)) / half)
    pe = np.concatenate([np.sin(e), np.cos(e)]).astype(np.float32)
    out = p.proj_b.astype(np.float64) + p.proj_w.astype(np.float64) @ pe.astype(np.float64)
    return out.astype(np.float32)


_FIX_AT = 10137
_FIX_WANT = bytes([0x0A, 0x0A, 0x08, 0x08, 0x07, 0x12, 0x04, 0x0A, 0x02, 0x08, 0x05])


def fix_vision_shape(model: bytes) -> bytes:
    """The fp32 encoder export declares the output of an If's `then` branch (never run
    at 1024^2) with 5 dims where shape inference finds 4. Patches that one byte of the
    model bytes (size unchanged; the file's SHA-256 has already been checked) and returns
    the patched copy. Idempotent; ValueError on an unexpected layout."""
    b = bytearray(model)
    seg = bytes(b[_FIX_AT : _FIX_AT + len(_FIX_WANT)])
    if len(seg) == len(_FIX_WANT) and seg[10] == 0x04 and seg[:10] == _FIX_WANT[:10]:
        return bytes(b)  # already patched
    if seg != _FIX_WANT:
        raise ValueError("SAM 2.1: encoder has unexpected layout")
    b[_FIX_AT + 10] = 0x04
    return bytes(b)


def to_seq(nchw: np.ndarray, c: int) -> np.ndarray:
    """NCHW (1, C, 64, 64) -> (4096, 1, C)."""
    return np.ascontiguousarray(np.asarray(nchw, np.float32).reshape(c, TOKENS).T.reshape(TOKENS, 1, c))


def to_nchw(seq: np.ndarray, c: int) -> np.ndarray:
    """(4096, C) -> NCHW (1, C, 64, 64)."""
    return np.ascontiguousarray(np.asarray(seq, np.float32).reshape(TOKENS, c).T.reshape(1, c, 64, 64))


def _interp_axis(n_out: int):
    s = np.clip((np.arange(n_out) + 0.5) * LOW / n_out - 0.5, 0, LOW - 1)
    i0 = np.minimum(np.floor(s).astype(np.int64), LOW - 2)
    return i0, s - i0


def logits_to_mask(logits: np.ndarray, W: int, H: int) -> np.ndarray:
    """256^2 logits -> uint8 mask (sigmoid) at frame size (W x H), bilinear like the
    F.interpolate(align_corners=False) in Transformers' post-processing."""
    lg = np.asarray(logits, np.float64).reshape(LOW, LOW)
    x0, fx = _interp_axis(W)
    y0, fy = _interp_axis(H)
    top_bot = lg[:, x0] * (1 - fx) + lg[:, x0 + 1] * fx  # (LOW, W)
    v = top_bot[y0] * (1 - fy)[:, None] + top_bot[y0 + 1] * fy[:, None]
    return _js_round(255 / (1 + np.exp(-v))).astype(np.uint8)


def prompt_frame(n_frames: int, index: int = 0) -> int:
    """Prompt frame: the requested index, clamped into [0, n_frames - 1]."""
    return min(max(int(index), 0), max(n_frames - 1, 0))


def prompt_pixels(prompt: dict | None, w: int, h: int) -> dict:
    """Prompt in pixels: {"box": [x0, y0, x1, y1] | None, "points": [{"x","y","label"}]}
    from the normalized one. If unmarked, a point at the centre (orbit videos circle
    the subject)."""
    if not prompt or (not prompt.get("box") and not prompt.get("points")):
        return {"box": None, "points": [{"x": w / 2, "y": h / 2, "label": 1}]}
    box = prompt.get("box")
    return {
        "box": [box[0] * w, box[1] * h, box[2] * w, box[3] * h] if box else None,
        "points": [{"x": p["x"] * w, "y": p["y"] * h, "label": p["label"]} for p in prompt.get("points", [])],
    }
