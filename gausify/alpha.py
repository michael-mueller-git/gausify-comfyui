"""Transparent-background (alpha) training dataset.

Ports of gausify-orbit:
- web/src/pipeline/erode.ts      (erode, ALPHA_ERODE)
- web/src/pipeline/decontam.ts   (decontaminate)
- web/src/pipeline/chroma.ts     (backgroundColor, despillFor, applyDespill)
- web/src/pipeline/masks.ts      (buildAlpha)
- web/src/host/job.ts            (buildAlphaDataset)

Per frame: A = eroded mask (straight alpha), edge colors unmixed from the
background (decontaminate, using the raw mask and the eroded alpha), then,
with a solid-color key, despill on the subject. Written as RGBA PNGs; the
COLMAP model is copied with images.bin renamed to `<stem>.png`.

Images are uint8 HxWx3 RGB arrays, masks uint8 HxW (255 = subject).
"""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .colmap_io import rename_images
from .types import OrbitCameras

#: Radius used in the training dataset (see ALPHA_ERODE in masks.rs).
ALPHA_ERODE = 1

# decontam.ts constants
BLOCK = 4
BLUR_RADIUS = 3
BLUR_PASSES = 2
BACKGROUND_MAX = 5
SOLID = 255
UNMIX_MIN = 0.25

LUMA = (0.2126, 0.7152, 0.0722)


def _js_round(x):
    """Math.round (halves toward +infinity), elementwise."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


# ------------------------------------------------------------------ erode


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Minimum over a (2r+1)^2 window, separable, with the border replicated."""
    mask = np.asarray(mask, dtype=np.uint8)
    if r <= 0:
        return mask.copy()
    h, w = mask.shape
    p = np.pad(mask, ((0, 0), (r, r)), mode="edge")
    out = p[:, 0:w].copy()
    for k in range(1, 2 * r + 1):
        np.minimum(out, p[:, k : k + w], out=out)
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    out = p[0:h].copy()
    for k in range(1, 2 * r + 1):
        np.minimum(out, p[k : k + h], out=out)
    return out


# ------------------------------------------------------------ decontaminate


def _box_blur(src: np.ndarray, axis: int) -> np.ndarray:
    """Box blur of radius BLUR_RADIUS along `axis` (0 = y, 1 = x), edges clamped."""
    r = BLUR_RADIUS
    n = src.shape[axis]
    pad = [(0, 0)] * src.ndim
    pad[axis] = (r, r)
    p = np.pad(src, pad, mode="edge")
    out = np.zeros_like(src)
    for k in range(2 * r + 1):
        out += np.take(p, np.arange(k, k + n), axis=axis)
    return out


def _field(rgb: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """Normalized average color of the picked pixels as a blurred grid
    (gh, gw, 3) float32; NaN where no picked pixel is nearby."""
    h, w = pick.shape
    gw, gh = math.ceil(w / BLOCK), math.ceil(h / BLOCK)
    # Sums of integers stay exact in float32 (< 2^24), as in the reference.
    s = np.zeros((gh * BLOCK, gw * BLOCK, 4), dtype=np.float64)
    s[:h, :w, :3] = np.where(pick[..., None], rgb[..., :3], 0)
    s[:h, :w, 3] = pick
    s = s.reshape(gh, BLOCK, gw, BLOCK, 4).sum(axis=(1, 3))
    for _ in range(BLUR_PASSES):
        s = _box_blur(s, 1)
        s = _box_blur(s, 0)
    s = s.astype(np.float32)
    wgt = s[..., 3:4]
    with np.errstate(invalid="ignore", divide="ignore"):
        color = np.where(wgt > 1e-3, s[..., :3] / wgt, np.nan).astype(np.float32)
    return color


def _grid_lookup(color: np.ndarray, h: int, w: int) -> np.ndarray:
    gh, gw = color.shape[:2]
    gy = np.minimum(np.arange(h) // BLOCK, gh - 1)
    gx = np.minimum(np.arange(w) // BLOCK, gw - 1)
    return color[gy[:, None], gx[None, :]]


def decontaminate(rgb: np.ndarray, mask: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """Replaces the color of the pixels whose training alpha is partial
    (0 < alpha < 255) with the subject color unmixed from the background:
    F = (I - (1 - a) B) / a, B estimated from clear background nearby; below
    alpha 0.25, the solid subject color nearby. `mask`: raw matte; `alpha`:
    training alpha (default: erode(mask, ALPHA_ERODE)). Returns a new uint8
    RGB (alpha channel kept if `rgb` is RGBA)."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    alpha = erode(mask, ALPHA_ERODE) if alpha is None else np.asarray(alpha, dtype=np.uint8)
    h, w = mask.shape
    bg = _grid_lookup(_field(rgb, mask <= BACKGROUND_MAX), h, w)
    solid = _grid_lookup(_field(rgb, mask == SOLID), h, w)
    out = rgb.copy()
    a = alpha.astype(np.float64)
    m = a / 255
    has_bg = ~np.isnan(bg[..., 0])
    has_solid = ~np.isnan(solid[..., 0])
    low = m < UNMIX_MIN
    use_solid = has_solid & low
    todo = (alpha != 0) & (alpha != 255) & (has_bg | use_solid)
    if not todo.any():
        return out
    mm = np.maximum(m, UNMIX_MIN)[todo][:, None]
    I = rgb[..., :3][todo].astype(np.float64)
    with np.errstate(invalid="ignore"):
        unmixed = (I - (1 - mm) * bg[todo].astype(np.float64)) / mm
    val = np.where(use_solid[todo][:, None], solid[todo].astype(np.float64), unmixed)
    out[..., :3][todo] = np.clip(_js_round(val), 0, 255).astype(np.uint8)  # Uint8ClampedArray clamps
    return out


# ----------------------------------------------------------------- despill


def background_color(samples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, int, int] | None:
    """Background color from the masks: per-channel median of the pixels with
    mask < 26, every 3rd pixel; None with fewer than 1000 such pixels."""
    hist = np.zeros((3, 256), dtype=np.int64)
    n = 0
    for rgb, mask in samples:
        flat_m = np.asarray(mask).reshape(-1)[::3]
        flat_c = np.asarray(rgb)[..., :3].reshape(-1, 3)[::3]
        sel = flat_c[flat_m < 26]
        n += len(sel)
        for c in range(3):
            hist[c] += np.bincount(sel[:, c], minlength=256)
    if n < 1000:
        return None
    out = []
    for c in range(3):
        cum = np.cumsum(hist[c])
        idx = np.nonzero(cum * 2 >= n)[0]
        out.append(int(idx[0]) if len(idx) else 255)
    return tuple(out)  # type: ignore[return-value]


def despill_for(key, strength: float) -> dict | None:
    """Only well-saturated green or blue: dominant channel - max of the others
    > 50. Returns {"channel", "others", "strength" (0-1)} or None."""
    channel = 0
    for c in (1, 2):
        if key[c] >= key[channel]:  # tie: the last one wins
            channel = c
    others = tuple(c for c in (0, 1, 2) if c != channel)
    other = max(key[others[0]], key[others[1]])
    if channel == 0 or key[channel] - other <= 50 or strength <= 0:
        return None
    return {"channel": channel, "others": others, "strength": min(1.0, max(0.0, strength / 100))}


def apply_despill(rgb: np.ndarray, mask: np.ndarray, d: dict) -> np.ndarray:
    """Over the subject (mask > 0): clamps the screen channel to the max of
    the other two and adds the removed luminance back as gray. Returns a new
    array."""
    out = np.array(rgb, dtype=np.uint8, copy=True)
    c = d["channel"]
    a, b = d["others"]
    px = out[..., :3].astype(np.float64)
    limit = np.maximum(px[..., a], px[..., b])
    spill = (px[..., c] - limit) * d["strength"]
    sel = (np.asarray(mask) != 0) & (spill > 0)
    if not sel.any():
        return out
    sp = spill[sel][:, None]
    gray = sp * LUMA[c]
    sub = np.zeros((1, 3))
    sub[0, c] = 1
    v = px[sel] - sub * sp + gray
    out[..., :3][sel] = _js_round(np.minimum(255, np.maximum(0, v))).astype(np.uint8)
    return out


def despill(rgb: np.ndarray, mask: np.ndarray, key, strength: float) -> np.ndarray:
    """`applyDespill` with `despillFor(key, strength)`; unchanged copy if None."""
    d = despill_for(key, strength)
    return np.array(rgb, dtype=np.uint8, copy=True) if d is None else apply_despill(rgb, mask, d)


# ------------------------------------------------------------ alpha dataset

_IMAGE_EXT = (".jpg", ".jpeg", ".png")


def _stem(name: str) -> str:
    """`stem` of web/src/host/files.ts: basename without the last extension."""
    return re.sub(r"\.[^.]+$", "", name.rsplit("/", 1)[-1], count=1)


def list_images(d: Path) -> list[str]:
    return sorted(p.name for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def _load(images_dir: Path, masks_dir: Path, name: str, max_side: float):
    with Image.open(images_dir / name) as im:
        rgb = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    k = max_side / max(w, h)
    if k < 1:
        w2, h2 = int(_js_round(w * k)), int(_js_round(h * k))
        rgb = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
    with Image.open(masks_dir / f"{_stem(name)}.png") as im:
        mask = np.asarray(im) if im.mode == "L" else np.asarray(im.convert("RGB"))[..., 0]
    if mask.shape != rgb.shape[:2]:
        if max(mask.shape) > max(rgb.shape[:2]):
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("mask size differs from the frame")
    return np.ascontiguousarray(rgb, dtype=np.uint8), np.ascontiguousarray(mask, dtype=np.uint8)


def alpha_frame(rgb: np.ndarray, raw: np.ndarray, d: dict | None = None) -> np.ndarray:
    """One RGBA frame of the dataset (buildAlpha's loop body)."""
    mask = erode(raw, ALPHA_ERODE)
    # Before despill: unmixing needs the photo's own background color.
    out = decontaminate(rgb, raw, mask)
    if d is not None:
        out = apply_despill(out, mask, d)
    return np.dstack([out[..., :3], mask])


def build_alpha(
    images_dir,
    masks_dir,
    out_dir,
    despill: bool = False,
    despill_strength: float = 100,
    max_side: float | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> list[str]:
    """buildAlpha: RGBA PNGs `<stem>.png` (straight alpha) in `out_dir`.
    `despill` is only meant for solid-color (chroma) masks, as in the
    reference (`s.despill && s.method === "chroma"`)."""
    max_side = math.inf if max_side is None else max_side
    images_dir, masks_dir, out_dir = Path(images_dir), Path(masks_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list_images(images_dir)
    d = None
    if despill:
        step = max(math.ceil(len(names) / 24), 1)
        samples = [_load(images_dir, masks_dir, names[i], max_side) for i in range(0, len(names), step)]
        key = background_color(samples)
        d = despill_for(key, despill_strength) if key else None
        if log:
            log(
                f"despill: #{''.join(f'{v:02x}' for v in key)} spill removed from the subject"
                if d
                else "despill: background is neither green nor blue; colors kept"
            )
    written = []
    for i, n in enumerate(names):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        rgb, raw = _load(images_dir, masks_dir, n, max_side)
        rgba = alpha_frame(rgb, raw, d)
        name = f"{_stem(n)}.png"
        Image.fromarray(rgba, "RGBA").save(out_dir / name, compress_level=1)
        written.append(name)
        if progress:
            progress((i + 1) / len(names), f"alpha frames {i + 1}/{len(names)}")
    return written


def build_alpha_dataset(
    cams: OrbitCameras,
    out_dir,
    despill: bool = False,
    despill_strength: float = 50.0,
    max_side: float | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Path:
    """buildAlphaDataset: out/images/<stem>.png RGBA + out/sparse/0 with
    cameras.bin and points3D.bin copied and images.bin renamed to .png (Brush
    looks the images up by name). `max_side` downscales the images; the model
    keeps its intrinsics unchanged, as in the reference (Brush derives the
    field of view from them, whatever the image size); None = no downscale.
    `despill` only for solid-color masks. (The app's default despill
    strength is 100; 50 here per the node-pack contract.) Returns `out_dir`."""
    max_side = math.inf if max_side is None else max_side
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    build_alpha(cams.images_dir, cams.masks_dir, out / "images", despill, despill_strength, max_side, log, progress, cancel)
    sparse = out / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    for f in ("cameras.bin", "points3D.bin"):
        shutil.copyfile(cams.sparse_dir / f, sparse / f)
    rename_images(cams.sparse_dir, lambda n: f"{_stem(n)}.png", out_dir=sparse)
    return out
