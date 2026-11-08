"""Chroma key (solid-colour background removal).

Port of the keying half of gausify-orbit `web/src/pipeline/chroma.ts`
(borderKey, keyDistance, chromaMask, DEFAULT_CHROMA). The despill half lives in
`gausify/alpha.py` (`despill_for`, `apply_despill`). Masks: 255 = subject.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from gausify.alpha import _js_round

DEFAULT_CHROMA = {
    "auto_color": True,
    "color": (0, 255, 0),
    "tolerance": 12,
    "softness": 6,
    "contiguous": True,
}

_SCALE = 4.42


def border_key(rgb: np.ndarray) -> tuple[int, int, int]:
    """Per-channel (upper) median of the border pixels, with step (w+h)/400."""
    h, w = rgb.shape[:2]
    step = max((w + h) // 400, 1)
    xs, ys = np.arange(0, w, step), np.arange(0, h, step)
    px = np.concatenate([rgb[0, xs, :3], rgb[h - 1, xs, :3], rgb[ys, 0, :3], rgb[ys, w - 1, :3]])
    srt = np.sort(px, axis=0)
    return tuple(int(v) for v in srt[len(srt) >> 1])  # type: ignore[return-value]


def key_distance(rgb: np.ndarray, key) -> np.ndarray:
    """Distance to the background colour, accepting the colour up to 1.6x brighter
    and never darker. `rgb` (..., 3) -> float32 (...)."""
    p = np.asarray(rgb)[..., :3].astype(np.float64)
    k = [float(v) for v in key]
    kk = max(k[0] * k[0] + k[1] * k[1] + k[2] * k[2], 1.0)
    s = np.minimum(1.6, np.maximum(1.0, (p[..., 0] * k[0] + p[..., 1] * k[1] + p[..., 2] * k[2]) / kk))
    d = np.sqrt(
        (p[..., 0] - s * k[0]) ** 2 + (p[..., 1] - s * k[1]) ** 2 + (p[..., 2] - s * k[2]) ** 2
    )
    return d.astype(np.float32)


def chroma_mask(rgb: np.ndarray, key, tolerance: float, softness: float, contiguous: bool) -> np.ndarray:
    """uint8 (H, W) mask, 255 = subject, from an RGB (H, W, 3) uint8 frame."""
    d = key_distance(rgb, key).astype(np.float64)  # stored as float32 in the reference
    tol = min(100.0, max(0.0, float(tolerance))) * _SCALE
    soft = max(min(100.0, max(0.0, float(softness))) * _SCALE, 1.0)
    lim = tol + soft
    keep = None
    if contiguous:
        # Only what connects to the border counts as background (holes in the subject stay).
        lab, n = ndimage.label(d <= lim)  # 4-connectivity
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        border = border[border != 0]
        reached = np.zeros(n + 1, dtype=bool)
        reached[border] = True
        keep = ~reached[lab]  # not reached -> subject
    mask = np.zeros(d.shape, dtype=np.uint8)
    soft_zone = (d > tol) & (d < lim)
    mask[soft_zone] = _js_round(255 * (d[soft_zone] - tol) / soft).astype(np.uint8)
    mask[d >= lim] = 255
    if keep is not None:
        mask[keep] = 255
    return mask


def chroma_masks(frames: np.ndarray, auto_color: bool, manual, tolerance: float, softness: float,
                 contiguous: bool, progress=None, cancel=None) -> tuple[np.ndarray, list, float]:
    """Masks of all `frames` (n,h,w,3 uint8). Returns (masks, keys, mean coverage 0..1)."""
    n = len(frames)
    masks = np.zeros(frames.shape[:3], np.uint8)
    keys, coverage = [], 0.0
    for i in range(n):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        key = border_key(frames[i]) if auto_color else manual
        keys.append(key)
        masks[i] = chroma_mask(frames[i], key, tolerance, softness, contiguous)
        coverage += float(masks[i].sum(dtype=np.float64)) / (255.0 * masks[i].size)
        if progress:
            progress((i + 1) / n, f"solid colour {i + 1}/{n}")
    return masks, keys, coverage / max(n, 1)
