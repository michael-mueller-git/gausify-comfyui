"""Exposure / white-balance deflicker of a training dataset (beyond the app).

Generated orbit videos often flicker: brightness and colour balance jump from
frame to frame. A splat can't match every frame at once and shows it as
blotchy colour and haze. Per frame and channel, the geometric mean of the
subject's pixels (alpha > 0.5 for RGBA cut-outs, else the whole frame;
clipped pixels left out) is pulled toward the rolling median of its
neighbours in orbit order, with a gain. Slow, real lighting changes around the
orbit survive (the median follows them); a reference-frame match would
flatten them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

#: Frames in the rolling median (odd).
WINDOW = 9
#: Largest correction per channel (gain within [1 / MAX_GAIN, MAX_GAIN]).
MAX_GAIN = 1.5
#: Pixels darker / brighter than this (0..1) are left out of the measurement.
CLIP_LO, CLIP_HI = 2 / 255, 253 / 255
#: Measured on frames scaled to at most this many pixels on the long side.
MEASURE_SIDE = 512


def _load(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    im = Image.open(path)
    im.load()
    if im.mode == "RGBA":
        a = np.asarray(im)
        return a[..., :3], a[..., 3]
    return np.asarray(im.convert("RGB")), None


def log_means(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray | None:
    """Per-channel mean of log(value) over the measured pixels (None when
    fewer than 64 qualify)."""
    h, w = rgb.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / MEASURE_SIDE)))
    x = rgb[::step, ::step].astype(np.float64) / 255.0
    ok = ((x > CLIP_LO) & (x < CLIP_HI)).all(-1)
    if alpha is not None:
        ok &= alpha[::step, ::step] > 127
    if ok.sum() < 64:
        return None
    return np.log(x[ok]).mean(0)


def rolling_target(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Median of each row's neighbours; rows of NaN (unmeasured frames) are
    ignored. Near the ends the window shrinks symmetrically (a one-sided
    window would pull the end frames of a slow lighting ramp toward the
    middle), so the first and last frames are left as they are."""
    n = len(values)
    r = max(0, int(window) // 2)
    out = np.empty_like(values)
    for i in range(n):
        k = min(r, i, n - 1 - i)
        seg = values[i - k : i + k + 1]
        out[i] = np.nanmedian(seg, axis=0) if np.isfinite(seg).any() else np.nan
    return out


def gains(values: np.ndarray, strength: float, window: int = WINDOW) -> np.ndarray:
    """(n, 3) multiplicative gains: exp(strength * (target - value)), capped;
    1 where a frame couldn't be measured."""
    t = rolling_target(values, window)
    g = np.exp(float(strength) * (t - values))
    g = np.clip(g, 1 / MAX_GAIN, MAX_GAIN)
    return np.where(np.isfinite(g), g, 1.0)


def deflicker_images(
    paths: Sequence[Path],
    strength: float,
    window: int = WINDOW,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict:
    """Corrects the images at `paths` (in orbit order) in place. RGBA stays
    RGBA (alpha untouched). Returns {"frames", "meanCorrection",
    "maxCorrection"} (percent gain change, over frames and channels)."""
    paths = [Path(p) for p in paths]
    n = len(paths)
    if strength <= 0 or n < 3:
        return {"frames": n, "meanCorrection": 0.0, "maxCorrection": 0.0}
    prog = progress or (lambda _f, _m: None)
    vals = np.full((n, 3), np.nan)
    for i, p in enumerate(paths):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        rgb, a = _load(p)
        m = log_means(rgb, a)
        if m is not None:
            vals[i] = m
        prog(0.5 * (i + 1) / n, f"deflicker: measuring {i + 1}/{n}")
    g = gains(vals, strength, window)
    for i, p in enumerate(paths):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        if np.allclose(g[i], 1.0, atol=1e-3):
            continue
        rgb, a = _load(p)
        out = np.clip(rgb.astype(np.float32) * g[i].astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
        im = Image.fromarray(np.dstack([out, a]) if a is not None else out, "RGBA" if a is not None else "RGB")
        im.save(p, compress_level=1)
        prog(0.5 + 0.5 * (i + 1) / n, f"deflicker: correcting {i + 1}/{n}")
    change = np.abs(g - 1.0) * 100
    return {"frames": n, "meanCorrection": round(float(change.mean()), 2),
            "maxCorrection": round(float(change.max()), 2)}
