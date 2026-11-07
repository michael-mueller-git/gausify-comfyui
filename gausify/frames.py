"""Frame selection: uniform candidates over the range, the sharpest of each
window (variance of the Laplacian), scaled to fit max_side without upscaling.

Port of extractFrames / framesFromImages / windowsOf / sharpness in
gausify-orbit web/src/pipeline/frames.ts, on an in-memory frame stack:

- video (`fps` given): the stack is the decoded video at `fps`; a seek to time
  `t` lands on source frame floor(t * fps) (frame i shows [i/fps, (i+1)/fps)),
  and the reference seeks to start + k/sampleFps + 0.5/fps, i.e. the nearest
  source frame, like ffmpeg's fps filter;
- images (`fps` None): framesFromImages semantics (every image a candidate).

Resizing: the browser's canvas `imageSmoothingQuality = "high"` downscale is
approximated with cv2.INTER_AREA (closest available box-like filter), for both
the 640 px sharpness thumbnail and the max_side output. The reference writes
JPEG q0.95 frames; here the selected frames are returned as float32 arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np

from .types import OrbitTimes

THUMB_SIDE = 640


def _js_round(x: float) -> int:
    return int(math.floor(x + 0.5))


def windows_of(total: int, target: int) -> np.ndarray:
    """Window of each candidate when `total` become `target` frames:
    lo(w) = floor(w*total/target), as in frames.rs."""
    window_of = np.zeros(total, dtype=np.int32)
    for w in range(target):
        lo = (w * total) // target
        hi = max(((w + 1) * total) // target, lo + 1)
        window_of[lo : min(hi, total)] = w
    return window_of


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(np.rint(np.asarray(img, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def sharpness(img: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian on Rec.709 luma rounded to 8 bits
    (same as frames.rs). `img`: HxWx3 RGB (uint8, or float in [0,1]) or an HxW
    luma image. Call it on the thumbnail (`thumbnail`)."""
    if img.ndim == 3:
        rgb = _to_u8(img[..., :3]).astype(np.float64)
        lum = np.rint(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    else:
        lum = np.asarray(img, dtype=np.float64)
        if img.dtype != np.uint8 and lum.max(initial=0) <= 1.0:
            lum = np.rint(lum * 255.0)
    h, w = lum.shape
    if w < 3 or h < 3:
        return 0.0
    c = lum[1:-1, 1:-1]
    lap = lum[1:-1, :-2] + lum[1:-1, 2:] + lum[:-2, 1:-1] + lum[2:, 1:-1] - 4 * c
    n = lap.size
    mean = float(lap.sum()) / n
    return float((lap * lap).sum()) / n - mean * mean


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def thumbnail_size(w: int, h: int) -> tuple[int, int]:
    ts = min(1.0, THUMB_SIDE / max(w, h))
    return max(1, _js_round(w * ts)), max(1, _js_round(h * ts))


def thumbnail(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    """8-bit RGB thumbnail, as drawn on the 2D canvas."""
    return _resize(_to_u8(frame[..., :3]), tw, th)


def output_size(w: int, h: int, max_side: int, fps: float | None) -> tuple[int, int]:
    """Size of the selected frames: fit max_side x max_side, no upscaling."""
    scale = min(1.0, max_side / max(w, h))
    if fps is None:
        return max(1, _js_round(w * scale)), max(1, _js_round(h * scale))
    return _js_round(w * scale), _js_round(h * scale)


def pick_frames(
    frames: np.ndarray,
    fps: float | None,
    target_frames: int,
    max_side: int,
    oversample: int = 3,
    trim_start: float = 0.0,
    trim_end: float | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> list[dict]:
    """The selection only: [{"index": source frame, "time": s, "sharpness": v}]
    (time = trim_start + k/sampleFps for video; the source index for images)."""
    n_src = int(frames.shape[0])
    H0, W0 = int(frames.shape[1]), int(frames.shape[2])
    if fps is None:
        total = n_src
        if not total:
            raise ValueError("No images to use as frames.")
        target = min(max(1, int(target_frames)), total)
        # Thumbnail that fits 640x640 of the image itself.
        tw, th = thumbnail_size(W0, H0)

        def src_of(k):
            return k

        def time_of(k):
            return float(k)
    else:
        if n_src == 0:
            raise ValueError("The selected video range is empty.")
        duration = n_src / fps
        start = max(0.0, float(trim_start))
        end = min(trim_end if trim_end is not None and trim_end > start else duration, duration)
        span = end - start
        if span <= 0.1:
            raise ValueError("The selected video range is empty.")
        wanted = target_frames * max(1, oversample)
        sample_fps = max(0.1, min(wanted / span, fps))
        total = max(1, _js_round(span * sample_fps))
        target = min(max(1, int(target_frames)), total)
        # Thumbnail of the max_side-scaled frame size, drawn from the source.
        tw, th = thumbnail_size(*output_size(W0, H0, max_side, fps))

        def src_of(k):
            # half a source frame ahead: the seek lands on the nearest frame
            t = min(start + k / sample_fps + 0.5 / fps, end - 1e-3)
            return min(max(int(math.floor(t * fps)), 0), n_src - 1)

        def time_of(k):
            return float(trim_start) + k / sample_fps

    window_of = windows_of(total, target)
    picks: list[dict] = []
    best: dict | None = None
    last_i, last_score = -1, 0.0
    for k in range(total):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        if k > 0 and window_of[k] != window_of[k - 1] and best is not None:
            picks.append(best)
            best = None
        i = src_of(k)
        if i != last_i:  # consecutive candidates may land on the same frame
            last_i, last_score = i, sharpness(thumbnail(frames[i], tw, th))
        score = last_score
        # tie: the last one wins, like Rust's max_by
        if best is None or score >= best["sharpness"]:
            best = {"index": i, "time": time_of(k), "sharpness": score}
        if progress is not None:
            progress((k + 1) / total, f"Frames: candidate {k + 1}/{total}")
    if best is not None:
        picks.append(best)
    return picks


def filter_picks(
    frames: np.ndarray,
    picks: list[dict],
    skip_duplicates: bool = False,
    duplicate_threshold: float = 1.0,
    min_sharpness: float = 0.0,
) -> tuple[list[dict], dict]:
    """Beyond the app: drops picks that repeat the previous kept one
    (`skip_duplicates`: mean absolute difference of 160 px grayscale
    thumbnails below `duplicate_threshold`, in 8-bit levels; frozen or repeated
    frames of generated videos), and picks blurrier than `min_sharpness` x the
    median sharpness of the picks (motion blur the window had no sharper
    candidate for). Returns (kept picks, {"duplicates": n, "blurry": n})."""
    dropped = {"duplicates": 0, "blurry": 0}
    out = list(picks)
    if min_sharpness > 0 and out:
        med = float(np.median([p["sharpness"] for p in out]))
        keep = [p for p in out if p["sharpness"] >= min_sharpness * med]
        dropped["blurry"] = len(out) - len(keep)
        out = keep
    if skip_duplicates and out:
        H0, W0 = int(frames.shape[1]), int(frames.shape[2])
        s = min(1.0, 160 / max(W0, H0))
        tw, th = max(1, _js_round(W0 * s)), max(1, _js_round(H0 * s))

        def gray(i):
            t = thumbnail(frames[i], tw, th).astype(np.float32)
            return 0.2126 * t[..., 0] + 0.7152 * t[..., 1] + 0.0722 * t[..., 2]

        keep, last = [out[0]], gray(out[0]["index"])
        for p in out[1:]:
            g = gray(p["index"])
            if p["index"] == keep[-1]["index"] or float(np.mean(np.abs(g - last))) < duplicate_threshold:
                dropped["duplicates"] += 1
                continue
            keep.append(p)
            last = g
        out = keep
    return out, dropped


def select_frames(
    frames: np.ndarray,
    fps: float | None,
    target_frames: int,
    max_side: int,
    oversample: int = 3,
    trim_start: float = 0.0,
    trim_end: float | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    skip_duplicates: bool = False,
    duplicate_threshold: float = 1.0,
    min_sharpness: float = 0.0,
    stats: dict | None = None,
) -> tuple[np.ndarray, OrbitTimes]:
    """Selected frames (K,H',W',3) float32 in [0,1], resized to fit max_side
    (no upscaling), and their OrbitTimes (video: seconds from the video start,
    trim_start + k/sampleFps; images: source image index, from_images=True;
    `indices`: source frame of each). `skip_duplicates` / `min_sharpness`: see
    filter_picks (off = the app's selection); `stats` receives the drop counts."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("frames must be (N,H,W,3)")
    picks = pick_frames(frames, fps, target_frames, max_side, oversample, trim_start, trim_end, progress, cancel)
    picks, dropped = filter_picks(frames, picks, skip_duplicates, duplicate_threshold, min_sharpness)
    if stats is not None:
        stats.update(dropped)
    W, H = output_size(int(frames.shape[2]), int(frames.shape[1]), max_side, fps)
    out = np.stack([
        _resize(np.ascontiguousarray(frames[p["index"], ..., :3], dtype=np.float32), W, H) for p in picks
    ]).astype(np.float32, copy=False)
    return out, OrbitTimes(times=[p["time"] for p in picks], from_images=fps is None,
                           indices=[int(p["index"]) for p in picks])


def select_masks(masks: np.ndarray, indices: list[int], width: int, height: int) -> np.ndarray:
    """The masks (N,H,W) float of the source frames at `indices`, resized to
    the selected frames' size (INTER_AREA down, linear up)."""
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim == 4:
        masks = masks[..., 0]
    out = []
    for i in indices:
        m = masks[i]
        if m.shape != (height, width):
            down = m.shape[0] * m.shape[1] > width * height
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)
        out.append(m)
    return np.clip(np.stack(out), 0.0, 1.0).astype(np.float32)
