"""Gausify Orbit Frame Select: the frames stage (web/src/pipeline/frames.ts),
options as the app's Sidebar "Frames" section."""

from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np

from gausify.presets import DEFAULT_PRESET, PRESETS

from .common import (
    CATEGORY,
    Progress,
    apply_preset,
    interrupted,
    preset_input,
    raise_if_interrupted,
    run_cancellable,
    to_numpy,
)

log = logging.getLogger("gausify.nodes")

_F = PRESETS[DEFAULT_PRESET]["frames"]


def _video_components(video):
    """Core VIDEO input -> (frames (N,H,W,3) float32, fps)."""
    c = video.get_components()
    images = getattr(c, "images", None)
    if images is None and isinstance(c, dict):
        images = c["images"]
    rate = getattr(c, "frame_rate", None)
    if rate is None and isinstance(c, dict):
        rate = c.get("frame_rate")
    fps = float(Fraction(rate)) if rate is not None else 0.0
    return images, fps


class GausifyFrameSelect:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "ORBIT_TIMES", "MASK")
    RETURN_NAMES = ("images", "frame_times", "masks")
    FUNCTION = "select"
    DESCRIPTION = ("Picks the sharpest frame of each slot along the orbit (gausify-orbit frames stage). "
                   "Give a video, or an image batch (fps 0 = an image sequence, in orbit order). Optional "
                   "masks of every source frame (e.g. tracked on the full video) are picked along.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": preset_input(),
                "frames": ("INT", {"default": _F["target_frames"], "min": 20, "max": 600, "step": 10,
                                   "tooltip": "Frames to keep (target)."}),
                "max_side": ("INT", {"default": _F["max_side"], "min": 480, "max": 3840, "step": 160,
                                     "tooltip": "Longest side of the kept frames (px)."}),
                "candidates": ("INT", {"default": _F["oversample"], "min": 1, "max": 8,
                                       "tooltip": "Candidates per kept frame; the sharpest wins."}),
                "trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e5, "step": 0.1,
                                         "tooltip": "Seconds skipped at the start."}),
                "trim_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1e5, "step": 0.1,
                                       "tooltip": "Last second used; -1 = the end."}),
            },
            "optional": {
                "video": ("VIDEO",),
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate of `images`; 0 = an image sequence (no timestamps)."}),
                "skip_duplicates": ("BOOLEAN", {"default": False, "tooltip": (
                    "Drop picks that repeat the previous one (frozen or repeated frames of generated videos). "
                    "Fewer frames may come out. Not in the app.")}),
                "min_sharpness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": (
                    "Drop picks blurrier than this x the median sharpness (motion blur; e.g. 0.4). "
                    "0 = off (the app). Fewer frames may come out.")}),
                "masks": ("MASK", {"tooltip": (
                    "One mask per source frame (white = subject), e.g. SAM2 or video matting run on the whole "
                    "video, where tracking works best. The masks of the picked frames come out of `masks`, "
                    "resized to the frames.")}),
            },
        }

    def select(self, preset, frames, max_side, candidates, trim_start, trim_end, video=None, images=None, fps=0.0,
               skip_duplicates=False, min_sharpness=0.0, masks=None):
        from gausify.frames import select_frames

        v = apply_preset("frames", preset, {"frames": frames, "max_side": max_side, "candidates": candidates})
        if video is not None:
            src, rate = _video_components(video)
        elif images is not None:
            src, rate = images, float(fps)
        else:
            raise ValueError("Gausify Orbit Frame Select needs a video or images.")
        arr = to_numpy(src).astype(np.float32, copy=False)
        if arr.ndim == 3:
            arr = arr[None]
        arr = arr[..., :3]
        if masks is not None:
            m = to_numpy(masks)
            m = m[None] if m.ndim == 2 else m
            if len(m) != len(arr):
                raise ValueError(f"{len(m)} masks for {len(arr)} source frames: give one mask per frame of the "
                                 "video / images (before selection)")
        progress = Progress()
        progress(0.0, "selecting frames")
        dropped: dict = {}
        out, times = run_cancellable(
            select_frames,
            arr,
            rate if rate and rate > 0 else None,
            int(v["frames"]),
            int(v["max_side"]),
            int(v["candidates"]),
            float(trim_start),
            None if trim_end is None or trim_end < 0 else float(trim_end),
            progress=progress,
            cancel=interrupted,
            skip_duplicates=bool(skip_duplicates),
            min_sharpness=float(min_sharpness),
            stats=dropped,
        )
        raise_if_interrupted()
        if any(dropped.values()):
            log.info("gausify frames: %d kept, dropped %d duplicates and %d blurry", len(out),
                     dropped.get("duplicates", 0), dropped.get("blurry", 0))
        if len(out) < 3:
            raise ValueError(f"only {len(out)} frames left after dropping duplicates/blurry frames; "
                             "lower min_sharpness or turn skip_duplicates off")
        progress(1.0, "frames selected")
        import torch

        sel = None
        if masks is not None:
            from gausify.frames import select_masks

            sel = torch.from_numpy(run_cancellable(select_masks, m, times.indices, out.shape[2], out.shape[1]))
        return (torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32)), times, sel)
