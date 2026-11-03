"""Gausify Orbit Frame Select: the frames stage (web/src/pipeline/frames.ts),
options as the app's Sidebar "Frames" section."""

from __future__ import annotations

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
    RETURN_TYPES = ("IMAGE", "ORBIT_TIMES")
    RETURN_NAMES = ("images", "frame_times")
    FUNCTION = "select"
    DESCRIPTION = ("Picks the sharpest frame of each slot along the orbit (gausify-orbit frames stage). "
                   "Give a video, or an image batch (fps 0 = an image sequence, in orbit order).")

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
            },
        }

    def select(self, preset, frames, max_side, candidates, trim_start, trim_end, video=None, images=None, fps=0.0):
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
        progress = Progress()
        progress(0.0, "selecting frames")
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
        )
        raise_if_interrupted()
        progress(1.0, "frames selected")
        import torch

        return (torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32)), times)
