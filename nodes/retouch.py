"""Gausify Orbit Retouch: render a trained splat along a new camera path
(Retouch Clip), restore that video with any ComfyUI video workflow, then
fine-tune the splat with the restored frames (Retouch Apply). Port of
gausify-orbit's Retouch (docs/retouch.md, gausify/retouch.py)."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np

from gausify.presets import REDUCE_HAZE_WEIGHT
from gausify.retouch import (
    DEFAULT_FRAMES,
    DEFAULT_LONG_SIDE,
    DEFAULT_RESTORED_RATIO,
    DEFAULT_STEPS,
    DEFAULT_TILT,
    LR_MEAN,
    MAX_SPEED,
    MOTIONS,
)
from gausify.types import OrbitCameras

from .clean import resolve_ply
from .common import (
    CATEGORY,
    Progress,
    content_hash,
    images_to_u8,
    interrupted,
    raise_if_interrupted,
    run_cancellable,
    temp_dir,
)

log = logging.getLogger("gausify.nodes")


def _photos(cameras: OrbitCameras):
    from gausify.retouch import read_lens, read_poses

    sparse = Path(cameras.sparse_dir)
    return read_lens((sparse / "cameras.bin").read_bytes()), read_poses((sparse / "images.bin").read_bytes())


class GausifyRetouchClip:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "GAUSIFY_RETOUCH_CLIP", "STRING")
    RETURN_NAMES = ("reference", "alpha", "clip", "info")
    FUNCTION = "clip"
    DESCRIPTION = ("Retouch, step 1: renders the splat along a camera path through angles the orbit didn't "
                   "cover (from a real photo and back). Restore `reference` with a video workflow (e.g. LTX "
                   "IC-LoRA), then give the restored frames and `clip` to Gausify Orbit Retouch Apply.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cameras": ("GAUSIFY_CAMERAS",),
            "ply_path": ("STRING", {"forceInput": True}),
            "motion": (MOTIONS, {"default": "wave", "tooltip": (
                "wave: a full loop, rising above the photos then dropping below; orbit_below / orbit_above: a "
                "full loop at another height; arc_below / arc_above: a third of the orbit around `photo`, "
                "deviating in height. Loops start and end at `photo`.")}),
            "tilt": ("FLOAT", {"default": DEFAULT_TILT, "min": 5.0, "max": 80.0, "step": 1.0,
                               "tooltip": "Largest height change, in degrees from the photos' height."}),
            "photo": ("INT", {"default": 0, "min": 0, "max": 100000,
                              "tooltip": "Index of the photo (in orbit order) the path starts from / centres on."}),
            "seconds": ("FLOAT", {"default": round(DEFAULT_FRAMES / 24, 1), "min": 0.5, "max": 60.0, "step": 0.1,
                                  "tooltip": "Clip duration; frames are rounded to 8n + 1 (LTX)."}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                              "tooltip": "Frame rate of the clip (use your restoration workflow's)."}),
            "long_side": ("INT", {"default": DEFAULT_LONG_SIDE, "min": 512, "max": 1920, "step": 32,
                                  "tooltip": "Long side of the reference video (multiples of 32)."}),
        }}

    def clip(self, cameras: OrbitCameras, ply_path, motion, tilt, photo, seconds, fps, long_side):
        import torch

        from gausify import brush
        from gausify.retouch import (
            RetouchClip,
            clip_frames,
            clip_lens,
            motion_path,
            orbit_frame,
            peak_speed,
            render_cameras,
        )

        src, _ = resolve_ply(ply_path)
        lens, photos = _photos(cameras)
        if len(photos) < 3:
            raise ValueError("too few registered photos for a retouch path")
        if not 0 <= int(photo) < len(photos):
            raise ValueError(f"photo {photo} is outside the {len(photos)} registered photos (0..{len(photos) - 1})")
        frames = clip_frames(float(seconds), float(fps))
        size, k = clip_lens(lens, int(long_side))
        poses = motion_path(photos, motion, int(photo), float(tilt), frames)
        clip = RetouchClip(motion=motion, size=size, intrinsics=k, poses=poses, fps=float(fps))
        speed = peak_speed(poses, float(fps), center=orbit_frame(photos)[0])
        clip.stats = {"frames": frames, "size": list(size), "peakSpeed": round(speed, 1), "photos": len(photos),
                      "startPhoto": photos[int(photo)].name}
        progress = Progress()
        rs = run_cancellable(brush.render, src, render_cameras(clip), True, progress, interrupted)
        raise_if_interrupted()
        rgb = np.stack([np.clip(r["rgba"][..., :3], 0, 1) for r in rs]).astype(np.float32)
        alpha = np.stack([np.clip(r["alpha"], 0, 1) for r in rs]).astype(np.float32)
        clip.alpha = alpha
        info = (f"{motion}: {frames} frames at {fps:g} fps, {size[0]}x{size[1]}, peak speed {speed:.0f} deg/s")
        if speed > MAX_SPEED:
            longer = math.ceil(seconds * speed / MAX_SPEED * 10) / 10
            info += f" (above {MAX_SPEED:.0f}: video models blur fast motion; try {longer:g} s)"
            clip.stats["warning"] = info
        log.info("gausify retouch clip: %s", info)
        return {"ui": {"text": [info]},
                "result": (torch.from_numpy(rgb), torch.from_numpy(alpha), clip, json.dumps(clip.stats, indent=2))}


class GausifyRetouchApply:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("ply_path", "num_splats", "stats")
    FUNCTION = "apply"
    OUTPUT_NODE = True
    DESCRIPTION = ("Retouch, step 2: fine-tunes the splat from itself on the photos plus some of the restored "
                   "frames (by default up to 25% of the photo count: more pulls it toward invented details). "
                   "Saves <splat>_retouch.ply next to the input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cameras": ("GAUSIFY_CAMERAS",),
                "ply_path": ("STRING", {"forceInput": True}),
                "clip": ("GAUSIFY_RETOUCH_CLIP",),
                "restored": ("IMAGE", {"tooltip": "The restored reference video: same frame count and aspect ratio "
                                                  "(any size)."}),
                "steps": ("INT", {"default": DEFAULT_STEPS, "min": 500, "max": 20000, "step": 500}),
                "restored_ratio": ("FLOAT", {"default": DEFAULT_RESTORED_RATIO, "min": 0.05, "max": 2.0, "step": 0.05,
                                             "tooltip": "Restored frames used, as a fraction of the photo count."}),
                "max_splats": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1,
                                         "tooltip": "Millions of Gaussians at most (use the training's)."}),
                "max_resolution": ("INT", {"default": 1920, "min": 480, "max": 3840, "step": 160}),
                "ssim": ("BOOLEAN", {"default": True, "tooltip": "SSIM loss (weight 0.2)."}),
                "reduce_haze": ("BOOLEAN", {"default": False, "tooltip": "Depth distortion (masked only)."}),
                "despill": ("BOOLEAN", {"default": False, "tooltip": "As on Train Splat (Solid Colour masks)."}),
                "despill_strength": ("INT", {"default": 100, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
                "drop_inconsistent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": (
                    "As on Train Splat: leave the pose check's outlier photos out (use the same value).")}),
                "deflicker": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": (
                    "As on Train Splat: even out exposure flicker of the photos (use the same value).")}),
            },
            "optional": {
                "advanced": ("GAUSIFY_TRAIN_SETTINGS", {"tooltip": "From Gausify Orbit Train Advanced."}),
            },
        }

    def apply(self, cameras: OrbitCameras, ply_path, clip, restored, steps, restored_ratio, max_splats,
              max_resolution, ssim, reduce_haze, despill, despill_strength, seed, drop_inconsistent=0.0,
              advanced=None, deflicker=0.0):
        from gausify import brush
        from gausify.retouch import build_retouch_dataset, sh_degree_of
        from gausify.train_config import to_cli_args, train_config

        src, root = resolve_ply(ply_path)
        out_ply = src.with_name(f"{src.stem}_retouch.ply")
        frames = images_to_u8(restored)
        alpha = clip.alpha
        if alpha is None:
            raise ValueError("the clip carries no render alpha: connect the clip from Gausify Orbit Retouch Clip")
        masked = bool(cameras.masked)
        steps = max(int(steps), 1)
        opts = {
            "total_steps": steps,
            "sh_degree": sh_degree_of(src),
            "max_splats": int(round(float(max_splats) * 1e6)),
            "max_resolution": int(max_resolution),
            "export_every": steps,
            "eval_split_every": 0,  # the new views would land in the test set
            "ssim_weight": 0.2 if ssim else 0.0,
            "lr_mean": LR_MEAN,
            "growth_stop_iter": round(steps / 2),
        }
        if reduce_haze and masked:
            opts["depth_distortion_weight"] = REDUCE_HAZE_WEIGHT
        progress = Progress()
        work = temp_dir() / "gausify" / f"retouch-{content_hash(str(src), clip.motion, len(frames))}"
        try:
            ds, stats = run_cancellable(
                build_retouch_dataset, cameras, clip, frames, alpha, work, float(restored_ratio),
                bool(despill), float(despill_strength), lambda s: log.info("gausify %s", s),
                progress.sub(0.0, 0.1), interrupted,
            )
            raise_if_interrupted()
            if float(deflicker) > 0:
                from .train import _deflicker

                stats["deflicker"] = _deflicker(cameras, ds, float(deflicker))
            if float(drop_inconsistent) > 0:
                from .train import _drop_inconsistent

                # the retouch dataset is always a copy
                stats["droppedPhotos"] = len(_drop_inconsistent(cameras, ds, float(drop_inconsistent)))
            cfg = dict(train_config(opts, masked))
            cfg.update(advanced or {})
            # The fine-tune's own settings win over the advanced ones (as the app).
            cfg["lr-mean"], cfg["growth-stop-iter"] = LR_MEAN, opts["growth_stop_iter"]
            cfg["export-every"] = steps
            cfg["seed"] = int(seed)
            args = to_cli_args(cfg)
            log.info("gausify retouch: %s from %s (%s)", ds, src, " ".join(args))
            r = run_cancellable(brush.train, ds, args, out_ply, src, progress=progress.sub(0.1, 1.0),
                                cancel=interrupted) or {}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        raise_if_interrupted()
        stats.update(splats=int(r.get("splats") or 0), steps=steps, motion=clip.motion)
        rel = out_ply.relative_to(root).as_posix()
        text = (f"{rel}: {stats['splats']} splats, fine-tuned {steps} steps on {stats['photos']} photos + "
                f"{stats['restored']} restored frames")
        progress(1.0, "retouch done")
        return {"ui": {"text": [text]}, "result": (rel, stats["splats"], json.dumps(stats, indent=2))}
