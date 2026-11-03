"""Gausify Orbit Train Splat: the training stage of web/src/host/job.ts (Brush),
options as the app's Sidebar "Training" section and the masks' despill."""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

from gausify.presets import DEFAULT_PRESET, PRESETS, REDUCE_HAZE_WEIGHT
from gausify.types import OrbitCameras

from .common import (
    CATEGORY,
    Progress,
    apply_preset,
    content_hash,
    interrupted,
    next_ply_path,
    preset_input,
    raise_if_interrupted,
    run_cancellable,
    temp_dir,
)

log = logging.getLogger("gausify.nodes")

_T = PRESETS[DEFAULT_PRESET]["train"]


def train_options(v: dict, masked: bool) -> dict:
    """Widget values (after the preset) -> TrainOptions (snake_case keys of
    presets.py), as the Sidebar toggles set them."""
    steps = int(v["steps"])
    o = {
        "total_steps": steps,
        "sh_degree": int(v["sh_degree"]),
        "max_splats": int(round(float(v["max_splats"]) * 1e6)),
        "max_resolution": int(v["max_resolution"]),
        # No preview snapshots in ComfyUI: one export, at the end.
        "export_every": steps,
        "eval_split_every": 16 if v["psnr"] else 0,
        "ssim_weight": 0.2 if v["ssim"] else 0.0,
    }
    if v.get("reduce_haze") and masked:
        o["depth_distortion_weight"] = REDUCE_HAZE_WEIGHT
    return o


class GausifyTrainSplat:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("ply_path", "num_splats", "psnr")
    FUNCTION = "train"
    OUTPUT_NODE = True
    DESCRIPTION = ("Trains a Gaussian splat with Brush (gausify-orbit training stage) and saves it as "
                   "<output>/<prefix>_NNNNN.ply. With masks, the subject is cut out (RGBA frames).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cameras": ("GAUSIFY_CAMERAS",),
                "preset": preset_input(),
                "steps": ("INT", {"default": _T["total_steps"], "min": 1000, "max": 100000, "step": 1000}),
                "sh_degree": ("INT", {"default": _T["sh_degree"], "min": 0, "max": 3}),
                "max_splats": ("FLOAT", {"default": _T["max_splats"] / 1e6, "min": 0.1, "max": 10.0, "step": 0.1,
                                         "tooltip": "Millions of Gaussians at most."}),
                "max_resolution": ("INT", {"default": _T["max_resolution"], "min": 480, "max": 3840, "step": 160}),
                "ssim": ("BOOLEAN", {"default": _T["ssim_weight"] > 0, "tooltip": "SSIM loss (weight 0.2)."}),
                "reduce_haze": ("BOOLEAN", {"default": False,
                                            "tooltip": "Depth distortion: less haze around the subject (masked only)."}),
                "psnr": ("BOOLEAN", {"default": _T["eval_split_every"] > 0,
                                     "tooltip": "Hold out every 16th frame and report its PSNR."}),
                "despill": ("BOOLEAN", {"default": False, "tooltip": "Remove green/blue screen spill (masked only; the app applies it only to chroma-key masks)."}),
                "despill_strength": ("INT", {"default": 100, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
                "filename_prefix": ("STRING", {"default": "gausify/orbit"}),
            },
        }

    def train(self, cameras: OrbitCameras, preset, steps, sh_degree, max_splats, max_resolution, ssim,
              reduce_haze, psnr, despill, despill_strength, seed, filename_prefix):
        from gausify import brush
        from gausify.train_config import to_cli_args, train_config

        v = apply_preset("train", preset, {
            "steps": steps, "sh_degree": sh_degree, "max_splats": max_splats,
            "max_resolution": max_resolution, "ssim": ssim, "psnr": psnr,
        })
        v["reduce_haze"] = reduce_haze
        masked = bool(cameras.masked)
        opts = train_options(v, masked)
        progress = Progress()
        dataset = Path(cameras.root)
        work = None
        if masked:
            from gausify.alpha import build_alpha_dataset

            key = content_hash(str(cameras.root), (Path(cameras.sparse_dir) / "images.bin").stat().st_mtime_ns,
                               bool(despill), float(despill_strength))
            work = temp_dir() / "gausify" / f"alpha-{key}"
            if work.exists():
                shutil.rmtree(work)
            progress(0.0, "building the RGBA dataset")
            dataset = Path(run_cancellable(
                build_alpha_dataset, cameras, work, despill=bool(despill), despill_strength=float(despill_strength),
                log=lambda s: log.info("gausify alpha: %s", s), progress=progress.sub(0.0, 0.02), cancel=interrupted,
            ))
            raise_if_interrupted()
        cfg = dict(train_config(opts, masked))
        cfg["export-every"] = opts["export_every"]
        cfg["seed"] = int(seed)
        args = to_cli_args(cfg)
        out_ply, rel = next_ply_path(filename_prefix)
        log.info("gausify train: %s -> %s (%s)", dataset, out_ply, " ".join(args))
        try:
            r = run_cancellable(brush.train, dataset, args, out_ply, None,
                                progress=progress.sub(0.02 if masked else 0.0, 1.0), cancel=interrupted)
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
        raise_if_interrupted()
        r = r or {}
        p = r.get("psnr")
        psnr_v = float(p) if p is not None and math.isfinite(float(p)) else 0.0
        splats = int(r.get("splats") or 0)
        progress(1.0, "done")
        return {"ui": {"text": [f"{rel}: {splats} splats" + (f", PSNR {psnr_v:.2f} dB" if p is not None else "")]},
                "result": (rel, splats, psnr_v)}
