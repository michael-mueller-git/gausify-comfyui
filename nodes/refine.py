"""Gausify Orbit Refine Poses: corrects the camera poses against a trained
splat (gausify/pose_refine.py), for a second Train Splat. Beyond the app."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gausify.pose_refine import (
    DEFAULT_FEATURES,
    DEFAULT_MAX_ROTATION,
    DEFAULT_MAX_SHIFT,
    DEFAULT_MAX_SIDE,
    DEFAULT_MIN_INLIERS,
)
from gausify.types import OrbitCameras

from .clean import resolve_ply
from .common import CATEGORY, Progress, content_hash, interrupted, raise_if_interrupted, run_cancellable

log = logging.getLogger("gausify.nodes")


class GausifyRefinePoses:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("GAUSIFY_CAMERAS", "STRING")
    RETURN_NAMES = ("cameras", "stats")
    FUNCTION = "refine"
    DESCRIPTION = ("Corrects the camera poses against a trained splat: renders it from every camera, matches "
                   "each render with its photo (ALIKED-LightGlue) and solves the photo's pose from the render's "
                   "depth. Train Splat again on the output cameras: less ghosting and sharper detail where "
                   "poses were slightly off. Train, refine, train again; a second round corrects further.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cameras": ("GAUSIFY_CAMERAS",),
            "ply_path": ("STRING", {"forceInput": True}),
            "max_side": ("INT", {"default": DEFAULT_MAX_SIDE, "min": 320, "max": 2560, "step": 32, "tooltip": (
                "Size the renders and photos are matched at (multiples of 32). Larger is more precise and "
                "slower.")}),
            "max_features": ("INT", {"default": DEFAULT_FEATURES, "min": 1024, "max": 4000, "step": 512,
                                     "tooltip": "Keypoints per image: picks the ALIKED-LightGlue model "
                                                "(1024, 2048, 3072 or 3584)."}),
            "min_inliers": ("INT", {"default": DEFAULT_MIN_INLIERS, "min": 10, "max": 1000, "step": 5,
                                    "tooltip": "A frame's pose changes only with at least this many PnP inliers."}),
            "max_rotation": ("FLOAT", {"default": DEFAULT_MAX_ROTATION, "min": 0.1, "max": 20.0, "step": 0.1,
                                       "tooltip": "Largest accepted rotation change (degrees); larger "
                                                  "corrections are treated as wrong matches and skipped."}),
            "max_shift": ("FLOAT", {"default": DEFAULT_MAX_SHIFT, "min": 0.1, "max": 20.0, "step": 0.1,
                                    "tooltip": "Largest accepted camera move, in % of the orbit radius."}),
        }}

    def refine(self, cameras: OrbitCameras, ply_path, max_side, max_features, min_inliers, max_rotation,
               max_shift):
        from gausify.pose_refine import refine_poses

        src, _ = resolve_ply(ply_path)
        root = Path(cameras.root)
        st = src.stat()
        params = (int(max_side), int(max_features), int(min_inliers), float(max_rotation), float(max_shift))
        key = content_hash(str(src), st.st_size, st.st_mtime_ns, str(root),
                           (Path(cameras.sparse_dir) / "images.bin").stat().st_mtime_ns, params)
        out = root.parent / f"{root.name}-refined-{key}"
        stats_file = out / "refine.json"
        if stats_file.is_file() and (out / "sparse" / "0" / "images.bin").is_file():
            stats = json.loads(stats_file.read_text())
        else:
            lines: list[str] = []

            def say(s: str):
                lines.append(s)
                log.info("gausify refine: %s", s)

            stats = run_cancellable(
                refine_poses, cameras, src, out, *params, progress=Progress(), cancel=interrupted, log=say,
            )
            raise_if_interrupted()
            stats["log"] = lines
            stats_file.write_text(json.dumps(stats))
        refined = OrbitCameras(root=out, names=list(cameras.names), width=cameras.width, height=cameras.height,
                               masked=cameras.masked, stats={**(cameras.stats or {}), "refine": stats})
        text = (f"refined {stats['refined']}/{stats['views']} poses, median error {stats['medianErrorBefore']} -> "
                f"{stats['medianErrorAfter']} px, largest {stats['maxRotationDeg']} deg / "
                f"{stats['maxShiftPercent']}%")
        return {"ui": {"text": [text]}, "result": (refined, json.dumps(stats, indent=2))}
