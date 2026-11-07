"""Gausify Orbit Poses: the poses stage of web/src/host/job.ts (runSfm, then the
optional pose check), options as the app's Sidebar "Poses" section."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from gausify.presets import DEFAULT_PRESET, PRESETS, SFM_MAX_FEATURES, SFM_MAX_SIDE
from gausify.sfm import ESTIMATORS
from gausify.types import OrbitCameras, OrbitTimes

from .common import (
    CATEGORY,
    Progress,
    apply_preset,
    content_hash,
    frame_names,
    images_to_u8,
    interrupted,
    masks_to_u8,
    preset_input,
    raise_if_interrupted,
    run_cancellable,
    temp_dir,
    write_pngs,
)

log = logging.getLogger("gausify.nodes")

_P = PRESETS[DEFAULT_PRESET]["poses"]

#: The app's mask defaults (src/App.tsx), used by the pose check's frames.
# masks.ts only despills chroma-key masks; masks from other nodes are "external" -> no despill.
DESPILL, DESPILL_STRENGTH = False, 100.0


def prepare_dataset(images, masks=None, key_extra=None, root: Path | None = None) -> tuple[Path, list[str], int, int, bool]:
    """Writes images/ (and masks/) of a dataset folder keyed by their content.
    Returns (root, names, width, height, masked)."""
    rgb = images_to_u8(images)
    n, h, w = rgb.shape[:3]
    m = masks_to_u8(masks, (h, w)) if masks is not None else None
    if m is not None and len(m) != n:
        raise ValueError(f"{len(m)} masks for {n} images")
    key = content_hash(rgb, m if m is not None else "nomask", key_extra or {})
    root = Path(root) if root is not None else temp_dir() / "gausify" / key
    names = frame_names(n)
    done = root / ".frames-done"
    if not done.exists():
        if root.exists():
            shutil.rmtree(root)
        write_pngs(rgb, root / "images", names)
        if m is not None:
            write_pngs(m, root / "masks", names, rgb=False)
        done.write_text("1")
    return root, names, w, h, m is not None


class GausifyPoses:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("GAUSIFY_CAMERAS", "STRING", "STRING")
    RETURN_NAMES = ("cameras", "colmap_path", "stats")
    FUNCTION = "poses"
    DESCRIPTION = ("Camera poses of an orbit (gausify-orbit poses stage): AKAZE + incremental SfM, or COLMAP's "
                   "SIFT + global mapper. Writes a COLMAP dataset folder (images/, masks/, sparse/0).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": preset_input(),
                "estimator": (list(ESTIMATORS), {"default": "akaze", "tooltip": (
                    "akaze: AKAZE + incremental SfM (the app); aliked_lightglue: learned features (RaCo-ALIKED) "
                    "matched by LightGlue + the same incremental SfM, more robust on skin, plain clothes, blur "
                    "and large angle steps (ONNX model ~65 MB, downloaded on first use; needs a GPU to be fast); "
                    "colmap: COLMAP SIFT + global mapper.")}),
                "match_window": ("INT", {"default": _P["match_window"], "min": 3, "max": 40,
                                         "tooltip": "Neighbours matched on each side."}),
                "closed_loop": ("BOOLEAN", {"default": _P["closed_loop"],
                                            "tooltip": "The orbit is a full turn: the last frames match the first."}),
                "flat_orbit": ("BOOLEAN", {"default": False, "tooltip": "Force the cameras onto one plane."}),
                "fill_missing": ("BOOLEAN", {"default": False,
                                             "tooltip": "Infer poses of unregistered frames from their neighbours."}),
                "use_mask_for_poses": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Features only on the subject (needs masks)."}),
                "pose_check": (["off", "auto", "always"], {"default": "off",
                               "tooltip": "Consistency check (and correction) of the poses against the masks."}),
                "sfm_max_side": ("INT", {"default": SFM_MAX_SIDE, "min": 320, "max": 4096, "step": 32,
                                         "tooltip": "akaze / aliked_lightglue: features are detected on the frames "
                                                    "scaled to fit this size (no upscaling; aliked_lightglue rounds "
                                                    "to multiples of 32). The app: 1600."}),
                "max_features": ("INT", {"default": SFM_MAX_FEATURES, "min": 500, "max": 32000, "step": 500,
                                         "tooltip": "akaze: the strongest features kept per frame. More helps "
                                                    "low-texture subjects (skin, plain clothes); slower matching. "
                                                    "The app: 4000. aliked_lightglue: picks the model with the "
                                                    "most keypoints up to this (1024, 2048, 3072 or 3584)."}),
            },
            "optional": {
                "frame_times": ("ORBIT_TIMES",),
                "masks": ("MASK",),
            },
        }

    def poses(self, images, preset, estimator, match_window, closed_loop, flat_orbit, fill_missing,
              use_mask_for_poses, pose_check, sfm_max_side=SFM_MAX_SIDE, max_features=SFM_MAX_FEATURES,
              frame_times=None, masks=None):
        from gausify.sfm import model_stats, run_sfm

        v = apply_preset("poses", preset, {"match_window": match_window, "closed_loop": closed_loop})
        opts = {
            "estimator": estimator,
            "window": int(v["match_window"]),
            "closed_loop": bool(v["closed_loop"]),
            "flat_orbit": bool(flat_orbit),
            "fill_missing": bool(fill_missing),
            "use_mask": bool(use_mask_for_poses and masks is not None),
            "pose_check": pose_check if masks is not None else "off",
            # Image-sequence frames carry source indices, not timestamps: as job.ts,
            # no times then (orbit_times falls back to the name order).
            "times": list(frame_times.times)
            if isinstance(frame_times, OrbitTimes) and not frame_times.from_images
            else None,
        }
        if estimator == "akaze" and (int(sfm_max_side), int(max_features)) != (SFM_MAX_SIDE, SFM_MAX_FEATURES):
            # Only when changed: the default cache keys stay those of earlier versions.
            opts["akaze"] = {"max_side": int(sfm_max_side), "max_features": int(max_features)}
        elif estimator == "aliked_lightglue":
            from gausify.sfm_lightglue import model_keypoints

            opts["akaze"] = {"max_side": int(sfm_max_side), "max_features": model_keypoints(int(max_features))}
        root, names, w, h, masked = prepare_dataset(images, masks)
        if len(names) < 3:
            raise ValueError("Too few images (at least 3).")
        cams = OrbitCameras(root=root, names=names, width=w, height=h, masked=masked)
        sparse = cams.sparse_dir
        key = content_hash(opts)
        stats_file = root / f"poses-{key}.json"
        current = root / "poses-current"
        if stats_file.exists() and current.exists() and current.read_text() == key and (sparse / "images.bin").exists():
            stats = json.loads(stats_file.read_text())
            cams.stats = stats
            return (cams, str(root), json.dumps(stats, indent=2))

        progress = Progress()
        lines: list[str] = []

        def say(s: str):
            lines.append(s)
            log.info("gausify poses: %s", s)

        if (root / "sparse").exists():
            shutil.rmtree(root / "sparse")
        if current.exists():
            current.unlink()
        sfm_progress = progress.sub(0.0, 0.7 if opts["pose_check"] != "off" else 1.0)
        times = opts["times"]
        model = run_cancellable(
            run_sfm,
            cams.images_dir,
            names,
            cams.masks_dir if opts["use_mask"] else None,
            estimator,
            opts["window"],
            opts["closed_loop"],
            opts["flat_orbit"],
            opts["fill_missing"],
            times if times is not None and len(times) == len(names) else None,
            progress=sfm_progress,
            cancel=interrupted,
            sparse_dir=sparse,
            log_fn=say,
            **({"max_side": opts["akaze"]["max_side"], "max_features": opts["akaze"]["max_features"]}
               if "akaze" in opts else {}),
        )
        raise_if_interrupted()
        stats = model_stats(model)
        if stats["registered"] < 3:
            raise RuntimeError(f"only {stats['registered']} cameras registered; the video can't be reconstructed")
        if opts["pose_check"] != "off" and masked:
            from gausify.posecheck import check_poses

            try:
                stats["check"] = run_cancellable(
                    check_poses, cams, opts["pose_check"], progress.sub(0.7, 1.0), interrupted,
                    despill=DESPILL, despill_strength=DESPILL_STRENGTH,
                )
            except Exception as e:  # as job.ts: the poses stay usable
                raise_if_interrupted()
                say(f"pose check failed, poses kept: {e}")
        stats["estimator"] = estimator
        stats["log"] = lines
        progress(1.0, "poses done")
        cams.stats = stats
        stats_file.write_text(json.dumps(stats, default=_json_default))
        current.write_text(key)
        return (cams, str(root), json.dumps(stats, indent=2, default=_json_default))


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
