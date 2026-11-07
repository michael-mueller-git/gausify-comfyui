"""Gausify Orbit Clean Splat: removes floaters and background from a trained
splat with the dataset's cameras (and masks). Beyond gausify-orbit: the app
has no such step. See gausify/splat_clean.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from gausify.types import OrbitCameras

from .common import CATEGORY, Progress, interrupted, output_dir, raise_if_interrupted, run_cancellable

log = logging.getLogger("gausify.nodes")

#: Views rendered for the PSNR check (evenly spaced).
CHECK_VIEWS = 12


def resolve_ply(ply_path: str) -> tuple[Path, Path]:
    """(absolute path, output root) of a ply_path from Train Splat (relative to
    the output folder) or an absolute path inside it."""
    root = output_dir().resolve()
    p = Path(ply_path)
    full = (p if p.is_absolute() else root / p).resolve()
    if not full.is_file():
        raise FileNotFoundError(f"splat not found: {full}")
    return full, root


def _load_u8(path: Path, gray: bool) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L" if gray else "RGB"))


class GausifyCleanSplat:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("ply_path", "num_splats", "stats")
    FUNCTION = "clean"
    OUTPUT_NODE = True
    DESCRIPTION = ("Removes splats that can't belong to the subject: outside the masks' visual hull (masked "
                   "datasets), unseen by every camera, far from the orbit, nearly transparent or huge. Saves "
                   "<splat>_clean.ply next to the input. Not part of the app.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cameras": ("GAUSIFY_CAMERAS",),
                "ply_path": ("STRING", {"forceInput": True}),
                "hull_carve": ("BOOLEAN", {"default": True, "tooltip": (
                    "Masked datasets: remove splats whose centre falls outside the subject mask in too many of "
                    "the views that see it (floaters, ghost silhouette, background).")}),
                "hull_tolerance": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": (
                    "Fraction of the views a splat may fall outside the mask in and still be kept. Higher "
                    "forgives masks that miss parts (hands, hair) in some frames; lower carves harder.")}),
                "mask_margin": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": (
                    "Masks are grown by this % of the image's long side before carving (protects the edges, "
                    "hair, small pose errors).")}),
                "drop_unseen": ("BOOLEAN", {"default": True,
                                            "tooltip": "Remove splats that no camera sees (behind or beside every view)."}),
                "max_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": (
                    "Remove splats farther from the orbit centre than this x the orbit radius (0 = off; "
                    "1.0 = the camera circle, e.g. 1.5 to drop distant background without masks).")}),
                "min_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.005,
                                          "tooltip": "Remove splats less opaque than this (0 = off; e.g. 0.01)."}),
                "max_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": (
                    "Remove splats whose largest axis exceeds this x the orbit radius (0 = off; e.g. 0.1 "
                    "drops big blurry blobs).")}),
                "check_psnr": ("BOOLEAN", {"default": True, "tooltip": (
                    f"Render {CHECK_VIEWS} training views before and after and report their PSNR: a clear drop "
                    "means the clean-up removed part of the subject (loosen hull_tolerance / mask_margin).")}),
            },
        }

    def clean(self, cameras: OrbitCameras, ply_path, hull_carve, hull_tolerance, mask_margin, drop_unseen,
              max_distance, min_opacity, max_size, check_psnr):
        from gausify.posecheck.job import read_views
        from gausify.splat_clean import clean_splat, render_psnr, spaced

        src, root = resolve_ply(ply_path)
        out = src.with_name(f"{src.stem}_clean.ply")
        sparse = Path(cameras.sparse_dir)
        views = read_views((sparse / "cameras.bin").read_bytes(), (sparse / "images.bin").read_bytes())
        masked = bool(cameras.masked)
        masks_dir = Path(cameras.masks_dir) if masked else None

        def mask_of(v):
            if masks_dir is None:
                return None
            f = masks_dir / f"{Path(v.name).stem}.png"
            return _load_u8(f, gray=True) if f.is_file() else None

        progress = Progress()
        stats = run_cancellable(
            clean_splat, src, out, views, mask_of if masked else None,
            hull_carve=bool(hull_carve) and masked, hull_tolerance=float(hull_tolerance),
            mask_margin=float(mask_margin) / 100.0, drop_unseen=bool(drop_unseen),
            max_distance=float(max_distance), min_opacity=float(min_opacity), max_size=float(max_size),
            progress=progress.sub(0.0, 0.8 if check_psnr else 1.0), cancel=interrupted,
        )
        raise_if_interrupted()
        if hull_carve and not masked:
            stats["note"] = "hull carving needs masks: skipped"
        if check_psnr and stats["kept"]:
            import cv2

            from gausify.alpha import ALPHA_ERODE, erode

            def target(v, w, h):
                rgb = _load_u8(Path(cameras.images_dir) / v.name, gray=False)
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                m = mask_of(v)
                if m is None:
                    return rgb
                a = cv2.resize(erode(m, ALPHA_ERODE), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
                return rgb * (a[..., None] / 255.0)  # premultiplied, like the render

            try:
                progress.sub(0.8, 1.0)(0.0, "clean: PSNR check")
                before, after = run_cancellable(render_psnr, [src, out], spaced(views, CHECK_VIEWS), target)
                stats["psnr"] = {"before": round(before, 3), "after": round(after, 3),
                                 "change": round(after - before, 3)}
            except Exception as e:  # the cleaned splat stays usable
                raise_if_interrupted()
                log.warning("gausify clean: PSNR check failed: %s", e)
                stats["psnr"] = {"error": str(e)}
        raise_if_interrupted()
        progress(1.0, "clean done")
        rel = out.relative_to(root).as_posix()
        removed = stats["input"] - stats["kept"]
        text = f"{rel}: {stats['kept']} splats ({removed} removed: " + (
            ", ".join(f"{k} {v}" for k, v in stats["removed"].items() if v) or "none") + ")"
        p = stats.get("psnr", {})
        if "change" in p:
            text += f", training-view PSNR {p['before']:.2f} -> {p['after']:.2f} dB"
        log.info("gausify clean: %s", text)
        return {"ui": {"text": [text]}, "result": (rel, int(stats["kept"]), json.dumps(stats, indent=2))}
