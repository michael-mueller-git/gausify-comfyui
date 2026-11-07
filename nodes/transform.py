"""Gausify Orbit Center Splat and Transform Splat: put the subject at the
origin, upright, and move / rotate / scale a splat (gausify/splat_transform.py).
Beyond gausify-orbit: the app has no such step."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gausify.splat_transform import ANCHORS, UP_AXES

from .clean import resolve_ply
from .common import CATEGORY, run_cancellable

log = logging.getLogger("gausify.nodes")

TRANSFORM_TOOLTIP = ("The transform applied, as JSON (4x4 matrix, scale, rotation, translation): feed it to "
                     "Transform Splat to apply the same move to another splat of this subject (e.g. a retrained "
                     "one).")


def _write(src: Path, root: Path, suffix: str, M) -> tuple[str, int]:
    from gausify.splat_clean import Ply, read_ply, write_ply
    from gausify.splat_transform import apply_transform, transform_header

    ply = read_ply(src)
    out = src.with_name(f"{src.stem}_{suffix}.ply")
    write_ply(out, Ply(header=transform_header(ply.header, M), vertex=ply.vertex), apply_transform(ply.vertex, M))
    return out.relative_to(root).as_posix(), len(ply.vertex)


class GausifyCenterSplat:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ply_path", "transform")
    OUTPUT_TOOLTIPS = ("The centred splat (<splat>_centered.ply next to the input).", TRANSFORM_TOOLTIP)
    FUNCTION = "center"
    OUTPUT_NODE = True
    DESCRIPTION = ("Moves the subject to the origin (feet on the floor, or its centre), stands it upright along "
                   "the viewer's up axis and turns it to face the viewer like the first frame. The SfM frame is "
                   "arbitrary (origin, scale and up wherever the poses put them), so a raw splat opens off-centre "
                   "and tilted. Saves <splat>_centered.ply next to the input. Use it last: Clean Splat, Retouch "
                   "and Refine Poses need the splat in the cameras' frame.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {"forceInput": True}),
                "anchor": (ANCHORS, {"default": "feet", "tooltip": (
                    "What goes to the origin: feet (centred horizontally, lowest point on the floor), center "
                    "(median of the opaque splats), bbox_center (centre of their box) or none (rotate / scale "
                    "only).")}),
                "upright": ("BOOLEAN", {"default": True, "tooltip": (
                    "Rotate so the subject stands along up_axis. Up comes from the cameras (the orbit plane's "
                    "normal) when they are connected, else from the splat's longest axis (a standing person).")}),
                "up_axis": (list(UP_AXES), {"default": "+y", "tooltip": (
                    "The viewer's up: +y for ComfyUI Preview3D, three.js and glTF; -y for viewers that expect "
                    "raw COLMAP / 3DGS files (y down); +z for Blender and other Z-up tools.")}),
                "face_first_camera": ("BOOLEAN", {"default": True, "tooltip": (
                    "With cameras and upright: turn the subject so the viewer's default front view shows what "
                    "the first frame saw.")}),
                "min_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.05, "tooltip": (
                    "Only splats at least this opaque count for the centre and the extent (ignores faint "
                    "floaters and haze).")}),
                "height": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": (
                    "Scale the subject to this height along up (0 = keep the SfM scale, which has no real "
                    "unit; e.g. 1.75 for a person in metres).")}),
            },
            "optional": {
                "cameras": ("GAUSIFY_CAMERAS", {"tooltip": (
                    "The dataset the splat was trained on: a reliable up direction and the first frame's side. "
                    "Without it, up is guessed from the splat's shape.")}),
            },
        }

    def center(self, ply_path, anchor, upright, up_axis, face_first_camera, min_opacity, height, cameras=None):
        from gausify.splat_clean import read_ply
        from gausify.splat_transform import center_transform, transform_json

        src, root = resolve_ply(ply_path)
        views = None
        if cameras is not None:
            from gausify.posecheck.job import read_views

            sparse = Path(cameras.sparse_dir)
            views = read_views((sparse / "cameras.bin").read_bytes(), (sparse / "images.bin").read_bytes())
            views = sorted(views, key=lambda v: v.name)  # the first frame first
        ply = read_ply(src)
        M, info = center_transform(ply.vertex, anchor=anchor, upright=bool(upright), up_axis=up_axis,
                                   face_first_camera=bool(face_first_camera), min_opacity=float(min_opacity),
                                   height=float(height), views=views)
        rel, n = run_cancellable(_write, src, root, "centered", M)
        tr = {**transform_json(M), "info": info}
        text = (f"{rel}: {n} splats, anchor {anchor}, " + (f"upright ({up_axis}, up from {info['up']})"
                if upright else "not rotated") + f", height {info['height']:.3g}"
                + (f" (scaled x{info['scale']:.3g})" if info["scale"] != 1 else ""))
        log.info("gausify center: %s", text)
        return {"ui": {"text": [text]}, "result": (rel, json.dumps(tr, indent=2))}


class GausifyTransformSplat:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ply_path", "transform")
    OUTPUT_TOOLTIPS = ("The moved splat (<splat>_moved.ply next to the input).", TRANSFORM_TOOLTIP)
    FUNCTION = "transform"
    OUTPUT_NODE = True
    DESCRIPTION = ("Moves, rotates and scales a splat: p' = offset + scale * R p, with R = rotate x, then y, "
                   "then z (degrees, about the origin). Positions, orientations, sizes and the view-dependent "
                   "colour all follow, so the splat looks the same, just placed elsewhere. Saves "
                   "<splat>_moved.ply next to the input.")

    @classmethod
    def INPUT_TYPES(cls):
        off = {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}
        rot = {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}
        return {
            "required": {
                "ply_path": ("STRING", {"forceInput": True}),
                "offset_x": ("FLOAT", {**off, "tooltip": "Move along x (splat units; after rotate and scale)."}),
                "offset_y": ("FLOAT", {**off, "tooltip": "Move along y."}),
                "offset_z": ("FLOAT", {**off, "tooltip": "Move along z."}),
                "rotate_x": ("FLOAT", {**rot, "tooltip": "Rotate about the x axis (degrees), first."}),
                "rotate_y": ("FLOAT", {**rot, "tooltip": "Then about the y axis."}),
                "rotate_z": ("FLOAT", {**rot, "tooltip": "Then about the z axis."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.01,
                                    "tooltip": "Uniform scale about the origin."}),
            },
            "optional": {
                "transform": ("STRING", {"forceInput": True, "tooltip": (
                    "A transform output of Center Splat or Transform Splat, applied before this node's own "
                    "move: replays a centring on another splat of the same subject. Leave it unconnected "
                    "when the splat is already the centred one.")}),
            },
        }

    def transform(self, ply_path, offset_x, offset_y, offset_z, rotate_x, rotate_y, rotate_z, scale,
                  transform=None):
        from gausify.splat_transform import parse_transform, rotation_xyz, similarity, transform_json

        src, root = resolve_ply(ply_path)
        M = similarity(float(scale), rotation_xyz(rotate_x, rotate_y, rotate_z), (offset_x, offset_y, offset_z))
        before = parse_transform(transform)
        if before is not None:
            M = M @ before
        rel, n = run_cancellable(_write, src, root, "moved", M)
        tr = transform_json(M)
        text = (f"{rel}: {n} splats moved by ({', '.join(f'{x:.3g}' for x in tr['translation'])}), rotated "
                f"{tr['rotationDeg']:.3g} deg, scaled x{tr['scale']:.3g}")
        log.info("gausify transform: %s", text)
        return {"ui": {"text": [text]}, "result": (rel, json.dumps(tr, indent=2))}
