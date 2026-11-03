"""Pose check after the poses stage (masked datasets): two short training
screens, one on the even frames and one on the odd ones; each frame is compared
with the silhouette of the screen that never saw it (score.py). When the orbit
disagrees with itself, or always if asked, each frame's pose is aligned to that
screen's render (align.py), the screens are retrained on the new poses and the
alignment repeats for a few rounds.

Port of web/src/posecheck/job.ts (gausify-orbit). Differences forced by the
host: Brush runs as `brush-cli` (gausify.brush), so a screen is trained to a
.ply and its held-out frames are rendered from it in batches (each frame's
alignment only depends on its own renders, so the result is the same as the
TS's frame-by-frame visit).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ..types import OrbitCameras
from .align import (
    DEFAULT_ALIGN,
    AlignOptions,
    align_cost,
    align_step,
    apply_step,
    choose_pose,
    edge_offset,
    prepare_target,
    quat_from_mat,
    rotation_angle,
    rotation_from_quat,
)
from .image import fill_holes
from .score import frame_value, opening_radius, orbit_score, xor_score

log = logging.getLogger("gausify.posecheck")

# Frames rendered per brush-cli call (memory: each frame keeps its target pyramid).
RENDER_BATCH = 16


@dataclass(frozen=True)
class Budget:
    steps: int
    resolution: int


@dataclass(frozen=True)
class PoseCheckSettings:
    detect: Budget = Budget(600, 480)  # check screens
    refine: Budget = Budget(750, 640)  # correction screens
    rounds: int = 3
    steps: int = 2  # alignment steps (re-renders) per frame and round
    accept: float = 0.0  # relative cost margin to leave the start pose
    threshold: float = 0.028  # orbit score above which "auto" corrects
    analysis: int = 640  # frames compared at this size (longer side)
    fill_holes: bool = True
    warm_start: bool = True
    verify: bool = True
    align: AlignOptions = field(default_factory=lambda: DEFAULT_ALIGN)


POSE_CHECK = PoseCheckSettings()

_CAMEL = {
    "fillHoles": "fill_holes",
    "warmStart": "warm_start",
    "contourHuber": "contour_huber",
    "contourCut": "contour_cut",
    "interiorHuber": "interior_huber",
    "interiorCut": "interior_cut",
    "interiorWeight": "interior_weight",
    "minGradient": "min_gradient",
    "contourBias": "contour_bias",
}


def settings() -> PoseCheckSettings:
    """Development: `$GAUSIFY_POSECHECK` (JSON, TS or snake_case keys)
    overrides fields, like `localStorage["gausify.poseCheck"]` in the TS."""
    try:
        o = json.loads(os.environ.get("GAUSIFY_POSECHECK", "null"))
    except ValueError:
        o = None
    if not isinstance(o, dict):
        return POSE_CHECK
    kw = {_CAMEL.get(k, k): v for k, v in o.items() if k != "align"}
    for b in ("detect", "refine"):
        if b in kw:
            kw[b] = Budget(**{**getattr(POSE_CHECK, b).__dict__, **kw[b]})
    al = {_CAMEL.get(k, k): v for k, v in (o.get("align") or {}).items()}
    return replace(POSE_CHECK, **kw, align=replace(POSE_CHECK.align, **al))


# --- COLMAP binary (web/src/mesh/colmap.ts readViews, web/src/sfm/colmap.ts rewriteImages)

_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
_SINGLE_FOCAL = {0, 2, 3, 8, 9}


@dataclass
class View:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray


def read_views(cameras_bin: bytes, images_bin: bytes) -> list[View]:
    """Registered images with their cameras, sorted by name."""
    cams = {}
    o = 8
    for _ in range(struct.unpack_from("<Q", cameras_bin, 0)[0]):
        cid, model, width, height = struct.unpack_from("<IiQQ", cameras_bin, o)
        o += 24
        n = _PARAMS.get(model)
        if n is None:
            raise ValueError(f"unsupported COLMAP camera model {model}")
        p = struct.unpack_from(f"<{n}d", cameras_bin, o)
        o += 8 * n
        single = model in _SINGLE_FOCAL
        cams[cid] = dict(
            width=width,
            height=height,
            fx=p[0],
            fy=p[0] if single else p[1],
            cx=p[1] if single else p[2],
            cy=p[2] if single else p[3],
        )
    views = []
    o = 8
    for _ in range(struct.unpack_from("<Q", images_bin, 0)[0]):
        vals = struct.unpack_from("<I7dI", images_bin, o)
        o += 4 + 56 + 4
        end = images_bin.index(b"\0", o)
        name = images_bin[o:end].decode()
        o = end + 1
        npts = struct.unpack_from("<Q", images_bin, o)[0]
        o += 8 + 24 * npts
        cam = cams.get(vals[8])
        if cam is None:
            raise ValueError(f"image {name} references a missing camera")
        views.append(View(name=name, R=rotation_from_quat(vals[1:5]), t=np.array(vals[5:8], dtype=np.float64), **cam))
    return sorted(views, key=lambda v: v.name)


def rewrite_images(images_bin: bytes, edit: Callable[[dict], dict | None]) -> bytes:
    """Copy of images.bin where `edit` renames, moves or drops (None) each
    image ({"name", "q", "t"}); the rest of each record is unchanged."""
    parts = []
    count = 0
    o = 8
    for _ in range(struct.unpack_from("<Q", images_bin, 0)[0]):
        head = o + 4 + 7 * 8 + 4
        end = images_bin.index(b"\0", head)
        f = struct.unpack_from("<7d", images_bin, o + 4)
        rec = edit({"name": images_bin[head:end].decode(), "q": list(f[:4]), "t": list(f[4:])})
        body = end + 1
        nxt = body + 8 + struct.unpack_from("<Q", images_bin, body)[0] * 24
        if rec is not None:
            top = bytearray(images_bin[o:head])
            struct.pack_into("<7d", top, 4, *[float(x) for x in (*rec["q"], *rec["t"])])
            parts += [bytes(top), rec["name"].encode(), b"\0", images_bin[body:nxt]]
            count += 1
        o = nxt
    return struct.pack("<Q", count) + b"".join(parts)


def _stem(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


# --- dependencies (other modules; tests replace these) ------------------------


def _train(dataset_dir: Path, opts: dict, out_ply: Path, init_ply: Path | None, progress, cancel) -> dict:
    from .. import brush
    from ..train_config import to_cli_args, train_config

    args = to_cli_args(train_config(opts, True))
    return brush.train(dataset_dir, args, out_ply, init_ply=init_ply, progress=progress, cancel=cancel)


def _render(ply: Path, cameras: list[dict]) -> list[dict]:
    from .. import brush

    return brush.render(ply, cameras)


def _build_frames(cams: OrbitCameras, out_dir: Path, max_side: int, despill: bool, despill_strength: float, progress) -> Path:
    """Frames rebuilt with the cutout at `max_side` (buildAlpha(..., S.analysis)).
    Returns the folder with the RGBA <stem>.png files."""
    from ..alpha import build_alpha_dataset

    params = inspect.signature(build_alpha_dataset).parameters
    kw = {}
    if "progress" in params:
        kw["progress"] = progress
    if "max_side" in params:
        kw["max_side"] = max_side
    d = Path(build_alpha_dataset(cams, out_dir, despill, despill_strength, **kw))
    d = d / "images" if (d / "images").is_dir() else d
    if "max_side" not in params:  # fallback: downscale the built frames
        for p in sorted(d.glob("*.png")):
            im = Image.open(p)
            s = max_side / max(im.size)
            if s < 1:
                im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR).save(p)
    return d


def _load_frame(path: Path) -> np.ndarray:
    """Frame at analysis size: (h, w, 4) uint8 RGBA with the training alpha."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)


# --- the check ----------------------------------------------------------------


def check_poses(
    cams: OrbitCameras,
    mode: str = "auto",
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    despill: bool = False,
    despill_strength: float = 50.0,
) -> dict | None:
    """Checks (and in "always" mode, or "auto" when flagged, corrects) the
    poses in `cams.sparse_dir`. Returns the PoseCheckStats fields, or None when
    skipped. The corrected poses are written to images.bin (3D points kept)."""
    if mode not in ("auto", "always"):
        raise ValueError(f"mode must be 'auto' or 'always', not {mode!r}")
    t0 = time.perf_counter()
    S = settings()
    model = Path(cams.sparse_dir)
    if not cams.masked:
        log.info("pose check: skipped (no masks)")
        return None
    cameras_bin = (model / "cameras.bin").read_bytes()
    images_bin = (model / "images.bin").read_bytes()
    views = read_views(cameras_bin, images_bin)
    if len(views) < 8:
        log.info("pose check: skipped (%d cameras)", len(views))
        return None
    poses = {v.name: (v.R.copy(), v.t.copy()) for v in views}
    radius = opening_radius(S.analysis)
    rep = progress or (lambda f, m: None)

    def check_cancel():
        if cancel and cancel():
            raise InterruptedError("cancelled")

    screen_at = 0
    phase = {"start": 0, "screens": 2, "from": 0.05}

    def share(f: float) -> float:
        return phase["from"] + (1 - phase["from"]) * (screen_at - phase["start"] + f) / phase["screens"]

    work = Path(cams.root) / ".posecheck"
    shutil.rmtree(work, ignore_errors=True)
    try:
        rep(0, f"pose check: frames 0/{len(views)}")
        frames_dir = _build_frames(
            cams, work / "alpha", S.analysis, despill, despill_strength, lambda f, m="": rep(0.05 * f, f"pose check: frames {m}")
        )
        frames = [{"view": v, "index": i, "png": frames_dir / f"{_stem(v.name)}.png"} for i, v in enumerate(views)]
        # One dataset per fold.
        for k in (0, 1):
            d = work / f"fold{k}"
            (d / "sparse" / "0").mkdir(parents=True, exist_ok=True)
            (d / "images").mkdir(parents=True, exist_ok=True)
            for f in ("cameras.bin", "points3D.bin"):
                shutil.copyfile(model / f, d / "sparse" / "0" / f)
            for fr in frames:
                if fr["index"] % 2 == k:
                    shutil.copyfile(fr["png"], d / "images" / f"{_stem(fr['view'].name)}.png")
        (work / "plys").mkdir(parents=True, exist_ok=True)

        plys: list[Path | None] = [None, None]

        def screen(k: int, budget: Budget, label: str, visit_batch, warm: bool = False, first=None):
            """Trains fold k on the current poses, then visits the held-out frames
            (in batches of RENDER_BATCH) with the screen's .ply."""
            nonlocal screen_at
            check_cancel()
            d = work / f"fold{k}"
            names = {fr["view"].name for fr in frames if fr["index"] % 2 == k}

            def fold_edit(r):
                if r["name"] not in names:
                    return None
                R, t = poses[r["name"]]
                return {"name": f"{_stem(r['name'])}.png", "q": quat_from_mat(R), "t": list(t)}

            (d / "sparse" / "0" / "images.bin").write_bytes(rewrite_images(images_bin, fold_edit))
            (d / "init.ply").unlink(missing_ok=True)
            init = plys[k] if warm and plys[k] is not None else None
            held = [fr for fr in frames if fr["index"] % 2 != k]
            ply = work / "plys" / f"screen{screen_at}_fold{k}.ply"
            opts = {
                "total_steps": budget.steps,
                "sh_degree": 1,
                "max_splats": 1_000_000,
                "max_resolution": budget.resolution,
                "export_every": 0,
                "eval_split_every": 0,
                "ssim_weight": 0.2,
                "depth_distortion_weight": 10,
            }
            _train(
                d,
                opts,
                ply,
                init,
                lambda f, m="": rep(share(0.8 * f), f"pose check: {label}, screen {k + 1}/2 {m}".rstrip()),
                cancel,
            )
            if first:
                first(ply, held)
            for i in range(0, len(held), RENDER_BATCH):
                check_cancel()
                rep(share(0.8 + 0.2 * i / len(held)), f"pose check: {label}, frame {i + 1}/{len(held)}")
                visit_batch(ply, held[i : i + RENDER_BATCH])
            if S.warm_start:
                plys[k] = ply
            screen_at += 1

        def camera(fr, pose, w, h):
            v = fr["view"]
            K = {"fx": v.fx * w / v.width, "fy": v.fy * h / v.height, "cx": v.cx * w / v.width, "cy": v.cy * h / v.height}
            cam = {"R": pose[0], "t": pose[1], "f": K["fx"], "fx": K["fx"], "fy": K["fy"], "cx": K["cx"], "cy": K["cy"], "width": w, "height": h}
            return K, cam

        def render(ply, batch, batch_poses, sizes):
            ks, cs = zip(*(camera(fr, p, w, h) for fr, p, (h, w) in zip(batch, batch_poses, sizes)))
            out = _render(ply, list(cs))
            if len(out) != len(batch):
                raise RuntimeError("Brush returned no splats to render")
            return list(ks), out

        def measure(label: str) -> list[float]:
            values = [0.0] * len(frames)
            for k in (0, 1):

                def visit(ply, batch):
                    rgbas = [_load_frame(fr["png"]) for fr in batch]
                    ks, rs = render(ply, batch, [poses[fr["view"].name] for fr in batch], [a.shape[:2] for a in rgbas])
                    for fr, rgba, r in zip(batch, rgbas, rs):
                        mask = (rgba[..., 3] > 127).astype(np.uint8)
                        if S.fill_holes:
                            mask = fill_holes(mask)
                        values[fr["index"]] = frame_value(xor_score(np.asarray(r["rgba"])[..., 3], mask, radius))

                screen(k, S.detect, label, visit)
            return values

        before = measure("measuring")
        score = orbit_score(before)
        flagged = score >= S.threshold
        log.info(
            "pose check: orbit score %.2f%% (threshold %.1f%%), %d steps x %d px",
            100 * score, 100 * S.threshold, S.detect.steps, S.detect.resolution,
        )
        result = {
            "score": score,
            "threshold": S.threshold,
            "frames": before,
            "flagged": flagged,
            "rounds": 0,
            "after": None,
            "afterFrames": None,
            "seconds": 0.0,
        }
        if mode == "auto" and not flagged:
            result["seconds"] = time.perf_counter() - t0
            return result

        phase = {"start": screen_at, "screens": 2 * S.rounds + (2 if S.verify else 0), "from": 0.0}
        for rnd in range(1, S.rounds + 1):
            label = f"round {rnd}/{S.rounds}"
            nxt: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            moved = 0.0
            bias = 0.0

            def halo(ply, held, rnd=rnd):
                # The screen's halo: median edge offset over the frames it did not see.
                nonlocal bias
                offsets = []
                for i in range(0, len(held), RENDER_BATCH):
                    check_cancel()
                    batch = held[i : i + RENDER_BATCH]
                    rgbas = [_load_frame(fr["png"]) for fr in batch]
                    _, rs = render(ply, batch, [poses[fr["view"].name] for fr in batch], [a.shape[:2] for a in rgbas])
                    for rgba, r in zip(rgbas, rs):
                        offsets.append(edge_offset(np.asarray(r["rgba"])[..., 3], prepare_target(rgba, 1)))
                offsets.sort()
                bias = offsets[len(offsets) >> 1] if offsets else 0.0
                log.info("pose check: round %d, screen halo %.2f px", rnd, bias)

            def visit(ply, batch):
                nonlocal moved
                rgbas = [_load_frame(fr["png"]) for fr in batch]
                sizes = [a.shape[:2] for a in rgbas]
                targets = [prepare_target(a, S.align.levels) for a in rgbas]
                o = replace(S.align, contour_bias=bias)
                cur = [poses[fr["view"].name] for fr in batch]
                visited: list[list] = [[] for _ in batch]
                costs: list[list[float]] = [[] for _ in batch]
                for s in range(S.steps + 1):
                    check_cancel()
                    ks, rs = render(ply, batch, cur, sizes)
                    for j, (K, r) in enumerate(zip(ks, rs)):
                        rr = {"rgba": r["rgba"], "depth": r["depth"]}
                        visited[j].append(cur[j])
                        if s == S.steps:  # the end pose: measured, not moved
                            costs[j].append(align_cost(rr, targets[j], K, o))
                            continue
                        step = align_step(rr, targets[j], K, o)
                        costs[j].append(step.cost)
                        cur[j] = apply_step(cur[j][0], cur[j][1], step.dx)
                for j, fr in enumerate(batch):
                    kept = choose_pose(costs[j], S.accept)
                    nxt[fr["view"].name] = visited[j][kept]
                    moved += rotation_angle(visited[j][kept][0], poses[fr["view"].name][0])

            for k in (0, 1):
                screen(k, S.refine, label, visit, warm=S.warm_start, first=halo)
            # Both folds aligned to screens trained on the same poses, then applied.
            poses.update(nxt)
            log.info("pose check: round %d, mean rotation change %.3f deg", rnd, np.degrees(moved / len(frames)))
            result["rounds"] = rnd

        # The model keeps its points; only the poses change.
        def final_edit(r):
            p = poses.get(r["name"])
            return {**r, "q": quat_from_mat(p[0]), "t": list(p[1])} if p else r

        images_bin = rewrite_images(images_bin, final_edit)
        (model / "images.bin").write_bytes(images_bin)
        if S.verify:
            after = measure("verifying")
            result["after"] = orbit_score(after)
            result["afterFrames"] = after
            log.info("pose check: orbit score after %d rounds %.2f%%", S.rounds, 100 * result["after"])
        result["seconds"] = time.perf_counter() - t0
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)
