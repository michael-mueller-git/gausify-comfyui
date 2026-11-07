"""Pose refinement against a trained splat (beyond the app).

Brush can't optimise the cameras while training, so leftover pose error stays
in the splat as ghosting and soft detail. This pass corrects the poses
against a trained splat, after which the splat is trained again:

1. render the splat from every camera (colour + expected depth), at the
   ALIKED-LightGlue network size;
2. match each render with its photo (RaCo-ALIKED + LightGlue);
3. lift the render's keypoints to 3D with the render depth (opaque pixels
   off depth edges only) and solve the photo's pose from these 2D-3D matches
   (PnP-RANSAC from the current pose, then Levenberg-Marquardt on the inliers);
4. keep the new pose only if it lowers the median reprojection error and
   stays within a rotation / camera-shift limit.

The render is the consensus of all frames. Where a frame's pose is off, its
photo sits shifted against the render, and PnP measures that shift. The
splat also partly fitted the frame itself, so one pass under-corrects.
Train, refine, train again converges further. The world frame, intrinsics
and 3D points are unchanged.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log_ = logging.getLogger("gausify.pose_refine")

#: Network long side for the render <-> photo matching.
DEFAULT_MAX_SIDE = 1024
#: Keypoints (selects the ALIKED-LightGlue model).
DEFAULT_FEATURES = 2048
#: Minimum PnP inliers for a pose to be considered.
DEFAULT_MIN_INLIERS = 40
#: Largest accepted rotation change (degrees).
DEFAULT_MAX_ROTATION = 2.0
#: Largest accepted camera move, in percent of the orbit radius.
DEFAULT_MAX_SHIFT = 3.0
#: RANSAC reprojection threshold (network pixels).
RANSAC_PX = 2.0
#: A new pose is used only when its median error drops by this factor and by
#: at least MIN_GAIN_PX (network px). Below that, the change is within the
#: matching noise, and small rotations and sideways shifts look alike: such
#: "corrections" moved correct poses by up to ~1 deg in tests.
GAIN_RATIO = 1.5
MIN_GAIN_PX = 0.25
#: Render pixels used for 3D points: alpha above this ...
MIN_ALPHA = 0.9
#: ... and depth within this fraction across the 3x3 neighbourhood (no depth edges).
MAX_DEPTH_STEP = 0.02
#: Views rendered per brush-cli call (memory).
RENDER_CHUNK = 16


@dataclass
class Refined:
    R: np.ndarray
    t: np.ndarray
    accepted: bool
    reason: str
    inliers: int = 0
    matches: int = 0
    before_px: float = float("nan")
    after_px: float = float("nan")
    rotation_deg: float = 0.0
    shift: float = 0.0  # fraction of the orbit radius


def sample_depth(depth: np.ndarray, alpha: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Expected depth at keypoints `uv` ((M, 2), pixel centres at +0.5);
    NaN where the render isn't opaque or the depth jumps (silhouette and
    self-occlusion edges, where the expected depth mixes surfaces)."""
    h, w = depth.shape
    x = np.clip(np.floor(uv[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.floor(uv[:, 1]).astype(np.int64), 0, h - 1)
    d = np.asarray(depth, dtype=np.float64)
    pad = np.pad(d, 1, mode="edge")
    stack = np.stack([pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    lo, hi = stack.min(0), stack.max(0)
    z = d[y, x]
    ok = (np.asarray(alpha)[y, x] > MIN_ALPHA) & (z > 0) & ((hi[y, x] - lo[y, x]) <= MAX_DEPTH_STEP * z)
    return np.where(ok, z, np.nan)


def unproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World points of pixels `uv` at camera depth `z` (pose world->cam R, t)."""
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - t[None]) @ R  # R^T (pc - t)


def reprojection(Xw: np.ndarray, uv: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pc = Xw @ R.T + t[None]
    z = np.maximum(pc[:, 2], 1e-9)
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.hypot(u - uv[:, 0], v - uv[:, 1])


def pose_from_matches(K: np.ndarray, R: np.ndarray, t: np.ndarray, uv_render: np.ndarray, z_render: np.ndarray,
                      uv_photo: np.ndarray, radius: float, min_inliers: int = DEFAULT_MIN_INLIERS,
                      max_rotation: float = DEFAULT_MAX_ROTATION, max_shift: float = DEFAULT_MAX_SHIFT) -> Refined:
    """New pose of the photo from render <-> photo matches (render keypoints
    with their depth, NaN = unusable), or the old one with the reason."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    ok = np.isfinite(z_render)
    keep = Refined(R=R, t=t, accepted=False, reason="", matches=int(ok.sum()))
    if ok.sum() < max(6, min_inliers):
        keep.reason = f"too few usable matches ({int(ok.sum())})"
        return keep
    Xw = unproject(uv_render[ok], z_render[ok], K, R, t)
    uv = uv_photo[ok].astype(np.float64)
    before = reprojection(Xw, uv, K, R, t)
    rvec0, _ = cv2.Rodrigues(R)
    found, rvec, tvec, inl = cv2.solvePnPRansac(
        Xw, uv, K, None, rvec0.copy(), t.reshape(3, 1).copy(), useExtrinsicGuess=True,
        iterationsCount=300, reprojectionError=RANSAC_PX, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not found or inl is None or len(inl) < min_inliers:
        keep.reason = f"PnP failed ({0 if inl is None else len(inl)} inliers)"
        return keep
    inl = inl.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(Xw[inl], uv[inl], K, None, rvec, tvec)
    R2, _ = cv2.Rodrigues(rvec)
    t2 = tvec.reshape(3)
    after = reprojection(Xw[inl], uv[inl], K, R2, t2)
    b_med, a_med = float(np.median(before[inl])), float(np.median(after))
    rot = math.degrees(float(np.arccos(np.clip((np.trace(R2 @ R.T) - 1) / 2, -1, 1))))
    shift = float(np.linalg.norm(-R2.T @ t2 - (-R.T @ t))) / max(radius, 1e-12)
    res = Refined(R=R2, t=t2, accepted=True, reason="", inliers=int(len(inl)), matches=int(ok.sum()),
                  before_px=b_med, after_px=a_med, rotation_deg=rot, shift=shift)
    if rot > max_rotation or shift * 100 > max_shift:
        reason = f"correction too large ({rot:.2f} deg, {shift * 100:.2f}% of the radius)"
    elif b_med < GAIN_RATIO * a_med or b_med - a_med < MIN_GAIN_PX:
        reason = f"no gain ({b_med:.2f} -> {a_med:.2f} px)"
    else:
        return res
    return Refined(R=R, t=t, accepted=False, reason=reason, inliers=res.inliers, matches=res.matches,
                   before_px=b_med, after_px=a_med, rotation_deg=rot, shift=shift)


def _link_tree(src: Path, dst: Path) -> None:
    """Copy of a folder of files, hard-linked where possible."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            try:
                os.link(f, dst / f.name)
            except OSError:
                shutil.copyfile(f, dst / f.name)


def _read_rgb_mask(cams, name: str, size: tuple[int, int], use_mask: bool):
    from .sfm_akaze.features import _read_mask, _read_rgb

    rgb = _read_rgb(Path(cams.images_dir) / name)
    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if size[0] < W else cv2.INTER_LINEAR)
    mask = None
    if use_mask:
        mp = Path(cams.masks_dir) / (Path(name).stem + ".png")
        if mp.is_file():
            mask = cv2.resize(_read_mask(mp, W, H), size, interpolation=cv2.INTER_NEAREST) > 127
    return small, mask


def refine_poses(cams, ply, out_root: Path, max_side: int = DEFAULT_MAX_SIDE,
                 max_features: int = DEFAULT_FEATURES, min_inliers: int = DEFAULT_MIN_INLIERS,
                 max_rotation: float = DEFAULT_MAX_ROTATION, max_shift: float = DEFAULT_MAX_SHIFT,
                 progress: Callable[[float, str], None] | None = None, cancel: Callable[[], bool] | None = None,
                 log: Callable[[str], None] | None = None, sessions: dict | None = None,
                 render: Callable | None = None) -> dict:
    """Refines the poses of `cams` against the splat `ply` and writes a new
    dataset at `out_root` (images/ and masks/ hard-linked, sparse/0 with the
    new images.bin). `sessions` / `render`: replacements for the
    ALIKED-LightGlue sessions and brush.render (tests). Returns the stats."""
    from . import sfm_lightglue as lg
    from .posecheck.align import quat_from_mat
    from .posecheck.job import read_views, rewrite_images
    from .splat_clean import orbit_centre

    say = log or log_.info
    prog = progress or (lambda _f, _m: None)

    def check():
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")

    sparse = Path(cams.sparse_dir)
    views = read_views((sparse / "cameras.bin").read_bytes(), (sparse / "images.bin").read_bytes())
    if len(views) < 3:
        raise ValueError("too few registered cameras to refine")
    _, radius = orbit_centre(views)
    if render is None:
        from .brush import render as render  # noqa: PLW0127
    if sessions is None:
        prog(0.0, "loading ALIKED-LightGlue")
        sessions = lg._sessions(max_features, "auto", lambda f, m: prog(0.0, m), say)
    use_mask = bool(getattr(cams, "masked", False))

    results: dict[str, Refined] = {}
    n = len(views)
    for c0 in range(0, n, RENDER_CHUNK):
        check()
        chunk = views[c0 : c0 + RENDER_CHUNK]
        sizes, Ks, cam_json = [], [], []
        for v in chunk:
            w, h = lg.infer_size(v.width, v.height, max_side)
            sx, sy = w / v.width, h / v.height
            K = np.array([[v.fx * sx, 0, v.cx * sx], [0, v.fy * sy, v.cy * sy], [0, 0, 1.0]])
            sizes.append((w, h))
            Ks.append(K)
            cam_json.append({"R": v.R, "t": v.t, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                             "width": w, "height": h})
        prog(0.05 + 0.9 * c0 / n, f"rendering {c0 + 1}-{c0 + len(chunk)}/{n}")
        renders = render(ply, cam_json)
        check()
        pairs, masks = [], []
        for v, (w, h), r in zip(chunk, sizes, renders):
            rgb = np.clip(np.asarray(r["rgba"])[..., :3] * 255 + 0.5, 0, 255).astype(np.uint8)
            photo, mask = _read_rgb_mask(cams, v.name, (w, h), use_mask)
            pairs.append((lg.to_input(rgb), lg.to_input(photo)))
            masks.append(mask)
        prog(0.05 + 0.9 * (c0 + len(chunk) / 2) / n, f"matching {c0 + 1}-{c0 + len(chunk)}/{n}")
        matched = lg.match_images(sessions, pairs)
        for v, K, r, mask, (ka, kb, m) in zip(chunk, Ks, renders, masks, matched):
            uv_r, uv_p = ka[m[:, 0]].astype(np.float64), kb[m[:, 1]].astype(np.float64)
            if mask is not None and len(uv_p):
                h, w = mask.shape
                px = np.clip(np.floor(uv_p[:, 0]).astype(np.int64), 0, w - 1)
                py = np.clip(np.floor(uv_p[:, 1]).astype(np.int64), 0, h - 1)
                on = mask[py, px]
                uv_r, uv_p = uv_r[on], uv_p[on]
            z = sample_depth(np.asarray(r["depth"]), np.asarray(r["alpha"]), uv_r)
            results[v.name] = pose_from_matches(K, v.R, v.t, uv_r, z, uv_p, radius, min_inliers, max_rotation,
                                                max_shift)
    check()

    # the new dataset: same images and masks, corrected images.bin
    tmp = out_root.with_name(out_root.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    _link_tree(Path(cams.images_dir), tmp / "images")
    if Path(cams.masks_dir).is_dir():
        _link_tree(Path(cams.masks_dir), tmp / "masks")
    (tmp / "sparse" / "0").mkdir(parents=True)
    for f in sparse.iterdir():
        if f.is_file() and f.name != "images.bin":
            shutil.copyfile(f, tmp / "sparse" / "0" / f.name)

    def edit(rec):
        r = results.get(rec["name"])
        if r is not None and r.accepted:
            rec = {**rec, "q": list(quat_from_mat(r.R)), "t": [float(x) for x in r.t]}
        return rec

    (tmp / "sparse" / "0" / "images.bin").write_bytes(rewrite_images((sparse / "images.bin").read_bytes(), edit))
    shutil.rmtree(out_root, ignore_errors=True)
    tmp.rename(out_root)

    acc = [r for r in results.values() if r.accepted]
    stats = {
        "views": n,
        "refined": len(acc),
        "medianErrorBefore": _round(np.median([r.before_px for r in results.values() if np.isfinite(r.before_px)]
                                              or [np.nan])),
        "medianErrorAfter": _round(np.median([(r.after_px if r.accepted else r.before_px)
                                              for r in results.values() if np.isfinite(r.before_px)] or [np.nan])),
        "maxRotationDeg": _round(max((r.rotation_deg for r in acc), default=0.0), 3),
        "maxShiftPercent": _round(max((r.shift * 100 for r in acc), default=0.0), 3),
        "frames": {name: ({"inliers": r.inliers, "before": _round(r.before_px), "after": _round(r.after_px),
                           "rotationDeg": _round(r.rotation_deg, 3), "shiftPercent": _round(r.shift * 100, 3)}
                          if r.accepted else {"kept": r.reason}) for name, r in results.items()},
    }
    say(f"refined {len(acc)}/{n} poses: median error {stats['medianErrorBefore']} -> "
        f"{stats['medianErrorAfter']} px (network size), largest correction {stats['maxRotationDeg']} deg / "
        f"{stats['maxShiftPercent']}% of the radius")
    for name, r in results.items():
        if not r.accepted:
            say(f"{name}: kept ({r.reason})")
    prog(1.0, "poses refined")
    return stats


def _round(x, nd: int = 2):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None
