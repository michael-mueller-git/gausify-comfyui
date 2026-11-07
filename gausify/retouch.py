"""Retouch: fine-tune a trained splat with restored renders of new angles.

Port of gausify-orbit's Retouch (docs/retouch.md):
- src/lib/retouch/trajectory.ts: the orbit presets of the Retouch panel
  (presetPath: arc below/above, 360 degree loop below/above, up-and-down
  wave), clipLens, clipFrames, peakSpeed. The view-based motions (from this
  view, turns from the 3D view, recorded motion) need the 3D viewer and are
  not ported; nor is the viewer pivot.
- src/lib/retouch/frames.ts: pickRestoredFrames, restoredLimit.
- src/lib/retouch/colmap.ts: appendCameras, appendImages.
- web/src/host/retouch.ts: applyRetouch (dataset of the photos plus the
  picked restored frames with the render's eroded alpha, fine-tune from the
  splat as init.ply: fewer steps, lr-mean 2e-6, growth stops at half).

Poses are COLMAP world->camera (qvec (w, x, y, z), tvec), camera looking down
+Z with +Y down.
"""

from __future__ import annotations

import math
import shutil
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: RetouchPanel.tsx defaults.
DEFAULT_STEPS = 3000
LR_MEAN = 2e-6
DEFAULT_TILT = 30.0
MAX_SPEED = 30.0
#: frames.ts: restored frames up to 25% of the photo count.
DEFAULT_RESTORED_RATIO = 0.25
#: trajectory.ts DEFAULT_CLIP.
DEFAULT_FRAMES, DEFAULT_LONG_SIDE, DEFAULT_HOLD, DEFAULT_RAMP = 97, 960, 0.0, 0.4

MOTIONS = ["wave", "orbit_below", "orbit_above", "arc_below", "arc_above"]


# ---------------------------------------------------------------- algebra


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def mat_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    q = np.array(q) / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def slerp(a, b, u: float) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    s = math.sin(th)
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / s


def quat_angle(a, b) -> float:
    return 2 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / (n or 1.0)


def axis_angle(axis, ang: float) -> np.ndarray:
    x, y, z = _unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    t = 1 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])  # fmt: skip


@dataclass
class Pose:
    qvec: np.ndarray  # world -> camera, (w, x, y, z)
    tvec: np.ndarray
    name: str | None = None  # photo name (anchor: exactly that photo's pose)


def to_world(p: Pose) -> tuple[np.ndarray, np.ndarray]:
    """(camera -> world rotation, camera centre)."""
    r = quat_to_mat(p.qvec)
    rt = r.T
    return rt, -(rt @ np.asarray(p.tvec, np.float64))


def from_world(rot: np.ndarray, center: np.ndarray, name: str | None = None) -> Pose:
    r = rot.T
    return Pose(qvec=mat_to_quat(r), tvec=-(r @ center), name=name)


def orbit_frame(cams: Sequence[Pose]) -> tuple[np.ndarray, np.ndarray]:
    """Orbit centre (the point closest to all optical axes) and "up" (mean of
    the cameras' -Y)."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    up = np.zeros(3)
    for c in cams:
        rot, center = to_world(c)
        fwd = rot[:, 2]
        p = np.eye(3) - np.outer(fwd, fwd)
        a += p
        b += p @ center
        up += -rot[:, 1]
    center = np.zeros(3) if abs(np.linalg.det(a)) < 1e-12 else np.linalg.solve(a, b)
    return center, _unit(up)


# ------------------------------------------------------------ trajectories


def profile(u: float, hold: float, ramp: float) -> float:
    """0 at the ends (anchors), 1 in the middle, cosine ramps."""
    d = min(u, 1 - u)
    if d <= hold:
        return 0.0
    if d >= hold + ramp:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * (d - hold) / ramp)


def clip_frames(seconds: float, fps: float) -> int:
    """Frames of a `seconds` clip at `fps`, at the nearest 8n + 1 (LTX), at least 17."""
    n = _js_round((seconds * fps - 1) / 8)
    return max(2, n) * 8 + 1


def _js_round(x: float) -> int:
    return int(math.floor(x + 0.5))


@dataclass
class Lens:
    width: int
    height: int
    fx: float
    fy: float


def clip_lens(lens: Lens, long_side: int) -> tuple[tuple[int, int], dict]:
    """Clip size (multiples of 32, long side ~long_side, the lens's aspect)
    and intrinsics with the principal point at the centre; the scale covers
    both sides (the excess is cropped equally)."""
    s = long_side / max(lens.width, lens.height)
    w = max(32, _js_round(lens.width * s / 32) * 32)
    h = max(32, _js_round(lens.height * s / 32) * 32)
    k = max(w / lens.width, h / lens.height)
    return (w, h), {"fx": lens.fx * k, "fy": lens.fy * k, "cx": w / 2, "cy": h / 2}


def _along_orbit(cams: Sequence[Pose], s: float):
    m = len(cams)
    i0 = math.floor(s)
    f = s - i0
    a, b = cams[i0 % m], cams[(i0 + 1) % m]
    ra, ca = to_world(a)
    rb, cb = to_world(b)
    center = ca * (1 - f) + cb * f
    rot = quat_to_mat(slerp(mat_to_quat(ra.T), mat_to_quat(rb.T), f))
    anchor = a.name if abs(f) < 1e-6 else b.name if abs(f - 1) < 1e-6 else None
    return rot.T, center, anchor


def _orbit_walk(cams: Sequence[Pose], orbit_c, up, start: float, span: float):
    """Walks `span` photos from index `start` at constant angular speed around
    the centre (photos are rarely evenly spaced)."""
    m = len(cams)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(up, ref))
    e2 = np.cross(up, e1)
    k0, k1 = math.floor(start), math.ceil(start + span)
    az: list[float] = []
    for k in range(k0, k1 + 1):
        r = to_world(cams[k % m])[1] - orbit_c
        a = math.atan2(float(r @ e2), float(r @ e1))
        if az:
            a -= 2 * math.pi * _js_round((a - az[-1]) / (2 * math.pi))
        az.append(a)
    direction = (1 if az[-1] > az[0] else -1 if az[-1] < az[0] else 0) or 1
    for i in range(1, len(az)):
        az[i] = max(az[i], az[i - 1]) if direction > 0 else min(az[i], az[i - 1])

    def at(s: float) -> float:
        i = min(max(math.floor(s) - k0, 0), len(az) - 2)
        return az[i] + (az[i + 1] - az[i]) * (s - k0 - i)

    a0, a1 = at(start), at(start + span)

    def walk(u: float):
        if abs(a1 - a0) < 1e-9 or u <= 0 or u >= 1:
            return _along_orbit(cams, start + span * min(1.0, max(0.0, u)))
        target = a0 + (a1 - a0) * u
        i = 0
        while i < len(az) - 2 and direction * (az[i + 1] - target) < 0:
            i += 1
        seg = az[i + 1] - az[i]
        f = 0.0 if abs(seg) < 1e-12 else min(1.0, max(0.0, (target - az[i]) / seg))
        return _along_orbit(cams, k0 + i + f)

    return walk


@dataclass
class RetouchClip:
    """ComfyUI type GAUSIFY_RETOUCH_CLIP: a trajectory and its render size."""

    motion: str
    size: tuple[int, int]  # (width, height)
    intrinsics: dict  # fx, fy, cx, cy
    poses: list[Pose]
    fps: float
    stats: dict = field(default_factory=dict)
    #: The render's alpha per frame (N,H,W) float in [0,1]: the restored frames' alpha.
    alpha: np.ndarray | None = None


def preset_path(cams: Sequence[Pose], center: float, elev_deg: float, frames: int, span: float | None = None,
                shape: str = "arc", hold: float = DEFAULT_HOLD, ramp: float = DEFAULT_RAMP) -> list[Pose]:
    """presetPath without a pivot or radius: walks the orbit around photo
    `center` (fractional index), rising or dropping `elev_deg` in the middle
    ("arc") or up then down ("wave"), as a rigid rotation about the centre."""
    orbit_c, up = orbit_frame(cams)
    n = frames
    span = (n - 1) / 2 if span is None else span
    start = center - span / 2
    walk = _orbit_walk(cams, orbit_c, up, start, span)
    poses = []
    for i in range(n):
        u = i / (n - 1)
        rot, c, anchor = walk(u)
        rp = profile(u, hold, ramp)
        inner = min(1.0, max(0.0, (u - hold) / (1 - 2 * hold)))
        w = rp * math.sin(2 * math.pi * inner) if shape == "wave" else rp
        r = c - orbit_c
        radial = r - up * float(r @ up)
        if w != 0 and np.linalg.norm(radial) > 1e-9:
            R = axis_angle(np.cross(_unit(radial), up), math.radians(elev_deg) * w)
            c = orbit_c + R @ r
            rot = R @ rot
        poses.append(from_world(rot, c, anchor if w == 0 else None))
    return poses


def motion_path(cams: Sequence[Pose], motion: str, photo: int, tilt: float, frames: int) -> list[Pose]:
    """The Retouch panel's orbit presets (RetouchPanel.tsx draftPlan): arcs over a
    third of the orbit centred on photo `photo`; loops and the wave go all the
    way round, starting and ending at it."""
    m = len(cams)
    if motion in ("arc_below", "arc_above"):
        return preset_path(cams, photo, -tilt if motion == "arc_below" else tilt, frames, span=_js_round(m / 3))
    if motion in ("orbit_below", "orbit_above", "wave"):
        return preset_path(cams, photo + m / 2, tilt if motion == "orbit_above" else -tilt, frames, span=m,
                           shape="wave" if motion == "wave" else "arc", ramp=0.0 if motion == "wave" else 0.25)
    raise ValueError(f"unknown motion {motion!r} (expected one of {MOTIONS})")


def peak_speed(poses: Sequence[Pose], fps: float, center=None, window: int = 3) -> float:
    """Peak change of view in degrees per second (over `window` frames); with
    `center`, travel over the distance to it counts too (zooms)."""
    if center is not None:
        centers = [to_world(p)[1] for p in poses]
        dist = [float(np.linalg.norm(c - center)) for c in centers]
        floor = 0.05 * (sorted(dist)[len(dist) >> 1] if dist else 0) or 1e-9

        def change(i, j):
            return max(quat_angle(poses[i].qvec, poses[j].qvec),
                       float(np.linalg.norm(centers[j] - centers[i])) / max(floor, (dist[i] + dist[j]) / 2))
    else:
        def change(i, j):
            return quat_angle(poses[i].qvec, poses[j].qvec)
    peak = 0.0
    for i in range(window, len(poses)):
        peak = max(peak, math.degrees(change(i - window, i)) * fps / window)
    return peak


# ---------------------------------------------------------- restored frames


def restored_limit(photos: int, ratio: float) -> int:
    return max(1, _js_round(photos * ratio))


def pick_restored_frames(counts: Sequence[int], max_total: float) -> list[list[int]]:
    """Indices picked from each clip, at most `max_total` in total, each clip's
    quota proportional to its length (at least 1), evenly spaced at the middle
    of each segment (skipping the ends)."""
    total = sum(counts)
    if max_total >= total:
        return [list(range(n)) for n in counts]
    quota = [max(1, _js_round(max_total * n / total)) if n else 0 for n in counts]
    excess = sum(quota) - max(max_total, sum(1 for n in counts if n))
    while excess > 0:
        i = quota.index(max(quota))
        if quota[i] <= 1:
            break
        quota[i] -= 1
        excess -= 1
    out = []
    for c, n in enumerate(counts):
        k = min(quota[c], n)
        out.append([min(n - 1, math.floor((j + 0.5) * n / k)) for j in range(k)])
    return out


# ------------------------------------------------------------------ COLMAP

_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
_PINHOLE = 1


def append_cameras(data: bytes, cams: Sequence[dict]) -> tuple[bytes, list[int]]:
    """cameras.bin with PINHOLE cameras ({width, height, fx, fy, cx, cy})
    appended; returns (bytes, their ids)."""
    (n,) = struct.unpack_from("<Q", data, 0)
    o, max_id = 8, 0
    for _ in range(n):
        cid, model = struct.unpack_from("<ii", data, o)
        max_id = max(max_id, cid)
        if model not in _PARAMS:
            raise ValueError(f"unknown COLMAP camera model: {model}")
        o += 24 + 8 * _PARAMS[model]
    out = bytearray(struct.pack("<Q", n + len(cams)) + data[8:o])
    ids = []
    for k, c in enumerate(cams):
        cid = max_id + 1 + k
        out += struct.pack("<iiQQ4d", cid, _PINHOLE, int(c["width"]), int(c["height"]),
                           float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]))
        ids.append(cid)
    return bytes(out), ids


def append_images(data: bytes, images: Sequence[dict], camera_ids: Sequence[int]) -> bytes:
    """images.bin with images ({name, qvec, tvec, camera: index into
    camera_ids}) appended, without 2D points."""
    (n,) = struct.unpack_from("<Q", data, 0)
    o, max_id = 8, 0
    for _ in range(n):
        (iid,) = struct.unpack_from("<i", data, o)
        max_id = max(max_id, iid)
        o += 4 + 7 * 8 + 4
        o = data.index(b"\0", o) + 1
        (pts,) = struct.unpack_from("<Q", data, o)
        o += 8 + pts * 24
    out = bytearray(struct.pack("<Q", n + len(images)) + data[8:o])
    for k, im in enumerate(images):
        out += struct.pack("<i7di", max_id + 1 + k, *[float(x) for x in (*im["qvec"], *im["tvec"])],
                           camera_ids[im["camera"]])
        out += im["name"].encode() + b"\0" + struct.pack("<Q", 0)
    return bytes(out)


# ----------------------------------------------------------------- dataset


def read_lens(cameras_bin: bytes) -> Lens:
    (n,) = struct.unpack_from("<Q", cameras_bin, 0)
    if not n:
        raise ValueError("cameras.bin has no camera")
    _cid, model, width, height = struct.unpack_from("<iiQQ", cameras_bin, 8)
    k = _PARAMS.get(model)
    p = struct.unpack_from(f"<{k}d", cameras_bin, 32)
    single = model in (0, 2, 3, 8, 9)
    return Lens(width=int(width), height=int(height), fx=p[0], fy=p[0] if single else p[1])


def read_poses(images_bin: bytes) -> list[Pose]:
    """Registered photos, sorted by name (orbit order)."""
    (n,) = struct.unpack_from("<Q", images_bin, 0)
    o, out = 8, []
    for _ in range(n):
        vals = struct.unpack_from("<i7di", images_bin, o)
        o += 64
        end = images_bin.index(b"\0", o)
        name = images_bin[o:end].decode()
        o = end + 1
        (pts,) = struct.unpack_from("<Q", images_bin, o)
        o += 8 + pts * 24
        out.append(Pose(qvec=np.array(vals[1:5]), tvec=np.array(vals[5:8]), name=name))
    return sorted(out, key=lambda p: p.name)


def render_cameras(clip: RetouchClip) -> list[dict]:
    from .posecheck.align import rotation_from_quat

    w, h = clip.size
    k = clip.intrinsics
    return [{"R": rotation_from_quat(p.qvec), "t": np.asarray(p.tvec), "fx": k["fx"], "fy": k["fy"],
             "cx": k["cx"], "cy": k["cy"], "width": w, "height": h} for p in clip.poses]


def _eroded_alpha(alpha: np.ndarray, w: int, h: int) -> np.ndarray:
    """The render's alpha at (w, h), binarized (> 127 of 255) and eroded 1 px
    (3x3 min), like the real frames (retouch.ts erodedAlpha)."""
    import cv2

    from .alpha import erode

    a8 = np.clip(np.rint(np.asarray(alpha, np.float32) * 255), 0, 255).astype(np.uint8)
    if a8.shape != (h, w):
        a8 = cv2.resize(a8, (w, h), interpolation=cv2.INTER_LINEAR)
    return erode(np.where(a8 > 127, 255, 0).astype(np.uint8), 1)


def build_retouch_dataset(
    cams,
    clip: RetouchClip,
    restored: np.ndarray,
    alpha: np.ndarray,
    out_dir,
    restored_ratio: float = DEFAULT_RESTORED_RATIO,
    despill: bool = False,
    despill_strength: float = 100.0,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> tuple[Path, dict]:
    """Dataset of the photos (RGBA cut-outs when masked) plus the picked
    restored frames (uint8 (N,H,W,3), any size with the clip's aspect) with
    the render's eroded alpha, and the model with one camera per clip and the
    restored images appended. Returns (out_dir, stats)."""
    from PIL import Image

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    n_clip = len(clip.poses)
    if len(restored) != n_clip:
        raise ValueError(f"the restored video has {len(restored)} frames; expected {n_clip} (the clip's)")
    rh, rw = int(restored.shape[1]), int(restored.shape[2])
    cw, ch = clip.size
    if abs(rw / rh - cw / ch) > 0.02 * (cw / ch):
        raise ValueError(f"the restored video is {rw}x{rh}; it must keep the clip's aspect ratio ({cw}x{ch})")
    sparse = out / "sparse" / "0"
    if cams.masked:
        from .alpha import build_alpha_dataset

        build_alpha_dataset(cams, out, despill=despill, despill_strength=despill_strength, log=log,
                            progress=(lambda f, m: progress(0.6 * f, m)) if progress else None, cancel=cancel)
    else:
        shutil.copytree(cams.images_dir, out / "images")
        sparse.mkdir(parents=True, exist_ok=True)
        for f in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copyfile(Path(cams.sparse_dir) / f, sparse / f)
    photos = len(read_poses((sparse / "images.bin").read_bytes()))
    (picked,) = pick_restored_frames([n_clip], restored_limit(photos, restored_ratio))
    new = []
    for j, i in enumerate(picked):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        a = _eroded_alpha(alpha[i], rw, rh)
        rgba = np.dstack([np.asarray(restored[i], np.uint8)[..., :3], a])
        name = f"retouch_{i:05d}.png"
        Image.fromarray(rgba, "RGBA").save(out / "images" / name, compress_level=1)
        p = clip.poses[i]
        new.append({"name": name, "qvec": p.qvec, "tvec": p.tvec, "camera": 0})
        if progress:
            progress(0.6 + 0.4 * (j + 1) / len(picked), f"retouch: restored frame {j + 1}/{len(picked)}")
    sx, sy = rw / cw, rh / ch
    k = clip.intrinsics
    cam_bin, ids = append_cameras((sparse / "cameras.bin").read_bytes(), [
        {"width": rw, "height": rh, "fx": k["fx"] * sx, "fy": k["fy"] * sy, "cx": k["cx"] * sx, "cy": k["cy"] * sy}])
    (sparse / "cameras.bin").write_bytes(cam_bin)
    # (masked: build_alpha_dataset already renamed the photos to their .png cut-outs)
    images_bin = (sparse / "images.bin").read_bytes()
    (sparse / "images.bin").write_bytes(append_images(images_bin, new, ids))
    stats = {"photos": photos, "restored": len(picked), "clipFrames": n_clip, "restoredSize": [rw, rh]}
    if log:
        log(f"retouch: {photos} photos + {len(picked)} of {n_clip} restored frames")
    return out, stats


def sh_degree_of(ply) -> int:
    """SH degree of a .ply from its f_rest_* count."""
    from .splat_clean import read_ply

    names = read_ply(ply).vertex.dtype.names or ()
    rest = sum(1 for n in names if n.startswith("f_rest_"))
    for d in range(4):
        if rest == 3 * ((d + 1) ** 2 - 1):
            return d
    raise ValueError(f"{ply}: {rest} f_rest coefficients match no SH degree")
