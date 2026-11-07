"""Clean-up of a trained splat (beyond gausify-orbit: the app has no such step).

Removes Gaussians that cannot belong to the subject, judged from the
training cameras of the dataset (COLMAP world frame, which is also the frame
of Brush's exported .ply: Brush applies no transform to COLMAP poses):

- hull carving (masked datasets): a splat whose centre projects outside the
  (dilated) subject mask in more than `hull_tolerance` of the views that see
  it lies outside the visual hull: floaters, ghost silhouettes, background;
- unseen: in front of no camera's image at all;
- distance: farther than `max_distance` x orbit radius from the orbit centre
  (the point nearest to every camera's optical axis);
- opacity: sigmoid(opacity) below `min_opacity`;
- size: largest axis (exp(scale)) above `max_size` x orbit radius.

The .ply is read and written generically (binary little-endian, one `vertex`
element, every property and comment kept), so it works for any SH degree.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}  # fmt: skip

#: Views used for carving at most (evenly spaced): plenty for a hull, bounded cost.
MAX_VIEWS = 72
#: Centres closer than this (camera z, in units of the orbit radius) don't count as seen.
NEAR = 1e-3


# ------------------------------------------------------------------ PLY


@dataclass
class Ply:
    header: list[str]  # header lines up to (without) end_header, element vertex count as read
    vertex: np.ndarray  # structured array, one record per splat


def read_ply(path) -> Ply:
    data = Path(path).read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError(f"{path}: not a PLY file")
    body = data.index(b"\n", end) + 1
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in [h.strip() for h in header]:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    count, fields, element = None, [], None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            element = parts[1]
            if element != "vertex":
                raise ValueError(f"{path}: unsupported PLY element {element!r} (expected only 'vertex')")
            count = int(parts[2])
        elif parts[0] == "property" and element == "vertex":
            if parts[1] == "list" or parts[1] not in _PLY_TYPES:
                raise ValueError(f"{path}: unsupported PLY property {line!r}")
            fields.append((parts[2], _PLY_TYPES[parts[1]]))
    if count is None or not fields:
        raise ValueError(f"{path}: no vertex element")
    dtype = np.dtype(fields)
    vertex = np.frombuffer(data, dtype=dtype, count=count, offset=body).copy()
    return Ply(header=header, vertex=vertex)


def write_ply(path, ply: Ply, vertex: np.ndarray | None = None) -> None:
    v = ply.vertex if vertex is None else vertex
    lines = [f"element vertex {len(v)}" if h.split()[:2] == ["element", "vertex"] else h for h in ply.header]
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(("\n".join(lines) + "\nend_header\n").encode("ascii"))
        f.write(np.ascontiguousarray(v).tobytes())
    tmp.replace(path)


# ------------------------------------------------------------------ geometry


def camera_centres(views: Sequence) -> np.ndarray:
    """World positions of the cameras (C = -R^T t)."""
    return np.array([-np.asarray(v.R).T @ np.asarray(v.t) for v in views], dtype=np.float64)


def orbit_centre(views: Sequence) -> tuple[np.ndarray, float]:
    """Least-squares point nearest to every optical axis, and the median
    camera distance to it (the orbit radius)."""
    C = camera_centres(views)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for v, c in zip(views, C):
        d = np.asarray(v.R)[2]  # viewing direction in world (third row of R)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    try:
        centre = np.linalg.solve(A, b) if np.linalg.cond(A) < 1e8 else C.mean(axis=0)
    except np.linalg.LinAlgError:
        centre = C.mean(axis=0)
    radius = float(np.median(np.linalg.norm(C - centre, axis=1)))
    return centre, max(radius, 1e-9)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, k)


def hull_counts(
    xyz: np.ndarray,
    views: Sequence,
    masks: Callable[[object], np.ndarray | None] | None,
    margin: float,
    near: float,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per splat: (number of views whose image contains its centre, of those
    the number where it falls outside the subject mask dilated by `margin`
    (fraction of the image's long side)). Without masks the second is 0."""
    n = len(xyz)
    seen = np.zeros(n, dtype=np.int32)
    outside = np.zeros(n, dtype=np.int32)
    P = xyz.astype(np.float64)
    for k, v in enumerate(views):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        Xc = P @ np.asarray(v.R, dtype=np.float64).T + np.asarray(v.t, dtype=np.float64)
        z = Xc[:, 2]
        front = z > near
        zs = np.where(front, z, 1.0)
        u = v.fx * Xc[:, 0] / zs + v.cx
        w = v.fy * Xc[:, 1] / zs + v.cy
        inside_img = front & (u >= 0) & (u < v.width) & (w >= 0) & (w < v.height)
        seen += inside_img
        m = masks(v) if masks is not None else None
        if m is not None:
            mh, mw = m.shape
            sx, sy = mw / v.width, mh / v.height
            sub = (m >= 128).astype(np.uint8) * 255
            sub = _dilate(sub, int(round(margin * max(mw, mh))))
            idx = np.nonzero(inside_img)[0]
            px = np.clip((u[idx] * sx).astype(np.int64), 0, mw - 1)
            py = np.clip((w[idx] * sy).astype(np.int64), 0, mh - 1)
            outside[idx[sub[py, px] == 0]] += 1
        if progress is not None:
            progress((k + 1) / len(views), f"clean: view {k + 1}/{len(views)}")
    return seen, outside


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -60, 60)))


def spaced(items: Sequence, n: int) -> list:
    if len(items) <= n:
        return list(items)
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, n).round()]


# ------------------------------------------------------------------ clean


def clean_splat(
    ply_in,
    ply_out,
    views: Sequence,
    masks: Callable[[object], np.ndarray | None] | None = None,
    hull_carve: bool = True,
    hull_tolerance: float = 0.25,
    mask_margin: float = 0.01,
    drop_unseen: bool = True,
    max_distance: float = 0.0,
    min_opacity: float = 0.0,
    max_size: float = 0.0,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict:
    """Writes the cleaned splat to `ply_out`; returns stats: input/kept counts
    and the splats removed per reason (a splat counts for the first reason
    that removes it, in the order hull, unseen, distance, opacity, size).
    `views`: posecheck.job.View-like (R, t world->camera, fx, fy, cx, cy,
    width, height). `masks(view)`: its subject mask (uint8, 255 = subject) or
    None. 0 disables `max_distance`, `min_opacity` and `max_size`."""
    if not views:
        raise ValueError("no registered cameras to judge the splat with")
    ply = read_ply(ply_in)
    v = ply.vertex
    names = v.dtype.names or ()
    for f in ("x", "y", "z"):
        if f not in names:
            raise ValueError(f"{ply_in}: no '{f}' property")
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    n = len(xyz)
    centre, radius = orbit_centre(views)
    keep = np.ones(n, dtype=bool)
    removed: dict[str, int] = {}

    def drop(reason: str, bad: np.ndarray) -> None:
        hit = bad & keep
        removed[reason] = int(hit.sum())
        keep[hit] = False

    use_masks = masks if hull_carve else None
    need_views = use_masks is not None or drop_unseen
    if need_views and n:
        seen, outside = hull_counts(xyz, spaced(views, MAX_VIEWS), use_masks, float(mask_margin),
                                    NEAR * radius, cancel, progress)
        if use_masks is not None:
            frac = np.divide(outside, seen, out=np.zeros(n), where=seen > 0)
            drop("hull", (seen > 0) & (frac > float(hull_tolerance)))
        if drop_unseen:
            drop("unseen", seen == 0)
    if max_distance > 0:
        drop("distance", np.linalg.norm(xyz - centre, axis=1) > max_distance * radius)
    if min_opacity > 0 and "opacity" in names:
        drop("opacity", _sigmoid(v["opacity"]) < min_opacity)
    scale_f = [f for f in ("scale_0", "scale_1", "scale_2") if f in names]
    if max_size > 0 and scale_f:
        big = np.exp(np.max(np.stack([v[f].astype(np.float64) for f in scale_f], axis=1), axis=1))
        drop("size", big > max_size * radius)
    Path(ply_out).parent.mkdir(parents=True, exist_ok=True)
    write_ply(ply_out, ply, v[keep])
    return {
        "input": n,
        "kept": int(keep.sum()),
        "removed": removed,
        "orbitCentre": [float(c) for c in centre],
        "orbitRadius": radius,
        "views": len(views),
    }


# ------------------------------------------------------------------ check


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return math.inf if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def render_psnr(
    plys: Sequence,
    views: Sequence,
    target: Callable[[object, int, int], np.ndarray],
    max_side: int = 800,
    render: Callable | None = None,
) -> list[float]:
    """Mean PSNR of each .ply rendered from `views` (downscaled to fit
    `max_side`) against `target(view, w, h)` -> (h, w, 3) float in [0, 1],
    premultiplied over black like Brush's render."""
    if render is None:
        from .brush import render
    cams, sizes = [], []
    for v in views:
        s = min(1.0, max_side / max(v.width, v.height))
        w, h = max(1, round(v.width * s)), max(1, round(v.height * s))
        sizes.append((w, h))
        cams.append({"R": v.R, "t": v.t, "fx": v.fx * w / v.width, "fy": v.fy * h / v.height,
                     "cx": v.cx * w / v.width, "cy": v.cy * h / v.height, "width": w, "height": h})
    targets = [target(v, w, h) for v, (w, h) in zip(views, sizes)]
    out = []
    for ply in plys:
        rs = render(ply, cams)
        vals = [psnr(np.clip(r["rgba"][..., :3], 0, 1), t) for r, t in zip(rs, targets)]
        finite = [x for x in vals if math.isfinite(x)]
        out.append(float(np.mean(finite)) if finite else math.inf)
    return out
