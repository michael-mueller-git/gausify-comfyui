"""Rigid (similarity) transforms of a trained splat, and centring / standing up
a subject (beyond gausify-orbit: the app has no such step).

The SfM world frame is arbitrary: its origin, scale and "up" are wherever the
pose estimation put them (COLMAP's convention: y points down). These helpers
move the splat into a viewer-friendly frame:

- `apply_transform`: p' = s R p + t for every Gaussian, consistently for
  everything a splat stores: positions, orientations (rot_0..3, w x y z),
  log scales (+ log s), normals and the view-dependent colour (f_rest_*:
  spherical harmonics, rotated band by band so highlights stay where they
  were on the subject). Every other property and the header (comments such as
  Brush's `SplatRenderMode`) are kept.
- `center_transform`: the transform that puts the subject's centre, box
  centre or feet at the origin, optionally upright (the orbit's normal along
  the viewer's up axis) and facing the viewer like the first frame.

SH layout (Brush's export, the INRIA layout): f_rest_* holds, per colour
channel (all of R, then G, then B), the coefficients of bands 1..degree in
order; the basis is Brush's (kernels/sh.rs, Sloan 2013), up to degree 4.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

#: Target frames: (up, front) of the viewer. front: where the first camera ends
#: up (the side the viewer looks from by default).
UP_AXES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "+y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # three.js / ComfyUI Preview3D / glTF: camera on +z
    "-y": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),  # COLMAP / OpenCV, as most splat viewers expect raw files
    "+z": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),  # Blender / Z-up: front view from -y
}
ANCHORS = ["feet", "center", "bbox_center", "none"]
#: Robust extent: percentiles of the opaque splats' positions (ignores floaters).
EXTENT_PERCENTILES = (1.0, 99.0)
#: Splats needed above min_opacity; with fewer, every splat counts.
MIN_OPAQUE = 16
MAX_SH_DEGREE = 4


# ------------------------------------------------------------------ matrices


def rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation by rx about x, then ry about y, then rz about z (degrees):
    R = Rz @ Ry @ Rx."""
    a, b, c = (math.radians(float(v)) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def similarity(scale: float = 1.0, R=None, t=None) -> np.ndarray:
    """4x4 matrix of p -> scale * R p + t."""
    M = np.eye(4)
    M[:3, :3] = float(scale) * (np.eye(3) if R is None else np.asarray(R, np.float64))
    if t is not None:
        M[:3, 3] = np.asarray(t, np.float64)
    return M


def decompose(M) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale, R, t) of a 4x4 similarity; ValueError if it isn't one (shear,
    non-uniform scale or a mirror)."""
    M = np.asarray(M, np.float64)
    if M.shape != (4, 4) or not np.allclose(M[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("transform: expected a 4x4 affine matrix")
    A = M[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    if s <= 1e-12:
        raise ValueError("transform: mirrored or degenerate matrix")
    R = A / s
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        raise ValueError("transform: not a rotation + uniform scale (shear or non-uniform scale)")
    return s, R, M[:3, 3].copy()


def quat_from_rot(R) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(R, np.float64)).as_quat()
    return np.array([w, x, y, z])


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a * b; b may be (N, 4)."""
    aw, ax, ay, az = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)  # fmt: skip


# ------------------------------------------------------------------ spherical harmonics


def sh_bands(d: np.ndarray, degree: int) -> list[np.ndarray]:
    """Brush's SH basis (without the DC band) at unit directions d (N, 3):
    one (N, 2l+1) array per band l = 1..degree, columns in storage order."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    bands = []
    if degree >= 1:
        f0a = 0.4886025119029199
        bands.append(np.stack([-f0a * y, f0a * z, -f0a * x], 1))
    if degree >= 2:
        z2 = z * z
        f0b = -1.0925484305920792 * z
        f1a = 0.5462742152960396
        fc1, fs1 = x * x - y * y, 2 * x * y
        p6 = 0.9461746957575601 * z2 - 0.31539156525252005
        bands.append(np.stack([f1a * fs1, f0b * y, p6, f0b * x, f1a * fc1], 1))
    if degree >= 3:
        f0c = -2.285228997322329 * z2 + 0.4570457994644658
        f1b = 1.445305721320277 * z
        f2a = -0.5900435899266435
        fc2, fs2 = x * fc1 - y * fs1, x * fs1 + y * fc1
        p12 = z * (1.865881662950577 * z2 - 1.119528997770346)
        bands.append(np.stack([f2a * fs2, f1b * fs1, f0c * y, p12, f0c * x, f1b * fc1, f2a * fc2], 1))
    if degree >= 4:
        f0d = z * (-4.683325804901025 * z2 + 2.007139630671868)
        f1c = 3.31161143515146 * z2 - 0.47308734787878
        f2b = -1.770130769779931 * z
        f3a = 0.6258357354491763
        fc3, fs3 = x * fc2 - y * fs2, x * fs2 + y * fc2
        p20 = 1.984313483298443 * z * p12 - 1.006230589874905 * p6
        bands.append(np.stack([f3a * fs3, f2b * fs2, f1c * fs1, f0d * y, p20, f0d * x, f1c * fc1, f2b * fc2,
                               f3a * fc3], 1))
    return bands


def _sample_dirs(n: int = 256) -> np.ndarray:
    """Fibonacci sphere: well-spread directions (plenty to fit every band)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def sh_rotation(R, degree: int) -> list[np.ndarray]:
    """Per band l, the (2l+1)^2 matrix D with c' = D c: the coefficients of a
    splat rotated by R, so colour'(d) = colour(R^T d) for every direction d.

    Each band spans the degree-l harmonics, which a rotation maps onto
    themselves, so B_l(R^T d) = B_l(d) M exactly; M is solved by least squares
    from sample directions (convention-proof: it uses the renderer's basis)."""
    D = _sample_dirs()
    R = np.asarray(R, np.float64)
    out = []
    for b, br in zip(sh_bands(D, degree), sh_bands(D @ R, degree)):  # rows of D @ R: R^T d
        M, *_ = np.linalg.lstsq(b, br, rcond=None)
        out.append(M)
    return out


def sh_degree_of(names: Sequence[str]) -> int:
    n = sum(1 for k in names if k.startswith("f_rest_"))
    if n % 3:
        raise ValueError(f"splat: {n} f_rest_* properties (not a multiple of 3)")
    per = n // 3
    deg = int(round(math.sqrt(per + 1))) - 1
    if (deg + 1) ** 2 - 1 != per or deg > MAX_SH_DEGREE:
        raise ValueError(f"splat: {per} SH coefficients per channel (not a degree 0-{MAX_SH_DEGREE} layout)")
    return deg


# ------------------------------------------------------------------ the splat


def apply_transform(vertex: np.ndarray, M) -> np.ndarray:
    """A copy of the splat (structured array, Brush/INRIA properties) moved by
    the similarity M (4x4)."""
    s, R, t = decompose(M)
    names = vertex.dtype.names or ()
    if not {"x", "y", "z"} <= set(names):
        raise ValueError("splat: no x/y/z properties")
    v = vertex.copy()

    def put(keys, arr):
        for k, col in zip(keys, np.asarray(arr).T):
            v[k] = col.astype(v.dtype[k])

    p = np.stack([vertex[k].astype(np.float64) for k in "xyz"], 1)
    put("xyz", p @ (s * R).T + t)
    if {"nx", "ny", "nz"} <= set(names):
        n = np.stack([vertex[k].astype(np.float64) for k in ("nx", "ny", "nz")], 1)
        put(("nx", "ny", "nz"), n @ R.T)
    rot = [f"rot_{i}" for i in range(4)]
    if set(rot) <= set(names):
        q = np.stack([vertex[k].astype(np.float64) for k in rot], 1)
        put(rot, quat_mul(quat_from_rot(R), q))  # same norm as before: Brush normalises when it loads
    if s != 1.0:
        for k in ("scale_0", "scale_1", "scale_2"):
            if k in names:
                v[k] = (vertex[k].astype(np.float64) + math.log(s)).astype(v.dtype[k])
    deg = sh_degree_of(names)
    if deg and not np.allclose(R, np.eye(3), atol=1e-12):
        per = (deg + 1) ** 2 - 1
        Ds = sh_rotation(R, deg)
        for ch in range(3):
            keys = [f"f_rest_{ch * per + k}" for k in range(per)]
            c = np.stack([vertex[k].astype(np.float64) for k in keys], 1)
            out, start = np.empty_like(c), 0
            for D in Ds:
                m = D.shape[0]
                out[:, start:start + m] = c[:, start:start + m] @ D.T
                start += m
            put(keys, out)
    return v


def transform_header(header: Sequence[str], M) -> list[str]:
    """The header with Brush's "comment Vertical axis: x y z" turned with the
    splat; other lines kept.

    Brush's viewer turns the model by rotation_arc(-Y, v) (its screen up is
    -Y), so v is the model's up rotated 180 degrees about -Y: up = F v with
    F = diag(-1, 1, -1) (F is its own inverse). The new v is F R F v."""
    _, R, _ = decompose(M)
    F = np.diag([-1.0, 1.0, -1.0])
    named = {"x": (1.0, 0.0, 0.0), "y": (0.0, -1.0, 0.0), "z": (0.0, 0.0, -1.0)}  # as Brush reads them
    out = []
    for line in header:
        parts = line.split(None, 1)
        rest = parts[1] if len(parts) > 1 else ""
        if parts and parts[0] == "comment" and rest.lower().startswith("vertical axis:"):
            val = rest.split(":", 1)[1].strip().lower()
            try:
                a = np.array(named[val] if val in named
                             else [float(x) for x in val.replace(",", " ").strip("[]").split()], np.float64)
                if a.shape == (3,) and np.linalg.norm(a) > 1e-12:
                    a = F @ R @ F @ a
                    line = "comment Vertical axis: " + " ".join(f"{x:.8g}" for x in a / np.linalg.norm(a))
            except ValueError:
                pass
        out.append(line)
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def opaque_positions(vertex: np.ndarray, min_opacity: float) -> np.ndarray:
    """Positions (N, 3) of the splats at least min_opacity opaque (all of
    them if fewer than MIN_OPAQUE are)."""
    p = np.stack([vertex[k].astype(np.float64) for k in "xyz"], 1)
    if "opacity" in (vertex.dtype.names or ()) and min_opacity > 0:
        keep = _sigmoid(vertex["opacity"]) >= min_opacity
        if keep.sum() >= MIN_OPAQUE:
            p = p[keep]
    if not len(p):
        raise ValueError("splat: no splats")
    return p


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def up_from_cameras(views: Sequence) -> np.ndarray:
    """World up from the cameras: the orbit plane's normal (the cameras circle
    the subject around it), on the side of the frames' "up" (-y of the image);
    the frames' mean up if the cameras don't form an arc."""
    from .orbit import _orbit_normal
    from .splat_clean import camera_centres

    hint = _unit(-np.mean([np.asarray(v.R)[1] for v in views], axis=0))
    n = _orbit_normal([c for c in camera_centres(views)]) if len(views) >= 3 else None
    if n is None:
        return hint
    n = _unit(n)
    return n if np.dot(n, hint) >= 0 else -n


def up_from_shape(points: np.ndarray) -> np.ndarray:
    """World up without cameras: the splat's longest axis (a standing person's
    height), on the side of -y (COLMAP: image y points down, and the first
    camera usually stands upright)."""
    c = points - np.median(points, axis=0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    a = vt[0]
    return a if a[1] <= 0 else -a


def _horizontal(v, up) -> np.ndarray | None:
    h = np.asarray(v, np.float64) - np.dot(v, up) * up
    n = np.linalg.norm(h)
    return None if n < 1e-9 else h / n


def align_rotation(up, front, up_axis: str) -> np.ndarray:
    """Rotation taking world `up` to the target up axis and the horizontal
    `front` (unit, perpendicular to up) to the target front."""
    U, F = (np.asarray(a, np.float64) for a in UP_AXES[up_axis])
    src = np.stack([up, front, np.cross(up, front)], 1)
    dst = np.stack([U, F, np.cross(U, F)], 1)
    return dst @ src.T


def any_front(up) -> np.ndarray:
    """A horizontal direction (the world axis least aligned with up)."""
    axis = np.eye(3)[int(np.argmin(np.abs(up)))]
    return _horizontal(axis, up)  # type: ignore[return-value]


def center_transform(vertex: np.ndarray, *, anchor: str = "feet", upright: bool = True, up_axis: str = "+y",
                     face_first_camera: bool = True, min_opacity: float = 0.5, height: float = 0.0,
                     views: Sequence | None = None) -> tuple[np.ndarray, dict]:
    """(4x4 similarity, info) that puts the subject at the origin.

    anchor: "feet" (centred horizontally, lowest point on the floor),
    "center" (median of the opaque splats), "bbox_center" (centre of their
    robust box) or "none". upright: rotate so world up becomes `up_axis` (and,
    with cameras and face_first_camera, the first camera's side the viewer's
    front). height > 0 scales the subject to that height (robust extent
    along up). Without upright, "feet" and the height use the target up axis
    in the splat's own frame."""
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}")
    if up_axis not in UP_AXES:
        raise ValueError(f"up_axis must be one of {list(UP_AXES)}")
    P = opaque_positions(vertex, float(min_opacity))
    info: dict = {"splats": int(len(vertex)), "opaque": int(len(P))}
    U = np.asarray(UP_AXES[up_axis][0])
    R = np.eye(3)
    if upright:
        if views:
            up, info["up"] = up_from_cameras(views), "cameras"
        else:
            up, info["up"] = up_from_shape(P), "shape"
        front = None
        if views and face_first_camera:
            from .splat_clean import camera_centres

            front = _horizontal(camera_centres(views[:1])[0] - np.median(P, axis=0), up)
        R = align_rotation(up, front if front is not None else any_front(up), up_axis)
        info["upWorld"] = [round(float(x), 6) for x in up]
    Q = P @ R.T  # the subject in the target orientation
    lo, hi = np.percentile(Q, EXTENT_PERCENTILES, axis=0)
    extent = float(np.dot(hi - lo, np.abs(U)))  # along up
    s = float(height) / extent if height and height > 0 and extent > 1e-12 else 1.0
    if anchor == "none":
        c = np.zeros(3)
    elif anchor == "center":
        c = np.median(Q, axis=0)
    else:
        c = (lo + hi) / 2
        if anchor == "feet":  # on the floor: the lowest point along up
            bottom = lo if U.sum() > 0 else hi
            c = c - np.dot(c, np.abs(U)) * np.abs(U) + np.dot(bottom, np.abs(U)) * np.abs(U)
    M = similarity(s, R, -s * c)
    info.update(anchor=anchor, upright=bool(upright), upAxis=up_axis, scale=round(s, 6),
                height=round(extent * s, 6), offset=[round(float(x), 6) for x in -s * c])
    return M, info


def transform_json(M) -> dict:
    s, R, t = decompose(M)
    return {"matrix": [[float(x) for x in row] for row in np.asarray(M)], "scale": s,
            "rotationDeg": round(math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))), 4),
            "translation": [float(x) for x in t]}


def parse_transform(text: str | None) -> np.ndarray | None:
    """4x4 matrix of a `transform` output (JSON with "matrix", or a bare
    4x4 list); None for an empty string."""
    import json

    if not text or not str(text).strip():
        return None
    obj = json.loads(text)
    M = np.asarray(obj["matrix"] if isinstance(obj, dict) else obj, np.float64)
    decompose(M)  # validates
    return M


def write_transformed(src, root, suffix: str, M) -> tuple[str, int]:
    """Writes `<src stem>_<suffix>.ply` next to `src` with the transform `M` applied.
    Returns the path relative to `root` and the splat count."""
    from pathlib import Path

    from .splat_clean import Ply, read_ply, write_ply

    src, root = Path(src), Path(root)
    ply = read_ply(src)
    out = src.with_name(f"{src.stem}_{suffix}.ply")
    write_ply(out, Ply(header=transform_header(ply.header, M), vertex=ply.vertex), apply_transform(ply.vertex, M))
    return out.relative_to(root).as_posix(), len(ply.vertex)
