"""COLMAP binary model I/O (cameras.bin / images.bin / points3D.bin).

Writer: port of `colmapFiles` in gausify-orbit web/src/sfm/colmap.ts (same bytes).
Reader: inverse of it (also reads models written by COLMAP itself).
`rewrite_images` / `rename_images`: `rewriteImages` / `renameImages` in
web/src/sfm/colmap.ts, as used by `buildAlphaDataset` in web/src/host/job.ts.

Layout written (little endian), exactly like the reference:
- cameras.bin: u64 1 | u32 camera_id=1 | i32 model=0 (SIMPLE_PINHOLE) |
  u64 width | u64 height | f64 f, cx, cy
- images.bin: u64 #registered | per registered image i (index order):
  u32 i+1 | f64 qw,qx,qy,qz | f64 tx,ty,tz | u32 camera 1 | name\\0 |
  u64 #obs | per observation (in point order): f64 x, f64 y, i64 point_id
  Only the keypoints that observe a 3D point are stored, so the points2D
  indices differ from the feature indices.
- points3D.bin: u64 #points | per point j: u64 j+1 | f64 X,Y,Z | u8 r,g,b |
  f64 error | u64 track length | per obs (u32 image_id, u32 point2D_idx).
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .types import Point3D, Pose, SfmModel

# COLMAP camera models: id -> number of params.
_CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 12}


def quat_from_mat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z), w >= 0, as quatFromMat in web/src/sfm/linalg.ts."""
    R = np.asarray(R, dtype=np.float64).reshape(9)
    tr = R[0] + R[4] + R[8]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        w, x, y, z = 0.25 * s, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s
    elif R[0] > R[4] and R[0] > R[8]:
        s = np.sqrt(1 + R[0] - R[4] - R[8]) * 2
        w, x, y, z = (R[7] - R[5]) / s, 0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s
    elif R[4] > R[8]:
        s = np.sqrt(1 + R[4] - R[0] - R[8]) * 2
        w, x, y, z = (R[2] - R[6]) / s, (R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s
    else:
        s = np.sqrt(1 + R[8] - R[0] - R[4]) * 2
        w, x, y, z = (R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    q = (float(w) / n, float(x) / n, float(y) / n, float(z) / n)
    return tuple(-v for v in q) if q[0] < 0 else q  # type: ignore[return-value]


def mat_from_quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def colmap_files(model: SfmModel, names: list[str] | None = None, xy: list[np.ndarray] | None = None) -> dict[str, bytes]:
    """The three files' bytes, like `colmapFiles(model, names, xy)`.
    `xy[i]`: keypoints (N,2) of image i in pixels (default `model.xy`)."""
    names = model.names if names is None else names
    xy = model.xy if xy is None else xy
    cams = struct.pack("<QIiQQddd", 1, 1, 0, int(model.width), int(model.height), float(model.f), float(model.cx), float(model.cy))

    per_image: dict[int, list[tuple[int, int]]] = {}
    for pi, p in enumerate(model.points):
        for i, k in p.obs:
            per_image.setdefault(int(i), []).append((int(k), pi))
    pt2d: dict[tuple[int, int], int] = {}
    parts: list[bytes] = []
    registered = [(p, i) for i, p in enumerate(model.poses) if p is not None]
    parts.append(struct.pack("<Q", len(registered)))
    for pose, i in registered:
        q = quat_from_mat(pose.R)
        t = np.asarray(pose.t, dtype=np.float64).reshape(3)
        parts.append(struct.pack("<I7dI", i + 1, *q, *t.tolist(), 1))
        parts.append(names[i].encode("utf-8") + b"\0")
        obs = per_image.get(i, [])
        parts.append(struct.pack("<Q", len(obs)))
        if obs:
            kp = np.asarray(xy[i], dtype=np.float32).reshape(-1, 2)
            rec = np.zeros(len(obs), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            feats = np.array([f for f, _ in obs], dtype=np.int64)
            # Float32Array -> f64, as the reference (xy are f32 there)
            rec["x"] = kp[feats, 0].astype(np.float64)
            rec["y"] = kp[feats, 1].astype(np.float64)
            rec["id"] = np.array([pt + 1 for _, pt in obs], dtype=np.int64)
            parts.append(rec.tobytes())
            for idx, (feat, _) in enumerate(obs):
                pt2d[(i, feat)] = idx
    images = b"".join(parts)

    parts = [struct.pack("<Q", len(model.points))]
    for pi, p in enumerate(model.points):
        X = np.asarray(p.X, dtype=np.float64).reshape(3)
        parts.append(struct.pack("<Q3d3BdQ", pi + 1, *X.tolist(), *(int(v) for v in p.rgb), float(p.error), len(p.obs)))
        if p.obs:
            parts.append(
                np.array(
                    [(int(i) + 1, pt2d.get((int(i), int(k)), 0)) for i, k in p.obs], dtype="<u4"
                ).tobytes()
            )
    return {"cameras.bin": cams, "images.bin": images, "points3D.bin": b"".join(parts)}


def write_model(model: SfmModel, sparse_dir, names: list[str] | None = None, xy: list[np.ndarray] | None = None) -> Path:
    """Writes cameras/images/points3D.bin into `sparse_dir` (e.g. root/sparse/0)."""
    d = Path(sparse_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in colmap_files(model, names, xy).items():
        (d / name).write_bytes(data)
    return d


def _read_cameras(data: bytes):
    (n,) = struct.unpack_from("<Q", data, 0)
    o = 8
    cams = {}
    for _ in range(n):
        cid, mid, w, h = struct.unpack_from("<IiQQ", data, o)
        o += 24
        k = _CAMERA_PARAMS.get(mid)
        if k is None:
            raise ValueError(f"unknown COLMAP camera model {mid}")
        params = struct.unpack_from(f"<{k}d", data, o)
        o += 8 * k
        cams[cid] = (mid, int(w), int(h), params)
    return cams


def read_model(sparse_dir, names: list[str] | None = None) -> SfmModel:
    """Reads a binary COLMAP model into an SfmModel.

    Images are matched to frames by name: `names` (all frames in order) gives
    the length of `poses` (None for unregistered); without it, the registered
    names sorted are used. Images whose name isn't in `names` are skipped. `xy[i]` holds the stored points2D and
    `obs` refers to them. SIMPLE_PINHOLE: f; PINHOLE and others: f = fx.
    """
    d = Path(sparse_dir)
    cams = _read_cameras((d / "cameras.bin").read_bytes())
    data = (d / "images.bin").read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    o = 8
    recs = []
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from("<I7dI", data, o)
        o += 64
        end = data.index(b"\0", o)
        name = data[o:end].decode("utf-8")
        o = end + 1
        (m,) = struct.unpack_from("<Q", data, o)
        o += 8
        pts = np.frombuffer(data, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")], count=m, offset=o)
        o += 24 * m
        recs.append((iid, name, (qw, qx, qy, qz), (tx, ty, tz), cid, pts))

    if names is None:
        names = sorted(r[1] for r in recs)
    index_of_name = {nm: i for i, nm in enumerate(names)}
    by_id: dict[int, int] = {}
    poses: list[Pose | None] = [None] * len(names)
    xy: list[np.ndarray] = [np.zeros((0, 2), dtype=np.float64) for _ in names]
    cam_id = None
    for iid, name, q, t, cid, pts in recs:
        i = index_of_name.get(name)
        if i is None:
            continue
        by_id[iid] = i
        poses[i] = Pose(R=mat_from_quat(q), t=np.array(t, dtype=np.float64))
        xy[i] = np.stack([pts["x"], pts["y"]], axis=1).astype(np.float64)
        cam_id = cid if cam_id is None else cam_id
    if cam_id is None:
        cam_id = next(iter(cams))
    mid, w, h, params = cams[cam_id]
    if mid == 0:
        f, cx, cy = params[0], params[1], params[2]
    elif mid in (1, 4, 5, 6, 8, 10):  # fx, fy, cx, cy, ...
        f, cx, cy = params[0], params[2], params[3]
    else:  # f, cx, cy, ...
        f, cx, cy = params[0], params[1], params[2]

    data = (d / "points3D.bin").read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    o = 8
    points: list[Point3D] = []
    for _ in range(n):
        _pid, X0, X1, X2, r, g, b, err, tl = struct.unpack_from("<Q3d3BdQ", data, o)
        o += 51
        tr = np.frombuffer(data, dtype="<u4", count=2 * tl, offset=o).reshape(-1, 2)
        o += 8 * tl
        obs = [(by_id[int(a)], int(k)) for a, k in tr if int(a) in by_id]
        points.append(Point3D(X=np.array([X0, X1, X2]), rgb=(int(r), int(g), int(b)), obs=obs, error=float(err)))

    reg = sum(p is not None for p in poses)
    stats = {"registered": reg, "total": len(names), "points": len(points)}
    return SfmModel(f=float(f), cx=float(cx), cy=float(cy), width=w, height=h, poses=poses, points=points,
                    names=list(names), xy=xy, stats=stats)


def rewrite_images(images_bin: bytes, edit: Callable[[dict], dict | None]) -> bytes:
    """Copy of images.bin where `edit({"name", "q", "t"})` renames, moves or
    drops (None) each image; the rest of each record is unchanged."""
    v = memoryview(images_bin)
    parts: list[bytes] = []
    count = 0
    o = 8
    (n,) = struct.unpack_from("<Q", images_bin, 0)
    for _ in range(n):
        head = o + 4 + 7 * 8 + 4
        end = images_bin.index(b"\0", head)
        f = struct.unpack_from("<7d", images_bin, o + 4)
        rec = edit({"name": images_bin[head:end].decode("utf-8"), "q": list(f[:4]), "t": list(f[4:])})
        body = end + 1
        (m,) = struct.unpack_from("<Q", images_bin, body)
        nxt = body + 8 + m * 24
        if rec is not None:
            top = bytearray(v[o:head])
            struct.pack_into("<7d", top, 4, *(float(x) for x in [*rec["q"], *rec["t"]]))
            parts += [bytes(top), rec["name"].encode("utf-8"), b"\0", bytes(v[body:nxt])]
            count += 1
        o = nxt
    return struct.pack("<Q", count) + b"".join(parts)


def rename_images_bytes(images_bin: bytes, rename: Callable[[str], str]) -> bytes:
    return rewrite_images(images_bin, lambda r: {**r, "name": rename(r["name"])})


def rename_images(sparse_dir, fn: Callable[[str], str], out_dir=None) -> Path:
    """`renameImages` on `sparse_dir/images.bin`, written to `out_dir`
    (default: in place). Returns the written path."""
    src = Path(sparse_dir) / "images.bin"
    data = rename_images_bytes(src.read_bytes(), fn)
    dst = Path(out_dir) / "images.bin" if out_dir is not None else src
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return dst
