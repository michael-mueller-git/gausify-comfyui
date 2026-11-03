"""Pose check: distance transform, XOR score and the 6-DOF alignment on a
ray-traced scene (port of web/src/posecheck/posecheck.test.ts), plus a
job-level run with Brush faked by the same ray tracer."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gausify.posecheck import align as A
from gausify.posecheck import job as J
from gausify.posecheck.align import (
    DEFAULT_ALIGN,
    align_cost,
    align_step,
    apply_step,
    choose_pose,
    exp_so3,
    prepare_target,
    quat_from_mat,
    rotation_from_quat,
)
from gausify.posecheck.image import distance_to, dt1, fill_holes, open_disk, signed_distance
from gausify.posecheck.score import frame_value, opening_radius, orbit_score, segments, xor_score
from gausify.types import OrbitCameras


def rand(seed: int):
    s = seed & 0xFFFFFFFF

    def r():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 2**32

    return r


def test_distance_transform_matches_brute_force():
    w, h, r = 37, 23, rand(3)
    m = np.array([1 if r() < 0.04 else 0 for _ in range(w * h)], dtype=np.uint8).reshape(h, w)
    d = distance_to(m)
    ty, tx = np.nonzero(m)
    for y in range(h):
        for x in range(w):
            best = np.min(np.hypot(tx - x, ty - y))
            assert abs(d[y, x] - best) < 1e-4


def test_dt1_matches_brute_force():
    r = rand(7)
    f = np.array([0.0 if r() < 0.2 else 1e20 for _ in range(50)])
    f[10] = 3.0  # a non-binary sample too
    d = dt1(f)
    q = np.arange(50)
    brute = np.min((q[:, None] - q[None, :]) ** 2 + f[None, :], axis=1)
    assert np.allclose(d, brute)


def test_distance_without_target_is_huge():
    assert np.all(distance_to(np.zeros((4, 5), np.uint8)) > 1e9)


def test_signed_distance():
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 1
    s = signed_distance(m)
    assert [s[10, 4], s[10, 5], s[10, 9], s[10, 1]] == [1, -1, -5, 4]


def test_opening_drops_thin_strands():
    m = np.zeros((30, 30), np.uint8)
    m[2:28, 3] = 1
    m[10:20, 10:20] = 1
    o = open_disk(m, 1)
    assert o[:, 3].sum() == 0
    assert o[:, 10:20].sum() == 100 - 4


def test_fill_holes():
    w, h = 12, 10
    m = np.zeros((h, w), np.uint8)
    m[1:9, 1:11] = 1
    m[4, 4] = m[4, 5] = m[5, 4] = 0
    m[1:6, 8] = 0
    m[0, 8] = 0
    f = fill_holes(m)
    assert [f[4, 4], f[4, 5], f[5, 4]] == [1, 1, 1]
    assert f[3, 8] == 0 and f[0, 0] == 0
    assert f.sum() == m.sum() + 3


def test_xor_score():
    w = h = 64

    def square(x0, y0, s):
        a = np.zeros((h, w), np.float32)
        a[y0 : y0 + s, x0 : x0 + s] = 1
        return a

    mask = (square(16, 16, 32) > 0.5).astype(np.uint8)
    s1 = xor_score(square(17, 16, 32), mask, 1)
    assert [s1["out"], s1["in"], s1["subject"]] == [0, 0, 1024]
    s4 = xor_score(square(20, 16, 32), mask, 1)
    assert 100 < s4["out"] <= 128 and 100 < s4["in"] <= 128
    assert abs(frame_value(s4) - s4["out"] / 1024) < 1e-12
    assert frame_value({"out": 5, "in": 0, "subject": 0}) == 0
    assert orbit_score([0.01, 0.03]) == 0.02
    assert segments([1, 1, 3, 3, 5], 2) == [
        {"start": 0, "end": 1, "value": 1},
        {"start": 2, "end": 3, "value": 3},
        {"start": 4, "end": 4, "value": 5},
    ]
    assert [opening_radius(s) for s in (320, 640, 960, 1280)] == [1, 1, 2, 2]


# --- alignment on a ray-traced scene ------------------------------------------

SPHERES = [(np.array([0, 0.25, 0]), 0.8), (np.array([0.35, -0.75, 0.1]), 0.38)]


def shade(p):
    v = (
        np.sin(9 * p[..., 0] + 2) * np.sin(7 * p[..., 1] - 1)
        + 0.6 * np.sin(17 * p[..., 2] + 5 * p[..., 0])
        + 0.4 * np.sin(23 * p[..., 1] + 11 * p[..., 2])
    )
    return 0.5 + 0.22 * v


def look_at(eye):
    eye = np.asarray(eye, dtype=np.float64)
    z = -eye / np.linalg.norm(eye)
    up = np.array([0, -1, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -(R @ eye)


def raytrace(R, t, K, w, h):
    """Colour (premultiplied, alpha from 2x2 supersampling) and depth, like Brush's renders."""
    c = -(R.T @ t)
    a = np.zeros((h, w))
    col = np.zeros((h, w))
    zs = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    for sx, sy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        dc = np.stack([(xx + sx - K["cx"]) / K["fx"], (yy + sy - K["cy"]) / K["fy"], np.ones_like(xx)], -1)
        d = dc @ R  # R^T dc
        best = np.full((h, w), np.inf)
        for sc, sr in SPHERES:
            o = c - sc
            Aq = (d * d).sum(-1)
            B = 2 * (d @ o)
            C = o @ o - sr * sr
            disc = B * B - 4 * Aq * C
            with np.errstate(invalid="ignore"):
                lam = (-B - np.sqrt(disc)) / (2 * Aq)
            hit = (disc >= 0) & (lam > 0) & (lam < best)
            best = np.where(hit, lam, best)
        hit = np.isfinite(best)
        bb = np.where(hit, best, 0)
        p = c + bb[..., None] * d
        a += 0.25 * hit
        col += np.where(hit, 0.25 * shade(p), 0)
        zs += 0.25 * bb
    rgba = np.stack([col, col, col, a], -1).astype(np.float32)
    depth = np.where(a > 0, zs / np.maximum(a, 1e-12), 0).astype(np.float32)
    return rgba, depth


def frame_rgba8(rgba):
    """Straight-alpha 8-bit frame from a premultiplied render (like Uint8ClampedArray: rounds, clamps)."""
    a = rgba[..., 3:4]
    rgb = rgba[..., :3] / np.maximum(a, 1e-6)
    out = np.concatenate([rgb, a], -1) * 255
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def pose_error(a, b):
    D = a[0] @ b[0].T
    deg = np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1)))
    ca, cb = a[0].T @ a[1], b[0].T @ b[1]
    return deg, float(np.linalg.norm(ca - cb))


W, H = 160, 200
KK = {"fx": 260.0, "fy": 260.0, "cx": W / 2, "cy": H / 2}
TRUTH = look_at([0.4, -0.2, -4.2])
FRAME, _ = raytrace(*TRUTH, KK, W, H)
TARGET = prepare_target(frame_rgba8(FRAME))


def xor_px(rgba):
    s = xor_score(rgba[..., 3], TARGET.mask, 1)
    return s["out"] + s["in"]


def recover(o: A.AlignOptions, start_truth=False):
    dR = exp_so3([0.006, -0.012, 0.008])
    pose = TRUTH if start_truth else (dR @ TRUTH[0], TRUTH[1] + np.array([0.03, -0.02, 0.04]))
    deg0, mv0 = pose_error(pose, TRUTH)
    xor0 = xor_px(raytrace(*pose, KK, W, H)[0])
    for _ in range(3):
        rgba, depth = raytrace(*pose, KK, W, H)
        s = align_step({"rgba": rgba, "depth": depth}, TARGET, KK, o)
        pose = apply_step(pose[0], pose[1], s.dx)
    deg1, mv1 = pose_error(pose, TRUTH)
    return (deg0, mv0, xor0), (deg1, mv1, xor_px(raytrace(*pose, KK, W, H)[0]))


def test_alignment_undamped_recovers():
    (d0, m0, x0), (d1, m1, x1) = recover(A.replace(DEFAULT_ALIGN, damping=0))
    assert d0 > 0.8
    assert d1 < d0 / 5, (d0, d1)
    assert m1 < m0 / 5, (m0, m1)
    assert x0 > 200 and x1 < x0 / 10, (x0, x1)


def test_alignment_damped_aligns_silhouettes():
    (_, _, x0), (_, _, x1) = recover(DEFAULT_ALIGN)
    assert x1 < x0 / 4, (x0, x1)


def test_choose_pose():
    assert choose_pose([1, 0.9, 0.95], 0.02) == 1
    assert choose_pose([1, 0.99, 0.985], 0.02) == 0
    assert choose_pose([1, 1.2, 1.1], 0.02) == 0
    assert choose_pose([1], 0.02) == 0


def test_alignment_cost_lower_at_truth():
    off = (exp_so3([0.006, -0.012, 0.008]) @ TRUTH[0], TRUTH[1])

    def at(p):
        rgba, depth = raytrace(*p, KK, W, H)
        return align_cost({"rgba": rgba, "depth": depth}, TARGET, KK)

    assert at(TRUTH) < at(off) / 2


def test_alignment_right_pose_stays_put():
    _, (d1, m1, _) = recover(DEFAULT_ALIGN, start_truth=True)
    assert d1 < 0.05 and m1 < 0.01, (d1, m1)


def test_alignment_contour_alone():
    (_, _, x0), (_, _, x1) = recover(A.replace(DEFAULT_ALIGN, interior_weight=0, damping=0))
    assert x1 < x0 / 5, (x0, x1)


def test_quaternion_roundtrip():
    R = exp_so3([0.3, -1.2, 2.0])
    assert np.allclose(rotation_from_quat(quat_from_mat(R)), R)


# --- COLMAP binary -------------------------------------------------------------


def cameras_bin(w, h, f, cx, cy):
    return struct.pack("<QIiQQ4d", 1, 1, 1, w, h, f, f, cx, cy)  # PINHOLE


def images_bin(records):
    """records: (id, name, q, t, n_points)."""
    out = struct.pack("<Q", len(records))
    for iid, name, q, t, pts in records:
        out += struct.pack("<I7dI", iid, *q, *t, 1) + name.encode() + b"\0" + struct.pack("<Q", pts)
        out += b"".join(struct.pack("<ddq", 10 + i, 0, -1) for i in range(pts))
    return out


def test_rewrite_images():
    b = images_bin([(1, "a.jpg", (1, 0, 0, 0), (1, 0, 0), 2), (2, "b.jpg", (1, 0, 0, 0), (2, 0, 0), 1)])
    out = J.rewrite_images(b, lambda r: None if r["name"] == "a.jpg" else {"name": "b.png", "q": [0, 1, 0, 0], "t": [1, 2, 3]})
    v = J.read_views(cameras_bin(64, 48, 50, 32, 24), out)
    assert [x.name for x in v] == ["b.png"]
    assert np.allclose(v[0].R, np.diag([1, -1, -1])) and np.allclose(v[0].t, [1, 2, 3])
    assert len(out) == 8 + 64 + len("b.png") + 1 + 8 + 24
    assert struct.unpack_from("<d", out, 8 + 64 + 6 + 8)[0] == 10
    assert J.rewrite_images(b, lambda r: r) == b


# --- job level, Brush faked by the ray tracer ----------------------------------


def make_dataset(root: Path, n=12, bad=(4, 5, 6)):
    """Orbit of n frames around the spheres; frames in `bad` get poses ~1 deg off."""
    w, h = 120, 150
    K = {"fx": 200.0, "fy": 200.0, "cx": w / 2, "cy": h / 2}
    truth, est, names = {}, [], []
    for i in range(n):
        a = 2 * np.pi * i / n
        R, t = look_at([4.2 * np.sin(a), -0.2, -4.2 * np.cos(a)])
        name = f"{i:05d}.png"
        names.append(name)
        truth[name] = (R, t)
        if i in bad:
            R, t = exp_so3([0.006, -0.012, 0.008]) @ R, t + np.array([0.03, -0.02, 0.04])
        est.append((i + 1, name, quat_from_mat(R), tuple(t), 3))
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(cameras_bin(w, h, K["fx"], K["cx"], K["cy"]))
    (sparse / "images.bin").write_bytes(images_bin(est[::-1]))  # unsorted on disk
    (sparse / "points3D.bin").write_bytes(b"points-unchanged")
    cams = OrbitCameras(root=root, names=names, width=w, height=h, masked=True)
    return cams, truth, K


@pytest.fixture
def fake_brush(monkeypatch):
    calls = {"train": [], "render": 0}

    def build_frames(cams, out_dir, max_side, despill, strength, progress):
        out_dir.mkdir(parents=True)
        views = J.read_views((cams.sparse_dir / "cameras.bin").read_bytes(), (cams.sparse_dir / "images.bin").read_bytes())
        v0 = views[0]
        K = {"fx": v0.fx, "fy": v0.fy, "cx": v0.cx, "cy": v0.cy}
        for name, (R, t) in calls["truth"].items():
            rgba, _ = raytrace(R, t, K, v0.width, v0.height)
            Image.fromarray(frame_rgba8(rgba), "RGBA").save(out_dir / name)
        return out_dir

    def train(dataset_dir, opts, out_ply, init_ply, progress, cancel):
        calls["train"].append({"dir": Path(dataset_dir).name, "opts": opts, "init": init_ply})
        assert sorted(p.name for p in (Path(dataset_dir) / "images").iterdir())
        progress(1.0, "done")
        Path(out_ply).write_text("ply")
        return {"psnr": None, "splats": 1}

    def render(ply, cameras):
        calls["render"] += 1
        out = []
        for c in cameras:
            K = {"fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"]}
            rgba, depth = raytrace(np.asarray(c["R"]), np.asarray(c["t"]), K, c["width"], c["height"])
            out.append({"rgba": rgba, "depth": depth})
        return out

    monkeypatch.setattr(J, "_build_frames", build_frames)
    monkeypatch.setattr(J, "_train", train)
    monkeypatch.setattr(J, "_render", render)
    return calls


def test_check_poses_corrects(tmp_path, fake_brush, monkeypatch):
    cams, truth, _ = make_dataset(tmp_path)
    fake_brush["truth"] = truth
    monkeypatch.setenv("GAUSIFY_POSECHECK", json.dumps({"rounds": 2}))
    seen = []
    stats = J.check_poses(cams, "always", progress=lambda f, m: seen.append(f))
    assert set(stats) == {"score", "threshold", "frames", "flagged", "rounds", "after", "afterFrames", "seconds"}
    assert stats["threshold"] == 0.028 and stats["rounds"] == 2 and len(stats["frames"]) == 12
    # The perturbed frames stick out; after correction the orbit agrees better.
    assert max(stats["frames"][4:7]) > 0 and stats["after"] < stats["score"]
    # Screens: 2 check + 2 per round + 2 verify; refine screens warm-start.
    tr = fake_brush["train"]
    assert len(tr) == 8 and [t["dir"] for t in tr[:2]] == ["fold0", "fold1"]
    assert tr[0]["opts"]["total_steps"] == 600 and tr[0]["opts"]["max_resolution"] == 480 and tr[0]["init"] is None
    assert tr[2]["opts"]["total_steps"] == 750 and tr[2]["init"] is not None
    assert tr[6]["init"] is None
    assert tr[0]["opts"]["depth_distortion_weight"] == 10 and tr[0]["opts"]["ssim_weight"] == 0.2
    # Poses moved toward the truth; points kept; work dir removed.
    sp = cams.sparse_dir
    views = {v.name: v for v in J.read_views((sp / "cameras.bin").read_bytes(), (sp / "images.bin").read_bytes())}
    assert (sp / "points3D.bin").read_bytes() == b"points-unchanged"
    for i in (4, 5, 6):
        n = f"{i:05d}.png"
        start = (exp_so3([0.006, -0.012, 0.008]) @ truth[n][0], truth[n][1])
        assert pose_error((views[n].R, views[n].t), truth[n])[0] < pose_error(start, truth[n])[0]
    assert not (tmp_path / ".posecheck").exists()
    assert all(0 <= f <= 1 for f in seen)


def test_check_poses_auto_not_flagged(tmp_path, fake_brush):
    cams, truth, _ = make_dataset(tmp_path, bad=())
    fake_brush["truth"] = truth
    before = (cams.sparse_dir / "images.bin").read_bytes()
    stats = J.check_poses(cams, "auto")
    assert stats["flagged"] is False and stats["rounds"] == 0 and stats["after"] is None
    assert (cams.sparse_dir / "images.bin").read_bytes() == before
    assert len(fake_brush["train"]) == 2


def test_check_poses_skips_small_orbits(tmp_path, fake_brush):
    cams, truth, _ = make_dataset(tmp_path, n=6, bad=())
    assert J.check_poses(cams, "always") is None
