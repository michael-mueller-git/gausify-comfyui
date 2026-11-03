"""Tests for gausify/sfm_akaze (port of web/src/sfm/{features.worker,matcher,geometry,
linalg,reconstruct,ba}.ts) on synthetic orbits with known poses."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from gausify.sfm_akaze import akaze_poses
from gausify.sfm_akaze.ba import bundle_adjust
from gausify.sfm_akaze.features import ImageFeatures, extract
from gausify.sfm_akaze.geometry import (
    ransac_fundamental,
    relative_pose,
    rng,
    sample,
    solve_pnp,
    tri_angle,
    triangulate,
)
from gausify.sfm_akaze.linalg import exp_so3, quat_from_mat, svd3
from gausify.sfm_akaze.matcher import match
from gausify.sfm_akaze.reconstruct import reconstruct

REF = Path(__file__).resolve().parents[1] / ".dev" / "ref"
F, W, H = 500.0, 640, 480
CX, CY = W / 2, H / 2


def look_at(c):
    """Camera at c looking at the origin, up ≈ +z (COLMAP: y down), as orbit.test.ts."""
    c = np.asarray(c, dtype=np.float64)
    z = -c / np.linalg.norm(c)
    x = np.cross(z, [0, 0, -1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ c


def orbit_cams(n=36, radius=4.0, height=1.0, arc=2 * math.pi):
    return [look_at([radius * math.cos(arc * i / n), radius * math.sin(arc * i / n), height]) for i in range(n)]


def sphere_points(n, seed=0, r=1.0):
    g = np.random.default_rng(seed)
    v = g.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * r * g.uniform(0.85, 1.0, size=(n, 1))


def project(R, t, X, f=F):
    c = X @ R.T + t
    return np.stack([f * c[:, 0] / c[:, 2] + CX, f * c[:, 1] / c[:, 2] + CY], 1), c[:, 2]


def visible(R, t, X):
    uv, z = project(R, t, X)
    cam = -R.T @ t
    facing = ((cam - X) * X).sum(1) > 0.15 * np.linalg.norm(cam - X, axis=1) * np.linalg.norm(X, axis=1)
    return (z > 0) & facing & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)


def rot_err_deg(Ra, Rb):
    M = Ra @ Rb.T
    return math.degrees(math.acos(min(1, max(-1, (np.trace(M) - 1) / 2))))


def umeyama(src, dst):
    """Similarity s, R, t with dst ≈ s R src + t."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A**2).sum(1).mean()
    return s, R, md - s * R @ ms


# ---------------------------------------------------------------- linalg / rng


def test_rng_matches_mulberry32_js():
    # values printed by geometry.ts's rng() under node
    r = rng(1)
    assert [r() for _ in range(5)] == [
        0.6270739405881613, 0.002735721180215478, 0.5274470399599522, 0.9810509674716741, 0.9683778982143849,
    ]
    r = rng(99)
    for _ in range(5000):
        r()
    assert r() == 0.06708628055639565
    r = rng(7919 * 3 + 5)
    assert r() == 0.6417257825378329
    s = sample(10, 8, rng(5))
    assert len(set(s)) == 8 and all(0 <= i < 10 for i in s)


def test_linalg_basics():
    g = np.random.default_rng(1)
    w = g.normal(size=3)
    R = exp_so3(w)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    M = g.normal(size=(3, 3))
    if np.linalg.det(M) < 0:  # svd3 builds U's last column as a cross product (det U = +1)
        M[0] = -M[0]
    U, S, V = svd3(M)
    assert np.allclose(U @ np.diag(S) @ V.T, M, atol=1e-9)
    assert np.all(np.diff(S) <= 0)
    q = quat_from_mat(R)
    assert q[0] >= 0 and np.isclose(np.linalg.norm(q), 1)


# ---------------------------------------------------------------- geometry


def two_views(n=400, outliers=0.2, noise=0.3, seed=3):
    g = np.random.default_rng(seed)
    cams = orbit_cams(36)
    (R1, t1), (R2, t2) = cams[0], cams[3]
    X = sphere_points(4000, seed)
    vis = visible(R1, t1, X) & visible(R2, t2, X)
    X = X[vis][:n]
    p1, _ = project(R1, t1, X)
    p2, _ = project(R2, t2, X)
    p1 = p1 + g.normal(scale=noise, size=p1.shape)
    p2 = p2 + g.normal(scale=noise, size=p2.shape)
    bad = g.random(len(X)) < outliers
    p2[bad] = g.uniform([0, 0], [W, H], size=(bad.sum(), 2))
    return (R1, t1), (R2, t2), X, p1, p2, bad


def test_ransac_fundamental_finds_inliers():
    _, _, X, p1, p2, bad = two_views()
    r = ransac_fundamental(p1, p2, 2.0, 7)
    assert r is not None
    inl = r["inliers"]
    assert inl[~bad].mean() > 0.95
    assert inl[bad].mean() < 0.1
    assert r["count"] == inl.sum()
    # deterministic (same seed, same result)
    r2 = ransac_fundamental(p1, p2, 2.0, 7)
    assert np.array_equal(r2["inliers"], inl) and np.allclose(r2["F"], r["F"])
    assert ransac_fundamental(p1[:11], p2[:11]) is None


def test_relative_pose_and_triangulation():
    (R1, t1), (R2, t2), X, p1, p2, bad = two_views(outliers=0.0, noise=0.0)
    r = ransac_fundamental(p1, p2, 2.0, 1)
    c = np.array([CX, CY])
    n1, n2 = (p1 - c) / F, (p2 - c) / F
    idx = np.nonzero(r["inliers"])[0]
    (Rr, tr), good = relative_pose(r["F"], F, CX, CY, n1, n2, idx)
    assert good >= 0.98 * len(idx)
    Rrel = R2 @ R1.T
    trel = t2 - Rrel @ t1
    assert rot_err_deg(Rr, Rrel) < 0.2
    assert np.dot(tr, trel / np.linalg.norm(trel)) > 0.999
    # exact triangulation with the true poses
    Xt = triangulate([(R1, t1), (R2, t2)], np.concatenate([n1[0], n2[0]]))
    assert np.allclose(Xt, X[0], atol=1e-6)
    assert np.isclose(tri_angle(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]) * 1e9), 0, atol=1e-6)
    assert np.isclose(tri_angle(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3)), 90)


@pytest.mark.parametrize("with_init", [False, True])
def test_solve_pnp(with_init):
    g = np.random.default_rng(5)
    R, t = orbit_cams(36)[7]
    X = sphere_points(3000, 2)
    X = X[visible(R, t, X)][:300]
    uv, _ = project(R, t, X)
    uv += g.normal(scale=0.3, size=uv.shape)
    # Without an initial pose, a first DLT sample with 0 inliers ends the TS RANSAC
    # (log(1 - 0) = 0 -> maxIter = -Infinity): faithful, so no outliers in that case.
    bad = g.random(len(X)) < (0.3 if with_init else 0.0)
    uv[bad] = g.uniform([0, 0], [W, H], size=(bad.sum(), 2))
    x = (uv - [CX, CY]) / F
    init = orbit_cams(36)[6] if with_init else None
    r = solve_pnp(X, x, 4.0 / F, init, 8)
    assert r is not None
    (Re, te), inl = r
    assert rot_err_deg(Re, R) < 0.2
    assert np.linalg.norm(-Re.T @ te - (-R.T @ t)) < 0.01 * 4
    assert len(inl) >= 0.9 * (~bad).sum()
    assert solve_pnp(X[:5], x[:5], 0.01) is None


# ---------------------------------------------------------------- BA


def test_bundle_adjust_converges():
    g = np.random.default_rng(11)
    cams = orbit_cams(12, arc=math.pi / 2)
    N = len(cams)
    Xt = sphere_points(800, 4)
    R = np.stack([c[0] for c in cams])
    t = np.stack([c[1] for c in cams])
    oi, op, ouv = [], [], []
    for i in range(N):
        v = np.nonzero(visible(R[i], t[i], Xt))[0]
        uv, _ = project(R[i], t[i], Xt[v])
        oi += [i] * len(v)
        op += v.tolist()
        ouv.append(uv + g.normal(scale=0.2, size=uv.shape))
    oi, op, ouv = np.array(oi), np.array(op), np.concatenate(ouv)
    # perturb all but the first camera, the points and the focal
    Rp, tp = R.copy(), t.copy()
    for i in range(1, N):
        Rp[i] = exp_so3(g.normal(scale=0.01, size=3)) @ R[i]
        tp[i] = t[i] + g.normal(scale=0.02, size=3)
    Xp = Xt + g.normal(scale=0.02, size=Xt.shape)
    # gauge: camera 0 and the scale (through camera 1's translation) stay put, focal varies
    var = list(range(1, N))
    r = bundle_adjust(F * 1.05, CX, CY, Rp, tp, var, Xp, oi, op, ouv, True, 2.0, 50)
    assert r["rmsBefore"] > 5
    assert r["rmsAfter"] < 0.35
    assert abs(r["f"] / F - 1) < 0.02
    # fixed cameras untouched
    assert np.array_equal(Rp[0], R[0])
    # fixed focal stays fixed
    r2 = bundle_adjust(F, CX, CY, R.copy(), t.copy(), [1, 2], Xt.copy(), oi, op, ouv, False, 2.0, 5)
    assert r2["f"] == F and r2["rmsAfter"] <= r2["rmsBefore"] + 1e-12


# ---------------------------------------------------------------- matcher


def _ref_match(da, db, ratio=0.8, max_dist=110):
    """Straight port of the shader + the CPU filter of matcher.ts."""
    pa = np.unpackbits(da, axis=1).astype(np.int64)
    pb = np.unpackbits(db, axis=1).astype(np.int64)
    D = (pa[:, None, :] != pb[None, :, :]).sum(2)

    def scan(D):
        out = []
        for row in D:
            best = second = 0xFFFF
            bi = 0
            for j, d in enumerate(row):
                if d < best:
                    second, best, bi = best, d, j
                elif d < second:
                    second = d
            out.append((bi, best, second))
        return out

    ab, ba = scan(D), scan(D.T)
    ia, ib = [], []
    for i, (j, best, second) in enumerate(ab):
        if best > max_dist or best > ratio * second:
            continue
        if ba[j][0] != i:
            continue
        ia.append(i)
        ib.append(j)
    return np.array(ia, dtype=np.int32), np.array(ib, dtype=np.int32)


def test_matcher_matches_reference_semantics():
    g = np.random.default_rng(0)
    base = g.integers(0, 256, size=(150, 61), dtype=np.uint8)
    da = base.copy()
    db = base[g.permutation(150)].copy()
    # flip a few bits in B, add duplicates (ties) and random descriptors
    flips = g.integers(0, 61 * 8, size=(150, 12))
    for k in range(150):
        for bit in flips[k][: g.integers(0, 12)]:
            db[k, bit // 8] ^= np.uint8(1 << (bit % 8))
    db = np.concatenate([db, db[:5], g.integers(0, 256, size=(40, 61), dtype=np.uint8)])
    da = np.concatenate([da, g.integers(0, 256, size=(20, 61), dtype=np.uint8)])
    descs = [da, db, db[:1]]
    out = match(descs, [(0, 1), (1, 0), (0, 2)], device=torch.device("cpu"))
    for m in out:
        ia, ib = _ref_match(descs[m["a"]], descs[m["b"]])
        assert m["idx"].dtype == np.int32 and m["idx"].shape[1] == 2
        assert np.array_equal(m["idx"][:, 0], ia) and np.array_equal(m["idx"][:, 1], ib)
    assert len(out[0]["idx"]) > 100


# ---------------------------------------------------------------- features


def _texture(h, w, seed=0):
    g = np.random.default_rng(seed)
    img = cv2.resize(g.integers(0, 256, size=(h // 8, w // 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(80):
        c = tuple(int(v) for v in g.integers(0, 256, 3))
        cv2.circle(img, (int(g.integers(0, w)), int(g.integers(0, h))), int(g.integers(3, 30)), c, -1)
    return img


def test_extract_features(tmp_path):
    rgb = _texture(600, 800)
    p = tmp_path / "00000.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    full = extract(p, max_side=1600, max_features=4000)
    assert full.width == 800 and full.height == 600 and full.n > 100
    assert full.xy.dtype == np.float32 and full.desc.shape == (full.n, 61)
    # colors sampled at floor(xy)
    k = 0
    assert tuple(full.rgb[k]) == tuple(rgb[int(full.xy[k, 1]), int(full.xy[k, 0])])
    # resized: coordinates back in full resolution, capped count, strongest first
    small = extract(p, max_side=400, max_features=50)
    assert small.n == 50
    assert small.xy[:, 0].max() < 800 and small.xy[:, 0].max() > 400
    # masked: only where mask > 0
    mask = np.zeros((600, 800), np.uint8)
    mask[:, :400] = 255
    mp = tmp_path / "m.png"
    cv2.imwrite(str(mp), mask)
    m = extract(p, max_side=1600, max_features=4000, mask=mp)
    assert m.n > 0 and m.n < full.n
    assert np.all(np.floor(m.xy[:, 0]) < 400)


# ---------------------------------------------------------------- reconstruction


def synthetic_features(n_cams=24, n_pts=3000, noise=0.3, outlier=0.05, window=3, seed=0, arc=2 * math.pi):
    """ImageFeatures + raw matches from a sphere seen by cameras on an orbit."""
    g = np.random.default_rng(seed)
    cams = orbit_cams(n_cams, arc=arc)
    X = sphere_points(n_pts, seed)
    imgs, kp_of = [], []
    for i, (R, t) in enumerate(cams):
        v = np.nonzero(visible(R, t, X))[0]
        v = v[g.permutation(len(v))]
        uv, _ = project(R, t, X[v])
        uv += g.normal(scale=noise, size=uv.shape)
        imgs.append(ImageFeatures(f"{i:05d}.png", W, H, len(v), uv.astype(np.float32), np.full((len(v), 3), 128, np.uint8)))
        kp_of.append({int(p): k for k, p in enumerate(v)})
    pairs = []
    closed = arc >= 2 * math.pi - 1e-9
    for i in range(n_cams):
        for o in range(1, window + 1):
            j = i + o
            if j >= n_cams and not closed:
                continue
            a, b = min(i, j % n_cams), max(i, j % n_cams)
            if a != b and (a, b) not in pairs:
                pairs.append((a, b))
    matches = []
    for a, b in pairs:
        common = [(kp_of[a][p], kp_of[b][p]) for p in kp_of[a] if p in kp_of[b]]
        idx = np.array(common, dtype=np.int32).reshape(-1, 2)
        nb = int(outlier * len(idx))
        if nb:  # wrong matches
            idx[:nb, 1] = g.integers(0, imgs[b].n, size=nb)
        matches.append({"a": a, "b": b, "idx": idx})
    return imgs, matches, cams


def check_poses(model, cams, rot_tol=0.5, center_tol=0.01):
    reg = [i for i, p in enumerate(model.poses) if p is not None]
    ce = np.stack([model.poses[i].center for i in reg])
    ct = np.stack([-cams[i][0].T @ cams[i][1] for i in reg])
    s, Rs, ts = umeyama(ce, ct)
    radius = np.linalg.norm(ct - ct.mean(0), axis=1).mean()
    err = np.linalg.norm(s * ce @ Rs.T + ts - ct, axis=1).max() / radius
    assert err < center_tol, f"center error {err * 100:.2f}% of the radius"
    worst = max(rot_err_deg(model.poses[i].R @ Rs.T, cams[i][0]) for i in reg)
    assert worst < rot_tol, f"rotation error {worst:.2f} deg"


def test_reconstruct_recovers_orbit():
    imgs, matches, cams = synthetic_features()
    logs = []
    model = reconstruct(imgs, matches, log=logs.append, closed_loop=True)
    assert model.stats["registered"] == len(cams)
    assert model.stats["total"] == len(cams)
    assert abs(model.f / F - 1) < 0.01, model.f
    assert model.stats["meanError"] < 1.0
    assert model.stats["points"] > 1000 and model.stats["observations"] >= 2 * model.stats["points"]
    check_poses(model, cams)
    p = model.points[0]
    assert len(p.obs) >= 2 and all(model.poses[i] is not None for i, _ in p.obs)
    assert all(0 <= k < imgs[i].n for i, k in p.obs)


def test_reconstruct_open_arc():
    imgs, matches, cams = synthetic_features(n_cams=16, arc=math.pi, seed=1)
    model = reconstruct(imgs, matches, closed_loop=False)
    assert model.stats["registered"] == 16
    check_poses(model, cams)


def test_reconstruct_no_parallax_raises():
    imgs, matches, cams = synthetic_features(n_cams=8, arc=math.radians(2), window=2)
    with pytest.raises(RuntimeError):
        reconstruct(imgs, matches)


# ---------------------------------------------------------------- end to end


@pytest.mark.slow
def test_akaze_poses_end_to_end(tmp_path):
    script = REF / "scripts" / "web-bench" / "synth-orbit.py"
    if not script.exists():
        pytest.skip("reference synth-orbit.py missing")
    out = tmp_path / "orbit"
    n = 40
    subprocess.run([sys.executable, str(script), str(out), str(n), "480", "360"], check=True, capture_output=True)
    names = sorted(p.name for p in (out / "images").iterdir())
    from gausify.pairs import orbit_pairs

    pairs = orbit_pairs(len(names), 8, True)
    t0 = time.perf_counter()
    prog = []
    model, matches = akaze_poses(out / "images", names, None, pairs, progress=lambda f, m: prog.append(f))
    dt = time.perf_counter() - t0
    print(f"\nakaze_poses: {n} frames 480x360 in {dt:.1f} s, stats {model.stats}")
    assert model.stats["mapper"] == "akaze"
    assert len(matches) == len(pairs) and all(m["idx"].dtype == np.int32 for m in matches)
    assert len(model.xy) == n and model.xy[0].shape[1] == 2
    assert model.stats["registered"] >= 0.9 * n
    assert prog and prog[-1] <= 1.0
    gt = json.loads((out / "gt.json").read_text())
    cams = []
    for p in gt["poses"]:
        if isinstance(p, dict):
            R = np.array(p.get("R", p.get("rotation")), dtype=np.float64).reshape(3, 3)
            t = np.array(p.get("t", p.get("translation")), dtype=np.float64)
        else:
            M = np.array(p, dtype=np.float64)
            R, t = M[:3, :3], M[:3, 3]
        cams.append((R, t))
    assert abs(model.f / gt["focal"] - 1) < 0.05
    check_poses(model, cams, rot_tol=2.0, center_tol=0.05)
