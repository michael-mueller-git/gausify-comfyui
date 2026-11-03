"""Port of web/src/sfm/orbit.test.ts (synthetic orbit with known poses)."""

import math

import numpy as np
import pytest

from gausify.orbit import correspondences, fill_missing, flatten_orbit, refine_pose
from gausify.types import Point3D, Pose, SfmModel


def unit(a):
    a = np.asarray(a, dtype=float)
    return a / np.linalg.norm(a)


def look_at(c):
    """Camera at `c` looking at the origin, up ~ +z (COLMAP: y down)."""
    c = np.asarray(c, dtype=float)
    z = unit(-c)
    x = unit(np.cross(z, [0, 0, -1]))
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return Pose(R=R, t=-(R @ c))


def center_of(p):
    return -(p.R.T @ p.t)


def angle_between(a, b):
    M = a.R @ b.R.T
    return math.acos(min(1, max(-1, (np.trace(M) - 1) / 2)))


def orbit(height_at=lambda i: 1.0):
    n = 96
    at = [i / 24 for i in range(n)]
    truth = []
    for i, s in enumerate(at):
        th = 2 * math.pi * 0.85 * (s / 4) * (0.8 + 0.2 * (s / 4))
        truth.append(look_at([4 * math.cos(th), 4 * math.sin(th), height_at(i)]))
    model = SfmModel(
        f=500, cx=320, cy=240, width=640, height=480,
        poses=[Pose(R=p.R.copy(), t=p.t.copy()) for p in truth],
        points=[
            Point3D(X=np.array([0.1, 0, 0]), rgb=(0, 0, 0), obs=[]),
            Point3D(X=np.array([0, 0.1, 0.2]), rgb=(0, 0, 0), obs=[]),
            Point3D(X=np.array([-0.1, -0.1, -0.1]), rgb=(0, 0, 0), obs=[]),
        ],
        names=[f"{i:05d}.png" for i in range(n)],
    )
    return at, truth, model


def test_fill_missing_middle_and_ends():
    at, truth, model = orbit()
    lost = [0, 1, 20, 21, 22, 23, 50, 94, 95]
    for i in lost:
        model.poses[i] = None
    assert fill_missing(model, None, at)["inferred"] == lost
    for i in lost:
        err = np.linalg.norm(center_of(model.poses[i]) - center_of(truth[i]))
        assert err < 0.005 * 4, f"frame {i}: {err}"
        ang = math.degrees(angle_between(model.poses[i], truth[i]))
        assert ang < 0.3, f"frame {i}: {ang}"
    c = center_of(model.poses[21])
    assert abs(math.hypot(c[0], c[1]) - 4) < 0.01


def test_fill_missing_long_way_round():
    at, truth, model = orbit()
    lost = list(range(10, 76))
    for i in lost:
        model.poses[i] = None
    fill_missing(model, None, at)
    worst = max(np.linalg.norm(center_of(model.poses[i]) - center_of(truth[i])) / 4 for i in lost)
    assert worst < 0.15, worst


def test_flatten_orbit():
    _, _, model = orbit(lambda i: 1 + 0.3 * math.sin(i / 5))
    xy = []
    for p in model.poses:
        out = []
        for pt in model.points:
            c = p.R @ pt.X + p.t
            out.append([500 * c[0] / c[2] + 320, 500 * c[1] / c[2] + 240])
        xy.append(np.array(out))
    for j, pt in enumerate(model.points):
        pt.obs = [(i, j) for i in range(len(model.poses))]
    dev = flatten_orbit(model, xy)
    assert dev is not None and dev > 0.05
    cs = [center_of(p) for p in model.poses]
    n = unit(np.cross(cs[30] - cs[0], cs[60] - cs[0]))
    for c in cs:
        assert abs(np.dot(n, c - cs[0])) < 1e-6
    for p in model.poses:
        c = p.t
        assert math.hypot(c[0] / c[2], c[1] / c[2]) < 1e-9
    assert math.isfinite(model.stats["meanError"])


def _lcg(seed=12345):
    state = [seed]

    def rnd():
        state[0] = (state[0] * 1103515245 + 12345) % 2**31
        return state[0] / 2**31 - 0.5

    return rnd


def _synthetic_corr(pose, n=200):
    rnd = _lcg()
    uv = np.zeros((n, 2))
    X = np.zeros((n, 3))
    for j in range(n):
        x = np.array([rnd() * 2, rnd() * 2, rnd() * 2])
        p = pose.R @ x + pose.t
        uv[j, 0] = 500 * p[0] / p[2] + 320 + rnd()
        uv[j, 1] = 500 * p[1] / p[2] + 240 + rnd()
        if j % 5 == 0:
            uv[j, 0] += rnd() * 400
            uv[j, 1] += rnd() * 400
        X[j] = x
    return uv, X


def test_refine_pose_rough_guess_with_wrong_matches():
    _, truth, model = orbit()
    pose = truth[30]
    c = center_of(pose)
    uv, X = _synthetic_corr(pose)
    c0 = c + np.array([0.08, -0.06, 0.05])
    guess = look_at(c0)
    r = refine_pose(model, guess.R, c0, 1, {"uv": uv, "X": X, "n": 200})
    assert r is not None
    err = np.linalg.norm(center_of(r["pose"]) - c) / 4
    assert err < 0.002
    assert math.degrees(angle_between(r["pose"], pose)) < 0.1
    assert r["inliers"] >= 150
    assert refine_pose(model, guess.R, c0, 1, {"uv": uv[:10], "X": X[:10], "n": 10}) is None


def test_correspondences_and_refined_fill():
    """End to end: a lost frame matched to a registered neighbour gets refined."""
    at, truth, model = orbit()
    rnd = _lcg(7)
    pts = [np.array([rnd() * 2, rnd() * 2, rnd() * 2]) for _ in range(120)]
    model.points = [Point3D(X=x, rgb=(0, 0, 0), obs=[(29, j)]) for j, x in enumerate(pts)]
    xy = [np.zeros((0, 2)) for _ in model.poses]
    for i in (29, 30):
        p = truth[i]
        xy[i] = np.array([[500 * c[0] / c[2] + 320, 500 * c[1] / c[2] + 240] for c in (p.R @ x + p.t for x in pts)])
    model.xy = xy
    model.poses[30] = None
    k = np.arange(120, dtype=np.int32)
    matches = [{"a": 29, "b": 30, "idx": np.stack([k, k], axis=1)}]
    corr = correspondences(model, None, matches)
    assert list(corr) == [30] and corr[30]["n"] == 120
    out = fill_missing(model, matches, at)
    assert out["inferred"] == [30]
    assert out["refined"] and out["refined"][0][0] == 30
    assert np.linalg.norm(center_of(model.poses[30]) - center_of(truth[30])) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__])
