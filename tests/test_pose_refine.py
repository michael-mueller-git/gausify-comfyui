"""Pose refinement against a trained splat: geometry, and the dataset it writes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gausify import pose_refine as pr

K = np.array([[500.0, 0, 320.5], [0, 500.0, 240.5], [0, 0, 1]])


def _rot(axis, deg):
    import cv2

    r, _ = cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * math.radians(deg))
    return r


def _scene(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([-1, -1, -0.5], [1, 1, 0.5], (n, 3))  # around the origin, camera 5 away


def _project(X, R, t):
    pc = X @ R.T + t
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1), pc[:, 2]


def _setup(err_deg=0.8, err_t=(0.03, -0.02, 0.0), noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = _scene(seed=seed)
    R_true, t_true = _rot([0, 1, 0.2], 10), np.array([0.0, 0.0, 5.0])
    R_old = _rot([1, 0.3, 0], err_deg) @ R_true
    t_old = t_true + np.array(err_t)
    uv_r, z_r = _project(X, R_old, t_old)  # the render: the splat at the (wrong) current pose
    uv_p, _ = _project(X, R_true, t_true)  # the photo: where the camera really was
    uv_p = uv_p + rng.normal(0, noise, uv_p.shape)
    return R_true, t_true, R_old, t_old, uv_r, z_r, uv_p


def test_recovers_perturbed_pose():
    R_true, t_true, R_old, t_old, uv_r, z_r, uv_p = _setup()
    r = pr.pose_from_matches(K, R_old, t_old, uv_r, z_r, uv_p, radius=5.0)
    assert r.accepted and r.inliers > 250 and r.after_px < 0.5 < r.before_px
    assert math.degrees(np.arccos(np.clip((np.trace(r.R @ R_true.T) - 1) / 2, -1, 1))) < 0.05
    assert np.linalg.norm(-r.R.T @ r.t - (-R_true.T @ t_true)) < 0.01
    assert 0.5 < r.rotation_deg < 1.5


def test_outliers_and_unusable_depth():
    R_true, t_true, R_old, t_old, uv_r, z_r, uv_p = _setup(seed=1)
    rng = np.random.default_rng(2)
    bad = rng.choice(len(uv_p), 90, replace=False)
    uv_p[bad] = rng.uniform(0, 640, (90, 2))  # wrong matches
    z_r[:40] = np.nan  # depth edges
    r = pr.pose_from_matches(K, R_old, t_old, uv_r, z_r, uv_p, radius=5.0)
    assert r.accepted and r.matches == 260
    assert math.degrees(np.arccos(np.clip((np.trace(r.R @ R_true.T) - 1) / 2, -1, 1))) < 0.1


def test_rejections():
    R_true, t_true, R_old, t_old, uv_r, z_r, uv_p = _setup(err_deg=5.0)
    r = pr.pose_from_matches(K, R_old, t_old, uv_r, z_r, uv_p, radius=5.0)
    assert not r.accepted and "too large" in r.reason
    assert (r.R == R_old).all() and (r.t == t_old).all()
    # a correct pose: nothing to gain, kept
    R_true, t_true, _, _, _, _, uv_p = _setup()
    uv_r, z_r = _project(_scene(), R_true, t_true)
    r = pr.pose_from_matches(K, R_true, t_true, uv_r, z_r, uv_p, radius=5.0)
    assert not r.accepted and "no gain" in r.reason
    # too few usable matches
    z_r[:] = np.nan
    r = pr.pose_from_matches(K, R_true, t_true, uv_r, z_r, uv_p, radius=5.0)
    assert not r.accepted and "too few" in r.reason


def test_sample_depth_edges():
    depth = np.full((20, 20), 4.0, np.float32)
    depth[:, 10:] = 6.0  # a depth edge between x 9 and 10
    alpha = np.ones((20, 20), np.float32)
    alpha[15:] = 0.5  # not opaque
    uv = np.array([[3.5, 3.5], [9.5, 5.5], [10.5, 5.5], [15.5, 5.5], [3.5, 17.5]])
    z = pr.sample_depth(depth, alpha, uv)
    assert z[0] == pytest.approx(4.0) and z[3] == pytest.approx(6.0)
    assert np.isnan(z[1]) and np.isnan(z[2]) and np.isnan(z[4])


def test_unproject_roundtrip():
    X = _scene(50)
    R, t = _rot([0.2, 1, 0], 25), np.array([0.1, -0.2, 5.0])
    uv, z = _project(X, R, t)
    np.testing.assert_allclose(pr.unproject(uv, z, K, R, t), X, atol=1e-9)
    assert pr.reprojection(X, uv, K, R, t).max() < 1e-9
