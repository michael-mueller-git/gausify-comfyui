"""Splat transforms: SH rotation, Gaussian geometry, centring and upright."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gausify import splat_transform as st
from gausify.splat_clean import read_ply, write_ply


def _rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _quat_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _splat(n=50, degree=3, seed=0, extra=("nx", "ny", "nz")):
    rng = np.random.default_rng(seed)
    per = (degree + 1) ** 2 - 1
    names = ["x", "y", "z", *extra, "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(3 * per)],
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(n, np.dtype([(k, "<f4") for k in names]))
    for k in names:
        v[k] = rng.normal(size=n)
    return v


def _colour(v, i, d, degree):
    """Brush's SH colour of splat i (all channels) in unit direction d (no DC)."""
    per = (degree + 1) ** 2 - 1
    basis = np.concatenate(st.sh_bands(d[None], degree), 1)[0]
    return np.array([sum(basis[k] * float(v[f"f_rest_{ch * per + k}"][i]) for k in range(per)) for ch in range(3)])


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_sh_rotation_keeps_colour_on_the_subject(degree):
    rng = np.random.default_rng(degree)
    v = _splat(degree=degree, extra=())
    R = _rand_rot(rng)
    out = st.apply_transform(v, st.similarity(1.0, R))
    for _ in range(5):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for i in range(5):  # the rotated splat, seen along R d, shows what the original showed along d
            np.testing.assert_allclose(_colour(out, i, R @ d, degree), _colour(v, i, d, degree), atol=2e-5)


def test_sh_rotation_matrices_are_orthogonal():
    R = st.rotation_xyz(30, -50, 110)
    for D in st.sh_rotation(R, 4):
        np.testing.assert_allclose(D @ D.T, np.eye(len(D)), atol=1e-9)


def test_geometry_follows_the_similarity():
    v = _splat(degree=2)
    R, s, t = st.rotation_xyz(20, 75, -140), 2.5, np.array([1.0, -2.0, 0.5])
    out = st.apply_transform(v, st.similarity(s, R, t))
    p = np.stack([v[k] for k in "xyz"], 1).astype(np.float64)
    np.testing.assert_allclose(np.stack([out[k] for k in "xyz"], 1), p @ (s * R).T + t, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.stack([out[k] for k in ("nx", "ny", "nz")], 1),
                               np.stack([v[k] for k in ("nx", "ny", "nz")], 1) @ R.T, atol=1e-5)
    for i in range(10):  # covariance: s^2 R Sigma R^T
        q0 = np.array([v[f"rot_{k}"][i] for k in range(4)], np.float64)
        q1 = np.array([out[f"rot_{k}"][i] for k in range(4)], np.float64)
        S0 = np.diag(np.exp([v[f"scale_{k}"][i] for k in range(3)]).astype(np.float64))
        S1 = np.diag(np.exp([out[f"scale_{k}"][i] for k in range(3)]).astype(np.float64))
        C0 = _quat_mat(q0) @ S0 @ S0 @ _quat_mat(q0).T
        C1 = _quat_mat(q1) @ S1 @ S1 @ _quat_mat(q1).T
        np.testing.assert_allclose(C1, s * s * R @ C0 @ R.T, rtol=1e-4, atol=1e-5)
        assert np.linalg.norm(q1) == pytest.approx(np.linalg.norm(q0), rel=1e-5)
    for k in ("opacity", "f_dc_0", "f_dc_2"):  # untouched
        assert (out[k] == v[k]).all()


def test_offset_only_and_roundtrip(tmp_path):
    v = _splat(degree=3)
    out = st.apply_transform(v, st.similarity(t=(0.5, 0, -1)))
    for k in v.dtype.names:
        if k not in "xyz":
            assert (out[k] == v[k]).all(), k  # no rotation: SH, quats, scales bit for bit
    M = st.similarity(1.7, st.rotation_xyz(10, 200, -30), (3, 2, 1))
    back = st.apply_transform(st.apply_transform(v, M), np.linalg.inv(M))
    rot = [f"rot_{i}" for i in range(4)]
    for k in v.dtype.names:
        if k not in rot:
            np.testing.assert_allclose(back[k], v[k], atol=2e-5, err_msg=k)
    q0, q1 = (np.stack([a[k] for k in rot], 1) for a in (v, back))
    np.testing.assert_allclose(q1 * np.sign((q0 * q1).sum(1))[:, None], q0, atol=2e-5)  # q and -q: same rotation
    # through a file: header and comments kept
    from gausify.splat_clean import Ply

    names = v.dtype.names
    header = ["ply", "format binary_little_endian 1.0", "comment SplatRenderMode: mip", f"element vertex {len(v)}",
              *[f"property float {k}" for k in names]]
    write_ply(tmp_path / "a.ply", Ply(header=header, vertex=v), out)
    r = read_ply(tmp_path / "a.ply")
    assert "comment SplatRenderMode: mip" in r.header and (r.vertex == out).all()


def test_quat_from_rot_half_turns():
    for R in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1]), st.rotation_xyz(0, 180, 0),
              st.rotation_xyz(37, -120, 179.9)):
        np.testing.assert_allclose(_quat_mat(st.quat_from_rot(R)), R, atol=1e-9)


def test_decompose_rejects_non_similarities():
    with pytest.raises(ValueError, match="shear|non-uniform"):
        st.decompose(np.diag([1.0, 2.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="mirror"):
        st.decompose(np.diag([1.0, 1.0, -1.0, 1.0]))
    s, R, t = st.decompose(st.similarity(2.0, st.rotation_xyz(5, 6, 7), (1, 2, 3)))
    assert s == pytest.approx(2.0) and t.tolist() == [1, 2, 3]
    assert st.parse_transform("") is None
    M = st.similarity(3.0, None, (1, 0, 0))
    import json

    np.testing.assert_allclose(st.parse_transform(json.dumps(st.transform_json(M))), M)


def test_sh_degree_of():
    assert st.sh_degree_of(["x"]) == 0
    assert st.sh_degree_of([f"f_rest_{i}" for i in range(45)]) == 3
    with pytest.raises(ValueError):
        st.sh_degree_of([f"f_rest_{i}" for i in range(10)])


# ------------------------------------------------------------------ centring


def _person(world_R, world_t, n=4000, seed=0):
    """A standing figure (1.8 tall, feet at 0, along +y of its own frame)
    expressed in a tilted, shifted SfM world: X_world = world_R X + world_t.
    Plus faint floaters far away."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1.8, n)
    a = rng.uniform(0, 2 * math.pi, n)
    r = 0.2 + 0.05 * np.sin(h * 3)
    local = np.stack([r * np.cos(a), h, r * np.sin(a)], 1)
    v = np.zeros(n + 40, np.dtype([(k, "<f4") for k in ("x", "y", "z", "opacity")]))
    P = np.concatenate([local @ world_R.T + world_t, rng.uniform(-30, 30, (40, 3))])
    for i, k in enumerate("xyz"):
        v[k] = P[:, i]
    v["opacity"] = np.r_[np.full(n, 3.0), np.full(40, -4.0)]  # sigmoid 0.95 / 0.02
    return v


def _cameras(world_R, world_t, k=24, radius=3.0, height=1.2):
    """Cameras orbiting the figure (COLMAP convention: image y down = local -y),
    the first one on the figure's local +z side, looking at it."""
    views = []
    target = np.array([0, 0.9, 0])
    for i in range(k):
        th = 2 * math.pi * i / k
        c = np.array([radius * math.sin(th), height, radius * math.cos(th)])
        f = (target - c) / np.linalg.norm(target - c)  # camera z (forward)
        x = np.cross([0, -1, 0], f)  # OpenCV: x = y cross z, camera y down (local -y)
        x /= np.linalg.norm(x)
        y = np.cross(f, x)
        Rc = np.stack([x, y, f])  # world(local) -> camera
        Rw = Rc @ world_R.T  # world(SfM) -> camera
        cw = world_R @ c + world_t
        views.append(SimpleNamespace(name=f"{i:05d}.png", R=Rw, t=-Rw @ cw))
    return views


WORLD_R = st.rotation_xyz(160, 25, -40)  # COLMAP-like: roughly upside down and tilted
WORLD_T = np.array([4.0, -2.0, 7.0])


def _apply(v, M):
    P = np.stack([v[k] for k in "xyz"], 1).astype(np.float64)
    return P @ M[:3, :3].T + M[:3, 3]


def test_center_feet_upright_with_cameras():
    v = _person(WORLD_R, WORLD_T)
    M, info = st.center_transform(v, views=_cameras(WORLD_R, WORLD_T))
    assert info["up"] == "cameras" and info["opaque"] == 4000
    P = _apply(v, M)[:4000]
    lo, hi = np.percentile(P, [1, 99], axis=0)
    assert abs(lo[1]) < 0.03 and 1.7 < hi[1] < 1.8  # feet on the floor, standing along +y
    assert np.abs(np.median(P[:, [0, 2]], axis=0)).max() < 0.03  # centred horizontally
    s, R, _ = st.decompose(M)
    assert s == pytest.approx(1.0)
    np.testing.assert_allclose(R @ WORLD_R, np.eye(3), atol=0.02)  # the figure's own frame back
    # the first camera ends up in front (+z)
    c0 = -np.asarray(_cameras(WORLD_R, WORLD_T)[0].R).T @ _cameras(WORLD_R, WORLD_T)[0].t
    assert (R @ c0 + M[:3, 3])[2] > 2.5


@pytest.mark.parametrize("axis", ["-y", "+z"])
def test_center_other_up_axes_and_height(axis):
    v = _person(WORLD_R, WORLD_T)
    M, info = st.center_transform(v, views=_cameras(WORLD_R, WORLD_T), up_axis=axis, height=1.0)
    P = _apply(v, M)[:4000]
    k, sign = {"-y": (1, -1), "+z": (2, 1)}[axis]
    up = sign * P[:, k]
    assert abs(np.percentile(up, 1)) < 0.02 and np.percentile(up, 99) == pytest.approx(1.0, abs=0.02)
    assert info["scale"] == pytest.approx(1.0 / 1.76, rel=0.03)


def test_center_from_shape_without_cameras():
    v = _person(WORLD_R, WORLD_T)
    M, info = st.center_transform(v, anchor="center")
    assert info["up"] == "shape"
    P = _apply(v, M)[:4000]
    np.testing.assert_allclose(np.median(P, axis=0), 0, atol=0.03)
    assert np.percentile(P[:, 1], 99) - np.percentile(P[:, 1], 1) > 1.7  # long axis along y


def test_center_ignores_floaters_and_anchor_none():
    v = _person(np.eye(3), np.zeros(3))
    M, _ = st.center_transform(v, anchor="bbox_center", upright=False)
    np.testing.assert_allclose(M[:3, :3], np.eye(3))
    assert np.abs(M[:3, 3] - [0, -0.9, 0]).max() < 0.05  # the floaters don't move the box
    M, _ = st.center_transform(v, anchor="none", upright=False)
    np.testing.assert_allclose(M, np.eye(4))
    with pytest.raises(ValueError):
        st.center_transform(v, anchor="middle")


def test_header_vertical_axis_follows_the_rotation():
    M = st.similarity(2.0, st.rotation_xyz(90, 0, 0), (1, 2, 3))
    h = st.transform_header(["ply", "comment Exported from Brush", "comment Vertical axis: 0 -1 0",
                             "comment vertical axis: z", "comment Vertical axis: garbage"], M)
    assert h[:2] == ["ply", "comment Exported from Brush"] and h[4] == "comment Vertical axis: garbage"
    # "0 -1 0": the model's up is -y (COLMAP); rotated 90 deg about x it is -z, which Brush writes as 0 0 1
    np.testing.assert_allclose([float(x) for x in h[2].split(":")[1].split()], [0, 0, 1], atol=1e-9)
    # "z" (Brush: v = -z, up +z) -> up -y -> v = 0 -1 0
    np.testing.assert_allclose([float(x) for x in h[3].split(":")[1].split()], [0, -1, 0], atol=1e-9)


def test_brush_vertical_axis_convention():
    """v -> Brush's model turn rotation_arc(-Y, v) must stand the model's up
    (F v) on the screen's up (-Y)."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(3)
    for _ in range(5):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        m = np.array([0.0, -1.0, 0.0])
        axis = np.cross(m, v)
        Q = Rotation.from_rotvec(axis / np.linalg.norm(axis) * math.acos(np.clip(m @ v, -1, 1))).as_matrix()
        np.testing.assert_allclose(Q @ (np.diag([-1.0, 1, -1]) @ v), m, atol=1e-9)
