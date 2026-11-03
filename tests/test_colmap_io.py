import struct

import numpy as np
import pytest

from gausify.colmap_io import colmap_files, mat_from_quat, quat_from_mat, read_model, rename_images, write_model
from gausify.types import Point3D, Pose, SfmModel


def _rot(axis, th):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def make_model():
    names = [f"{i:05d}.jpg" for i in range(4)]
    poses = [Pose(_rot([0, 1, 0.2], 0.3 * i), np.array([0.1 * i, -0.2, 4.0])) for i in range(4)]
    poses[2] = None
    rng = np.random.default_rng(0)
    xy = [rng.uniform(0, 600, size=(10, 2)).astype(np.float32) for _ in names]
    points = [
        Point3D(np.array([0.1, 0.2, 0.3]), (10, 20, 30), [(0, 4), (1, 7), (3, 1)], 0.5),
        Point3D(np.array([-0.3, 0.0, 0.1]), (255, 0, 128), [(1, 2), (0, 9)], 1.25),
    ]
    return SfmModel(f=700.0, cx=320.0, cy=240.0, width=640, height=480, poses=poses, points=points, names=names, xy=xy)


def test_quat_roundtrip():
    for R in [np.eye(3), _rot([1, 0, 0], 3.0), _rot([0, 1, 1], 2.5), _rot([1, -2, 0.5], 1.1), _rot([0, 0, 1], np.pi)]:
        q = quat_from_mat(R)
        assert q[0] >= 0
        np.testing.assert_allclose(mat_from_quat(q), R, atol=1e-12)


def test_byte_layout():
    m = make_model()
    files = colmap_files(m)
    cams = files["cameras.bin"]
    assert struct.unpack("<QIiQQddd", cams) == (1, 1, 0, 640, 480, 700.0, 320.0, 240.0)
    img = files["images.bin"]
    (n,) = struct.unpack_from("<Q", img, 0)
    assert n == 3
    iid, *qt, cid = struct.unpack_from("<I7dI", img, 8)
    assert iid == 1 and cid == 1
    assert img[72:82] == b"00000.jpg\0"
    (nobs,) = struct.unpack_from("<Q", img, 82)
    assert nobs == 2  # (feat 4 -> point 1), (feat 9 -> point 2), in point order
    x, y, pid = struct.unpack_from("<ddq", img, 90)
    assert (x, y, pid) == (float(m.xy[0][4, 0]), float(m.xy[0][4, 1]), 1)
    x, y, pid = struct.unpack_from("<ddq", img, 114)
    assert pid == 2 and x == float(m.xy[0][9, 0])
    pts = files["points3D.bin"]
    (n,) = struct.unpack_from("<Q", pts, 0)
    assert n == 2
    pid, X, Y, Z, r, g, b, err, tl = struct.unpack_from("<Q3d3BdQ", pts, 8)
    assert (pid, r, g, b, err, tl) == (1, 10, 20, 30, 0.5, 3)
    # track: (image_id, points2D index in that image)
    assert struct.unpack_from("<6I", pts, 59) == (1, 0, 2, 0, 4, 0)


def test_write_read_roundtrip(tmp_path):
    m = make_model()
    write_model(m, tmp_path / "sparse" / "0")
    r = read_model(tmp_path / "sparse" / "0", names=m.names)
    assert (r.f, r.cx, r.cy, r.width, r.height) == (700.0, 320.0, 240.0, 640, 480)
    assert r.poses[2] is None
    for a, b in zip(m.poses, r.poses):
        if a is not None:
            np.testing.assert_allclose(b.R, a.R, atol=1e-12)
            np.testing.assert_allclose(b.t, a.t)
    assert len(r.points) == 2
    for a, b in zip(m.points, r.points):
        np.testing.assert_allclose(b.X, a.X)
        assert b.rgb == a.rgb and b.error == a.error
        for (i, k), (i2, k2) in zip(a.obs, b.obs):
            assert i == i2
            np.testing.assert_allclose(r.xy[i2][k2], m.xy[i][k])
    # without names: registered names only
    r2 = read_model(tmp_path / "sparse" / "0")
    assert r2.names == ["00000.jpg", "00001.jpg", "00003.jpg"]


def test_rename_images(tmp_path):
    m = make_model()
    d = write_model(m, tmp_path / "s")
    before = (d / "images.bin").read_bytes()
    out = rename_images(d, lambda n: n.replace(".jpg", ".png"), out_dir=tmp_path / "o")
    data = out.read_bytes()
    assert len(data) == len(before)
    assert data.replace(b".png", b".jpg") == before
    r = read_model(d, names=m.names)
    (tmp_path / "o" / "cameras.bin").write_bytes((d / "cameras.bin").read_bytes())
    (tmp_path / "o" / "points3D.bin").write_bytes((d / "points3D.bin").read_bytes())
    r2 = read_model(tmp_path / "o", names=[n.replace(".jpg", ".png") for n in m.names])
    np.testing.assert_allclose(r2.poses[3].t, r.poses[3].t)


def test_pycolmap_reads_it(tmp_path):
    try:
        from gausify import cuda_libs  # noqa: F401  (another agent's module)

        cuda_libs.preload()
    except Exception:
        pass
    pycolmap = pytest.importorskip("pycolmap")
    m = make_model()
    d = write_model(m, tmp_path / "s")
    rec = pycolmap.Reconstruction(str(d))
    assert rec.num_reg_images() == 3 and rec.num_points3D() == 2
    cam = rec.cameras[1]
    assert cam.model.name == "SIMPLE_PINHOLE"
    img = rec.images[2]
    assert img.name == "00001.jpg"
    np.testing.assert_allclose(img.cam_from_world().rotation.matrix(), m.poses[1].R, atol=1e-9)
