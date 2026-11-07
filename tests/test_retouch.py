"""Port of src/lib/retouch/{trajectory,frames,colmap}.test.ts (the ported
parts) plus the retouch dataset and nodes."""

from __future__ import annotations

import json
import math
import struct
import types
from pathlib import Path

import numpy as np
import pytest

from gausify.retouch import (
    Lens,
    Pose,
    RetouchClip,
    append_cameras,
    append_images,
    build_retouch_dataset,
    clip_frames,
    clip_lens,
    mat_to_quat,
    motion_path,
    orbit_frame,
    peak_speed,
    pick_restored_frames,
    preset_path,
    quat_to_mat,
    read_lens,
    read_poses,
    restored_limit,
    sh_degree_of,
    to_world,
)

UP = np.array([0.0, -1.0, 0.0])


def look_at(az_deg, el_deg, r, name=""):
    az, el = math.radians(az_deg), math.radians(el_deg)
    c = np.array([r * math.cos(el) * math.sin(az), -r * math.sin(el), r * math.cos(el) * math.cos(az)])
    f = -c / np.linalg.norm(c)
    d = -(UP - f * (UP @ f))
    down = d / np.linalg.norm(d)
    right = np.cross(down, f)
    R = np.stack([right, down, f])
    return Pose(qvec=mat_to_quat(R), tvec=-(R @ c), name=name)


ORBIT = [look_at(i * 6, 10, 3, f"{i:05d}.jpg") for i in range(60)]
LENS = Lens(width=960, height=1920, fx=1600, fy=1600)


def elevation(p):
    c = to_world(p)[1]
    return math.degrees(math.asin(float(c @ UP) / np.linalg.norm(c)))


def angle_between(a, b):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def test_quaternion_round_trip():
    for p in ORBIT[:10]:
        assert np.allclose(mat_to_quat(quat_to_mat(p.qvec)), p.qvec, atol=1e-9)


def test_orbit_frame():
    center, up = orbit_frame(ORBIT)
    assert np.linalg.norm(center) < 1e-6 and up @ UP > 0.999


def test_clip_lens():
    assert clip_lens(LENS, 960) == ((480, 960), {"fx": 800, "fy": 800, "cx": 240, "cy": 480})
    assert clip_lens(Lens(1920, 1080, 1500, 1500), 960)[0] == (960, 544)


def test_preset_minus_30():
    p = preset_path(ORBIT, 20, -30, 97)
    assert len(p) == 97
    assert p[0].name == ORBIT[56].name and p[96].name == ORBIT[44].name
    assert angle_between(p[0].qvec, ORBIT[56].qvec) < 1e-4
    assert abs(elevation(p[48]) - (10 - 30)) < 0.5
    assert abs(np.linalg.norm(to_world(p[48])[1]) - 3) < 1e-6
    for i in range(1, len(p)):
        assert angle_between(p[i - 1].qvec, p[i].qvec) < 4, i


def test_clip_frames():
    assert clip_frames(97 / 24, 24) == 97
    assert clip_frames(3.2, 30) == 97
    assert clip_frames(8, 30) == 241
    assert clip_frames(0.1, 30) == 17


def test_360_below():
    frames = 241
    p = preset_path(ORBIT, 10 + len(ORBIT) / 2, -25, frames, span=len(ORBIT))
    assert p[0].name == ORBIT[10].name and p[frames - 1].name == ORBIT[10].name
    assert abs(elevation(p[frames >> 1]) - (10 - 25)) < 0.5
    for i in range(1, len(p)):
        assert angle_between(p[i - 1].qvec, p[i].qvec) < 4, i


def test_360_wave():
    frames = 241
    p = preset_path(ORBIT, len(ORBIT) / 2, -20, frames, span=len(ORBIT), shape="wave", ramp=0.1)
    els = [elevation(q) for q in p]
    assert min(els[: frames >> 1]) < 10 - 18 and max(els[frames >> 1 :]) > 10 + 18
    assert abs(els[0] - 10) < 1e-6 and abs(els[-1] - 10) < 1e-6
    assert p[0].name == ORBIT[0].name


def test_speed_flat_turn():
    flat = preset_path(ORBIT, 30, 0, 97, span=len(ORBIT))
    assert abs(peak_speed(flat, 24) - 90) < 2


def test_irregular_spacing_constant_speed():
    az, a = [], 0.0
    while a < 360 - 1e-9:
        az.append(a)
        a += 3 if len(az) % 2 else 9
    az.append(359.97)
    cams = [look_at(x, 10, 3, f"{i:05d}.jpg") for i, x in enumerate(az)]
    p = preset_path(cams, len(cams) / 2, 0, 121, span=len(cams))
    assert p[0].name == cams[0].name and p[120].name == cams[0].name
    steps = [angle_between(p[i].qvec, p[i + 1].qvec) for i in range(120)]
    assert all(abs(s - 3.0) < 0.05 for s in steps), steps


def test_motion_presets():
    for m in ("wave", "orbit_below", "orbit_above"):
        p = motion_path(ORBIT, m, 7, 30, 97)
        assert p[0].name == ORBIT[7].name and p[-1].name == ORBIT[7].name, m
    below = motion_path(ORBIT, "orbit_below", 7, 30, 97)
    above = motion_path(ORBIT, "orbit_above", 7, 30, 97)
    assert elevation(below[48]) < 10 - 25 and elevation(above[48]) > 10 + 25
    arc = motion_path(ORBIT, "arc_above", 30, 20, 97)
    assert arc[0].name == ORBIT[20].name and arc[-1].name == ORBIT[40].name  # a third of the orbit around 30
    with pytest.raises(ValueError):
        motion_path(ORBIT, "spin", 0, 30, 97)


# ---------------------------------------------------------------- frames.ts


def test_restored_limit_and_pick():
    assert restored_limit(107, 0.25) == 27 and restored_limit(3, 0.05) == 1
    assert pick_restored_frames([3, 2], 10) == [[0, 1, 2], [0, 1]]
    a, b = pick_restored_frames([89, 177], 30)
    assert (len(a), len(b)) == (10, 20)
    for sel, n in ((a, 89), (b, 177)):
        assert sel[0] > 0 and sel[-1] < n - 1
        gaps = [y - x for x, y in zip(sel, sel[1:])]
        assert max(gaps) - min(gaps) <= 1
    sel = pick_restored_frames([500, 9], 10)
    assert len(sel[1]) == 1 and len(sel[0]) + len(sel[1]) == 10


# ---------------------------------------------------------------- colmap.ts


def cameras_bin():
    return struct.pack("<QiiQQ3d", 1, 1, 0, 960, 1920, 1600, 480, 960)


def images_bin():
    out = struct.pack("<Q", 2)
    for iid, name, pts in ((3, "00000.jpg", 0), (7, "00001.jpg", 1)):
        out += struct.pack("<i7di", iid, 1, 0, 0, 0, 0.1, 0.2, 0.3, 1) + name.encode() + b"\0" + struct.pack("<Q", pts)
        out += b"".join(struct.pack("<ddq", 1.5, 2.5, 9) for _ in range(pts))
    return out


def test_append_cameras_and_images():
    cams, ids = append_cameras(cameras_bin(), [{"width": 480, "height": 960, "fx": 800, "fy": 800, "cx": 240, "cy": 480}])
    assert ids == [2] and struct.unpack_from("<Q", cams)[0] == 2
    assert cams[: 8] != cameras_bin()[:8] and cams[8:56] == cameras_bin()[8:]
    assert struct.unpack_from("<iiQQ4d", cams, 56) == (2, 1, 480, 960, 800.0, 800.0, 240.0, 480.0)
    assert read_lens(cameras_bin()) == Lens(960, 1920, 1600.0, 1600.0)
    imgs = append_images(images_bin(), [{"name": "retouch_00001.png", "qvec": [1, 0, 0, 0], "tvec": [1, 2, 3],
                                          "camera": 0}], ids)
    poses = read_poses(imgs)
    assert [p.name for p in poses] == ["00000.jpg", "00001.jpg", "retouch_00001.png"]
    assert np.allclose(poses[2].tvec, [1, 2, 3])
    (iid, *_, cam) = struct.unpack_from("<i7di", imgs, len(images_bin()))
    assert (iid, cam) == (8, 2)


# ---------------------------------------------------------------- dataset


def _dataset(tmp_path, masked):
    import cv2

    from gausify.colmap_io import write_model
    from gausify.types import OrbitCameras, SfmModel
    from gausify.types import Pose as SfmPose

    names = [f"{i:05d}.png" for i in range(12)]
    cams = [look_at(i * 30, 10, 3, n) for i, n in enumerate(names)]
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for n in names:
        cv2.imwrite(str(root / "images" / n), np.full((64, 48, 3), 100, np.uint8))
        m = np.zeros((64, 48), np.uint8)
        m[16:48, 12:36] = 255
        cv2.imwrite(str(root / "masks" / n), m)
    model = SfmModel(f=50, cx=24, cy=32, width=48, height=64, names=names, points=[],
                     poses=[SfmPose(R=quat_to_mat(c.qvec), t=np.asarray(c.tvec)) for c in cams])
    write_model(model, root / "sparse" / "0", names)
    return OrbitCameras(root=root, names=names, width=48, height=64, masked=masked)


def _clip(photos, frames=17):
    size, k = clip_lens(read_lens_of(photos), 64)
    return RetouchClip(motion="wave", size=size, intrinsics=k, poses=motion_path(photos, "wave", 0, 30, frames),
                       fps=24.0, alpha=np.ones((frames, size[1], size[0]), np.float32))


def read_lens_of(_):
    return Lens(width=48, height=64, fx=50, fy=50)


@pytest.mark.parametrize("masked", [False, True])
def test_build_retouch_dataset(tmp_path, masked):
    from PIL import Image

    cams = _dataset(tmp_path, masked)
    photos = read_poses((cams.sparse_dir / "images.bin").read_bytes())
    clip = _clip(photos)
    clip.alpha[:, :4] = 0.0  # top rows transparent in the render
    w, h = clip.size
    restored = np.full((17, h * 2, w * 2, 3), 200, np.uint8)  # restored at twice the size
    out, stats = build_retouch_dataset(cams, clip, restored, clip.alpha, tmp_path / "rt", restored_ratio=0.25)
    assert stats["photos"] == 12 and stats["restored"] == 3 and stats["restoredSize"] == [w * 2, h * 2]
    poses = read_poses((out / "sparse" / "0" / "images.bin").read_bytes())
    names = [p.name for p in poses]
    assert sum(n.startswith("retouch_") for n in names) == 3 and len(names) == 15
    assert all((out / "images" / n).is_file() for n in names)
    rt = [n for n in names if n.startswith("retouch_")]
    assert rt == ["retouch_00002.png", "retouch_00008.png", "retouch_00014.png"]
    img = np.asarray(Image.open(out / "images" / rt[0]))
    assert img.shape == (h * 2, w * 2, 4) and img[0, 0, 3] == 0 and img[h, w, 3] == 255
    cam_data = (out / "sparse" / "0" / "cameras.bin").read_bytes()
    assert struct.unpack_from("<Q", cam_data)[0] == 2
    k = clip.intrinsics
    assert struct.unpack_from("<iiQQ4d", cam_data, len(cam_data) - 56)[2:] == (
        w * 2, h * 2, k["fx"] * 2, k["fy"] * 2, k["cx"] * 2, k["cy"] * 2)
    # the clip's restored frame keeps its pose
    i = next(j for j, p in enumerate(poses) if p.name == "retouch_00008.png")
    assert np.allclose(poses[i].qvec, clip.poses[8].qvec) and np.allclose(poses[i].tvec, clip.poses[8].tvec)
    if masked:
        assert all(n.endswith(".png") for n in names)
        assert np.asarray(Image.open(out / "images" / "00000.png")).shape[2] == 4


def test_build_retouch_dataset_rejects_mismatch(tmp_path):
    cams = _dataset(tmp_path, False)
    photos = read_poses((cams.sparse_dir / "images.bin").read_bytes())
    clip = _clip(photos)
    w, h = clip.size
    with pytest.raises(ValueError, match="frames"):
        build_retouch_dataset(cams, clip, np.zeros((16, h, w, 3), np.uint8), clip.alpha, tmp_path / "a")
    with pytest.raises(ValueError, match="aspect"):
        build_retouch_dataset(cams, clip, np.zeros((17, h, w * 2, 3), np.uint8), clip.alpha, tmp_path / "b")


def test_sh_degree_of(tmp_path):
    for d, rest in ((0, 0), (1, 9), (2, 24)):
        make_ply_with_rest(tmp_path / f"{d}.ply", rest)
        assert sh_degree_of(tmp_path / f"{d}.ply") == d


def make_ply_with_rest(path, rest):
    from gausify.splat_clean import Ply, write_ply

    props = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
             "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(rest)]]
    header = ["ply", "format binary_little_endian 1.0", "element vertex 0", *[f"property float {n}" for n in props]]
    v = np.zeros(4, dtype=np.dtype([(n, "<f4") for n in props]))
    write_ply(path, Ply(header=header, vertex=v))


# ---------------------------------------------------------------- nodes


def test_retouch_nodes(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from nodes import common
    from nodes.retouch import GausifyRetouchApply, GausifyRetouchClip
    out_dir = tmp_path / "output"
    (out_dir / "g").mkdir(parents=True)
    fp = types.SimpleNamespace(get_output_directory=lambda: str(out_dir), get_temp_directory=lambda: str(tmp_path))
    monkeypatch.setattr(common, "folder_paths", fp)
    cams = _dataset(tmp_path, True)
    make_ply_with_rest(out_dir / "g" / "orbit_00001.ply", 9)
    rendered = []

    def render(ply, cameras, color=True, progress=None, cancel=None):
        rendered.append(len(cameras))
        return [{"rgba": np.full((c["height"], c["width"], 4), 0.5, np.float32),
                 "alpha": np.ones((c["height"], c["width"]), np.float32),
                 "depth": np.ones((c["height"], c["width"]), np.float32)} for c in cameras]

    trained = {}

    def train(dataset_dir, args, out_ply, init_ply=None, progress=None, cancel=None):
        trained.update(args=list(args), init=Path(init_ply), names=sorted(p.name for p in (Path(dataset_dir) / "images").iterdir()))
        Path(out_ply).write_text("ply")
        return {"splats": 777}

    monkeypatch.setattr("gausify.brush.render", render)
    monkeypatch.setattr("gausify.brush.train", train)
    r = GausifyRetouchClip().clip(cams, "g/orbit_00001.ply", "orbit_below", 30.0, 3, 1.0, 24.0, 512)
    ref, alpha, clip, info = r["result"]
    n = clip_frames(1.0, 24.0)
    assert rendered == [n] and tuple(ref.shape)[0] == n and tuple(alpha.shape)[0] == n
    assert clip.poses[0].name == "00003.png" and json.loads(info)["frames"] == n
    r = GausifyRetouchApply().apply(cams, "g/orbit_00001.ply", clip, ref, 1000, 0.25, 2.0, 1600, True, True,
                                    False, 100, 5)
    rel, splats, stats = r["result"]
    a = trained["args"]
    assert rel == "g/orbit_00001_retouch.ply" and splats == 777 and (out_dir / rel).is_file()
    assert trained["init"] == out_dir / "g" / "orbit_00001.ply"
    assert float(a[a.index("--lr-mean") + 1]) == pytest.approx(2e-6)
    assert a[a.index("--growth-stop-iter") + 1] == "500" and a[a.index("--total-train-iters") + 1] == "1000"
    assert a[a.index("--sh-degree") + 1] == "1" and "--eval-split-every" not in a
    assert "--depth-distortion-weight" in a  # masked + reduce_haze
    assert sum(x.startswith("retouch_") for x in trained["names"]) == json.loads(stats)["restored"] == 3
    # a restored video with another frame count is refused
    with pytest.raises(ValueError, match="frames"):
        GausifyRetouchApply().apply(cams, "g/orbit_00001.ply", clip, ref[:-1], 1000, 0.25, 2.0, 1600, True,
                                    False, False, 100, 5)
    assert torch is not None
