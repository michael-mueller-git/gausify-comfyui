"""Splat clean-up on a synthetic orbit: a sphere subject, floaters around it,
disc masks (the sphere's silhouette)."""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from gausify.posecheck.job import View
from gausify.splat_clean import clean_splat, orbit_centre, read_ply, render_psnr, write_ply

W, H, F = 400, 300, 500.0
RADIUS = 4.0
SUBJECT = 0.5


def orbit_views(n=24, centre=(0.3, 0.0, -0.2)):
    c0 = np.asarray(centre, dtype=np.float64)
    views = []
    for i in range(n):
        a = 2 * math.pi * i / n
        C = c0 + RADIUS * np.array([math.sin(a), 0.0, -math.cos(a)])
        z = (c0 - C) / np.linalg.norm(c0 - C)
        y = np.array([0.0, 1.0, 0.0])  # COLMAP: y down
        x = np.cross(y, z)
        R = np.stack([x, y, z])
        views.append(View(name=f"{i:05d}.png", width=W, height=H, fx=F, fy=F, cx=W / 2, cy=H / 2, R=R, t=-R @ C))
    return views


def disc_mask(v, centre):
    Xc = v.R @ np.asarray(centre) + v.t
    u, w = F * Xc[0] / Xc[2] + v.cx, F * Xc[1] / Xc[2] + v.cy
    r = F * SUBJECT / math.sqrt(Xc[2] ** 2 - SUBJECT**2)
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (int(round(u)), int(round(w))), int(math.ceil(r)), 255, -1)
    return m


HEADER = [
    "ply",
    "format binary_little_endian 1.0",
    "comment Exported from Brush",
    "comment SH degree: 1",
    "element vertex 0",
    *[f"property float {n}" for n in ("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
                                      "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")],
    *[f"property float f_rest_{i}" for i in range(9)],
]


def make_ply(path, xyz, opacity=None, scale=None):
    from gausify.splat_clean import _PLY_TYPES, Ply

    fields = [(h.split()[2], _PLY_TYPES[h.split()[1]]) for h in HEADER if h.startswith("property")]
    v = np.zeros(len(xyz), dtype=np.dtype(fields))
    v["x"], v["y"], v["z"] = xyz.T
    v["opacity"] = 3.0 if opacity is None else opacity
    for k in ("scale_0", "scale_1", "scale_2"):
        v[k] = math.log(0.01) if scale is None else scale
    v["rot_0"] = 1.0
    v["f_rest_8"] = np.arange(len(xyz))  # marker to check records stay intact
    write_ply(path, Ply(header=list(HEADER), vertex=v))


def scene(rng, centre):
    d = rng.normal(size=(2000, 3))
    subject = np.asarray(centre) + d / np.linalg.norm(d, axis=1, keepdims=True) * SUBJECT * rng.uniform(0, 1, (2000, 1))
    fl = rng.uniform(-2.5, 2.5, size=(4000, 3))
    fl = fl[np.linalg.norm(fl[:, [0, 2]], axis=1) > 1.2][:500] + np.asarray(centre)
    return subject, fl


def test_ply_roundtrip(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    make_ply(tmp_path / "a.ply", xyz)
    p = read_ply(tmp_path / "a.ply")
    assert len(p.vertex) == 10 and "comment SH degree: 1" in p.header
    write_ply(tmp_path / "b.ply", p, p.vertex[::2])
    q = read_ply(tmp_path / "b.ply")
    assert len(q.vertex) == 5 and np.array_equal(q.vertex, p.vertex[::2])
    assert (tmp_path / "b.ply").read_bytes().count(b"element vertex 5\n") == 1


def test_read_ply_rejects_ascii(tmp_path):
    (tmp_path / "a.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n")
    with pytest.raises(ValueError, match="little-endian"):
        read_ply(tmp_path / "a.ply")


def test_orbit_centre():
    c = (0.3, 0.0, -0.2)
    centre, radius = orbit_centre(orbit_views(centre=c))
    assert np.allclose(centre, c, atol=1e-6) and radius == pytest.approx(RADIUS)


def test_hull_carving_keeps_subject_removes_floaters(tmp_path):
    c = (0.3, 0.0, -0.2)
    views = orbit_views(centre=c)
    subject, floaters = scene(np.random.default_rng(1), c)
    xyz = np.concatenate([subject, floaters]).astype(np.float32)
    make_ply(tmp_path / "in.ply", xyz)
    masks = {v.name: disc_mask(v, c) for v in views}
    s = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name])
    out = read_ply(tmp_path / "out.ply").vertex
    ids = set(out["f_rest_8"].astype(int).tolist())
    n_sub = len(subject)
    assert set(range(n_sub)) <= ids  # every subject splat kept
    kept_floaters = len(ids) - n_sub
    assert kept_floaters <= 0.05 * len(floaters), kept_floaters
    assert s["input"] == len(xyz) and s["kept"] == len(out) and s["removed"]["hull"] + s["removed"]["unseen"] > 0.9 * len(floaters)
    # the records are intact
    sel = out["f_rest_8"].astype(int)
    assert np.allclose(np.stack([out["x"], out["y"], out["z"]], 1), xyz[sel])


def test_partial_mask_loss_is_forgiven(tmp_path):
    """A mask that misses the subject in a few frames does not carve it."""
    c = (0.0, 0.0, 0.0)
    views = orbit_views(centre=c)
    subject, _ = scene(np.random.default_rng(2), c)
    make_ply(tmp_path / "in.ply", subject.astype(np.float32))
    masks = {v.name: disc_mask(v, c) for v in views}
    for v in views[:4]:  # 4/24 frames lose the subject
        masks[v.name][:] = 0
    s = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name], hull_tolerance=0.25)
    assert s["kept"] == len(subject)
    s = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name], hull_tolerance=0.1)
    assert s["kept"] < len(subject)


def test_unseen_distance_opacity_size(tmp_path):
    views = orbit_views(centre=(0, 0, 0))
    xyz = np.array([[0, 0, 0], [0, -50, 0], [0.2, 0, 0], [0, 0, 1.0], [0, 0.1, 0], [3.5, 0, 0]], np.float32)
    # 1: far above every view (unseen); 5: beyond 0.8 x radius (distance)
    opac = np.array([3, 3, -6, 3, 3, 3], np.float32)  # 2: sigmoid(-6) ~ 0.0025
    scale = np.log(np.array([0.01, 0.01, 0.01, 2.0, 0.01, 0.01], np.float32))  # 3: huge
    make_ply(tmp_path / "in.ply", xyz, opac, scale)
    s = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, None, max_distance=0.8, min_opacity=0.01,
                    max_size=0.1)
    out = read_ply(tmp_path / "out.ply").vertex
    assert sorted(out["f_rest_8"].astype(int).tolist()) == [0, 4]
    assert s["removed"] == {"unseen": 1, "distance": 1, "opacity": 1, "size": 1}
    # all filters off: nothing removed
    s = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, None, drop_unseen=False)
    assert s["kept"] == 6 and s["removed"] == {}


def test_render_psnr_uses_scaled_cameras():
    views = orbit_views(n=2)
    seen = []

    def render(ply, cams):
        seen.append(cams)
        return [{"rgba": np.full((c["height"], c["width"], 4), 0.5 if ply == "a" else 0.25, np.float32)} for c in cams]

    vals = render_psnr(["a", "b"], views, lambda v, w, h: np.full((h, w, 3), 0.5), max_side=200, render=render)
    assert math.isinf(vals[0]) and vals[1] == pytest.approx(10 * math.log10(1 / 0.0625))
    c = seen[0][0]
    assert (c["width"], c["height"]) == (200, 150) and c["fx"] == pytest.approx(F / 2)


# ---------------------------------------------------------------- node


def test_clean_node(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from gausify.colmap_io import write_model
    from gausify.types import OrbitCameras, Pose, SfmModel
    from nodes import clean as clean_node
    from nodes import common

    out_dir = tmp_path / "output"
    (out_dir / "g").mkdir(parents=True)
    fp = types.SimpleNamespace(get_output_directory=lambda: str(out_dir), get_temp_directory=lambda: str(tmp_path))
    monkeypatch.setattr(common, "folder_paths", fp)

    c = (0.0, 0.0, 0.0)
    views = orbit_views(n=12, centre=c)
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for v in views:
        cv2.imwrite(str(root / "images" / v.name), np.full((H, W, 3), 128, np.uint8))
        cv2.imwrite(str(root / "masks" / v.name), disc_mask(v, c))
    model = SfmModel(
        width=W, height=H, f=F, cx=W / 2, cy=H / 2,
        poses=[Pose(R=np.asarray(v.R), t=np.asarray(v.t)) for v in views],
        points=[], names=[v.name for v in views],
    )  # fmt: skip
    write_model(model, root / "sparse" / "0", model.names)
    cams = OrbitCameras(root=root, names=model.names, width=W, height=H, masked=True)
    subject, floaters = scene(np.random.default_rng(3), c)
    make_ply(out_dir / "g" / "orbit_00001.ply", np.concatenate([subject, floaters]).astype(np.float32))

    renders = []

    def fake_render(ply, cams_):
        renders.append(Path(ply).name)
        return [{"rgba": np.zeros((k["height"], k["width"], 4), np.float32)} for k in cams_]

    monkeypatch.setattr("gausify.brush.render", fake_render)
    node = clean_node.GausifyCleanSplat()
    r = node.clean(cams, "g/orbit_00001.ply", True, 0.25, 1.0, True, 0.0, 0.0, 0.0, True)
    rel, n, stats = r["result"]
    s = json.loads(stats)
    assert rel == "g/orbit_00001_clean.ply" and (out_dir / rel).is_file()
    assert len(subject) <= n <= len(subject) + 0.05 * len(floaters)
    assert s["removed"]["hull"] > 0 and "change" in s["psnr"]
    assert renders == ["orbit_00001.ply", "orbit_00001_clean.ply"]
    # unmasked dataset: no carving, a note
    cams.masked = False
    r = node.clean(cams, "g/orbit_00001.ply", True, 0.25, 1.0, True, 0.0, 0.0, 0.0, False)
    s = json.loads(r["result"][2])
    assert "hull" not in s["removed"] and "note" in s and "psnr" not in s
    with pytest.raises(FileNotFoundError):
        node.clean(cams, "g/missing.ply", True, 0.25, 1.0, True, 0.0, 0.0, 0.0, False)


def test_hull_extent_removes_spilling_blobs(tmp_path):
    c = (0.0, 0.0, 0.0)
    views = orbit_views(centre=c)
    xyz = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.45, 0]], np.float32)
    # 0: small splat at the centre; 1: blob of radius ~1.2 (2 sigma) centred on the subject;
    # 2: small splat just inside the silhouette's edge
    scale = np.log(np.array([0.01, 0.6, 0.01], np.float32))
    make_ply(tmp_path / "in.ply", xyz, scale=scale)
    masks = {v.name: disc_mask(v, c) for v in views}
    s = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name])
    assert s["kept"] == 3  # by centre: all inside
    s = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name], hull_extent=True)
    out = read_ply(tmp_path / "out.ply").vertex
    assert sorted(out["f_rest_8"].astype(int).tolist()) == [0, 2] and s["removed"]["hull"] == 1


def test_cache_reuses_the_view_analysis(tmp_path, monkeypatch):
    import gausify.splat_clean as sc

    c = (0.0, 0.0, 0.0)
    views = orbit_views(centre=c)
    subject, floaters = scene(np.random.default_rng(4), c)
    make_ply(tmp_path / "in.ply", np.concatenate([subject, floaters]).astype(np.float32))
    masks = {v.name: disc_mask(v, c) for v in views}
    calls = []
    real = sc.hull_counts
    monkeypatch.setattr(sc, "hull_counts", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache: dict = {}
    a = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name], hull_tolerance=0.1,
                    cache=cache)
    b = clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name], hull_tolerance=0.9,
                    cache=cache)
    assert len(calls) == 1 and b["kept"] >= a["kept"] and b["hullTolerance"] == 0.9
    clean_splat(tmp_path / "in.ply", tmp_path / "out.ply", views, lambda v: masks[v.name], mask_margin=0.05,
                cache=cache)
    assert len(calls) == 2  # another margin: recomputed


def test_clean_node_auto_tolerance(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from nodes import clean as clean_node

    cams, out_dir = _node_dataset(monkeypatch, tmp_path)
    seen_tols = []
    import gausify.splat_clean as sc

    real_clean = sc.clean_splat

    def spy(*a, **k):
        seen_tols.append(k["hull_tolerance"])
        return real_clean(*a, **k)

    monkeypatch.setattr(sc, "clean_splat", spy)

    n0 = len(read_ply(out_dir / "g" / "orbit_00001.ply").vertex)

    def fake_psnr(plys, views, target, max_side=800, render=None):
        # PSNR falls by 0.001 dB per removed splat
        return [30.0 - 0.001 * (n0 - len(read_ply(p).vertex)) for p in plys]

    monkeypatch.setattr(sc, "render_psnr", fake_psnr)
    node = clean_node.GausifyCleanSplat()
    r = node.clean(cams, "g/orbit_00001.ply", True, 0.0, 1.0, True, 0.0, 0.0, 0.0, True, False, True, 0.01)
    s = json.loads(r["result"][2])
    assert len(seen_tols) >= 2 and seen_tols == sorted(seen_tols) and seen_tols[0] == 0.0
    assert s["auto"]["tries"][0]["tolerance"] == 0.0 and len(s["auto"]["tries"]) == len(seen_tols)
    assert s["hullTolerance"] == seen_tols[-1]
    # off: a single pass
    seen_tols.clear()
    node.clean(cams, "g/orbit_00001.ply", True, 0.0, 1.0, True, 0.0, 0.0, 0.0, True, False, False, 0.01)
    assert seen_tols == [0.0]


def _node_dataset(monkeypatch, tmp_path):
    from gausify.colmap_io import write_model
    from gausify.types import OrbitCameras, Pose, SfmModel
    from nodes import common

    out_dir = tmp_path / "output"
    (out_dir / "g").mkdir(parents=True)
    fp = types.SimpleNamespace(get_output_directory=lambda: str(out_dir), get_temp_directory=lambda: str(tmp_path))
    monkeypatch.setattr(common, "folder_paths", fp)
    c = (0.0, 0.0, 0.0)
    views = orbit_views(n=12, centre=c)
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for v in views:
        cv2.imwrite(str(root / "images" / v.name), np.full((H, W, 3), 128, np.uint8))
        cv2.imwrite(str(root / "masks" / v.name), disc_mask(v, c))
    model = SfmModel(width=W, height=H, f=F, cx=W / 2, cy=H / 2,
                     poses=[Pose(R=np.asarray(v.R), t=np.asarray(v.t)) for v in views], points=[],
                     names=[v.name for v in views])
    write_model(model, root / "sparse" / "0", model.names)
    cams = OrbitCameras(root=root, names=model.names, width=W, height=H, masked=True)
    subject, floaters = scene(np.random.default_rng(3), c)
    make_ply(out_dir / "g" / "orbit_00001.ply", np.concatenate([subject, floaters]).astype(np.float32))
    return cams, out_dir
