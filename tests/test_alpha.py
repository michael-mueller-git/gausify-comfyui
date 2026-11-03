"""Ports of erode.test.ts, decontam.test.ts, chroma.test.ts (despill) and a
dataset round trip for build_alpha_dataset."""

import math

import numpy as np
import pytest
from PIL import Image

from gausify.alpha import (
    apply_despill,
    background_color,
    build_alpha_dataset,
    decontaminate,
    despill_for,
    erode,
)
from gausify.colmap_io import read_model, write_model
from gausify.types import OrbitCameras, Point3D, Pose, SfmModel


def test_erode_shrinks_one_pixel_not_at_border():
    w, h = 10, 8
    m = np.zeros((h, w), np.uint8)
    m[2:6, 2:6] = 255
    m[:, 9] = 255
    m[:, 8] = 128
    e = erode(m, 1)
    assert [e[2, 2], e[3, 3], e[4, 4], e[5, 5]] == [0, 255, 255, 0]
    assert [e[0, 9], e[7, 9], e[3, 8]] == [128, 128, 0]
    assert (erode(m, 0) == m).all()


def _ref_decontaminate(rgba, mask, alpha):
    """Literal loop port of decontam.ts (float32 sums, Float32Array colors)."""
    h, w = mask.shape
    B, R = 4, 3

    def field(pick):
        gw, gh = math.ceil(w / B), math.ceil(h / B)
        s = np.zeros((gh, gw, 4), np.float32)
        for y in range(h):
            for x in range(w):
                if pick(mask[y, x]):
                    s[y // B, x // B, :3] += rgba[y, x, :3]
                    s[y // B, x // B, 3] += 1
        for _ in range(2):
            for dx, dy in ((1, 0), (0, 1)):
                o = np.zeros_like(s)
                for y in range(gh):
                    for x in range(gw):
                        for k in range(-R, R + 1):
                            sx = min(max(x + k * dx, 0), gw - 1)
                            sy = min(max(y + k * dy, 0), gh - 1)
                            o[y, x] += s[sy, sx]
                s = o
        col = np.full((gh, gw, 3), np.nan, np.float32)
        for y in range(gh):
            for x in range(gw):
                if s[y, x, 3] > 1e-3:
                    col[y, x] = s[y, x, :3] / s[y, x, 3]
        return col, gw, gh

    bg, gw, gh = field(lambda m: m <= 5)
    so, _, _ = field(lambda m: m == 255)
    out = rgba.copy()
    for y in range(h):
        for x in range(w):
            a = int(alpha[y, x])
            if a in (0, 255):
                continue
            m = a / 255
            cy, cx = min(y // B, gh - 1), min(x // B, gw - 1)
            hb, hs = not math.isnan(bg[cy, cx, 0]), not math.isnan(so[cy, cx, 0])
            if not hb and not (hs and m < 0.25):
                continue
            mm = max(m, 0.25)
            us = hs and m < 0.25
            for c in range(3):
                v = float(so[cy, cx, c]) if us else (int(rgba[y, x, c]) - (1 - mm) * float(bg[cy, cx, c])) / mm
                out[y, x, c] = min(255, max(0, math.floor(v + 0.5)))
    return out


def _edge_case():
    w, h = 96, 64
    F, Bc = np.array([200, 30, 40]), np.array([240, 240, 235])
    xs = np.arange(w)
    row = np.clip((xs - 40) / 8 * 255, 0, 255)
    # Uint8Array assignment truncates
    mask = np.tile(row.astype(np.uint8), (h, 1))
    alpha = erode(mask, 1)
    m = alpha[..., None] / 255
    rgb = np.floor(m * F + (1 - m) * Bc + 0.5).astype(np.uint8)
    return rgb, mask, alpha, F


def test_decontam_red_subject_on_white_wall():
    rgb, mask, alpha, F = _edge_case()
    out = decontaminate(rgb, mask, alpha)
    dist = lambda px, x: np.linalg.norm(px[30, x].astype(float) - F)  # noqa: E731
    for x in range(41, 48):
        if 0 < alpha[30, x] < 255:
            assert dist(out, x) < 12
            assert dist(out, x) < dist(rgb, x)
    assert (out[30, 80] == rgb[30, 80]).all()
    assert (out[30, 5] == rgb[30, 5]).all()


def test_decontam_no_background_keeps_colors():
    mask = np.full((32, 32), 200, np.uint8)
    rgb = np.tile(np.array([10, 120, 230], np.uint8), (32, 32, 1))
    assert (decontaminate(rgb, mask, mask.copy()) == rgb).all()


def test_decontam_matches_reference_loop():
    rng = np.random.default_rng(3)
    h, w = 23, 30
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = rng.choice(np.array([0, 3, 40, 128, 250, 255], np.uint8), size=(h, w))
    mask[:, :8] = 0
    mask[:, 20:] = 255
    alpha = erode(mask, 1)
    np.testing.assert_array_equal(decontaminate(rgb, mask, alpha), _ref_decontaminate(rgb, mask, alpha))
    # default alpha = eroded mask
    np.testing.assert_array_equal(decontaminate(rgb, mask), _ref_decontaminate(rgb, mask, alpha))


LUMA = [0.2126, 0.7152, 0.0722]


def _apply(d, p):
    return apply_despill(np.array([[p]], np.uint8), np.array([[255]], np.uint8), d)[0, 0].tolist()


def test_despill_removes_green_spill_without_darkening():
    d = despill_for([30, 200, 40], 100)
    assert d
    ball = _apply(d, [30, 80, 40])
    assert ball[1] <= max(ball[0], ball[2])
    assert sum(v * LUMA[i] for i, v in enumerate(ball)) > 50
    for keep in ([200, 150, 120], [220, 220, 30], [60, 60, 60]):
        assert _apply(d, keep) == keep
    half = _apply(despill_for([30, 200, 40], 50), [40, 140, 40])
    assert half[1] > half[0] and half[1] < 140
    assert despill_for([20, 60, 210], 100)["channel"] == 2
    assert despill_for([150, 190, 150], 100) is None
    assert despill_for([210, 120, 60], 100) is None
    assert despill_for([30, 200, 40], 0) is None


def test_despill_exact_value():
    # spill = 80 - 40 = 40, gray = 40 * 0.7152 = 28.608
    assert _apply(despill_for([30, 200, 40], 100), [30, 80, 40]) == [59, 69, 69]


def test_despill_leaves_mask_zero():
    px = np.array([[[30, 200, 40]]], np.uint8)
    out = apply_despill(px, np.array([[0]], np.uint8), despill_for([30, 200, 40], 100))
    assert out.tolist() == px.tolist()


def test_background_color():
    rgb = np.zeros((60, 60, 3), np.uint8)
    rgb[...] = [20, 200, 30]
    rgb[:10] = [0, 0, 0]
    mask = np.zeros((60, 60), np.uint8)
    assert background_color([(rgb, mask)]) == (20, 200, 30)
    assert background_color([(rgb[:10], mask[:10])]) is None  # < 1000 samples


def test_build_alpha_dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rgb, mask, alpha, _ = _edge_case()
    names = ["00000.jpg", "00001.jpg", "00002.jpg"]
    for n in names:
        Image.fromarray(rgb).save(root / "images" / n.replace(".jpg", ".png"))
        Image.fromarray(mask).save(root / "masks" / n.replace(".jpg", ".png"))
    pngs = [n.replace(".jpg", ".png") for n in names]
    model = SfmModel(
        f=100.0, cx=48.0, cy=32.0, width=96, height=64,
        poses=[Pose(np.eye(3), np.array([0, 0, 4.0 + i])) for i in range(3)],
        points=[Point3D(np.zeros(3), (1, 2, 3), [(0, 0), (1, 0)])],
        names=pngs, xy=[np.array([[1.0, 2.0]])] * 3,
    )
    write_model(model, root / "sparse" / "0")
    cams = OrbitCameras(root=root, names=pngs, width=96, height=64, masked=True)
    out = build_alpha_dataset(cams, tmp_path / "alpha", despill=False)
    assert sorted(p.name for p in (out / "images").iterdir()) == pngs
    with Image.open(out / "images" / "00001.png") as im:
        assert im.mode == "RGBA"
        a = np.asarray(im)
    np.testing.assert_array_equal(a[..., 3], alpha)
    np.testing.assert_array_equal(a[..., :3], decontaminate(rgb, mask, alpha))
    r = read_model(out / "sparse" / "0", names=pngs)
    assert all(p is not None for p in r.poses)
    assert (out / "sparse" / "0" / "points3D.bin").read_bytes() == (root / "sparse" / "0" / "points3D.bin").read_bytes()
    # downscale keeps intrinsics
    out2 = build_alpha_dataset(cams, tmp_path / "alpha2", max_side=48)
    with Image.open(out2 / "images" / "00000.png") as im:
        assert im.size == (48, 32)
    assert read_model(out2 / "sparse" / "0", names=pngs).f == 100.0


if __name__ == "__main__":
    pytest.main([__file__])
