"""Exposure / white-balance deflicker."""

from __future__ import annotations

import numpy as np
from PIL import Image

from gausify import deflicker as df


def _write(path, rgb, alpha=None):
    if alpha is None:
        Image.fromarray(rgb, "RGB").save(path)
    else:
        Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)


def _frame(level, h=32, w=40, tint=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(0)
    base = rng.integers(60, 160, (h, w, 3)).astype(np.float64)
    return np.clip(base * level * np.array(tint), 0, 255).astype(np.uint8)


def test_gains_remove_flicker_keep_slow_change():
    n = 30
    slow = np.linspace(-0.3, 0.3, n)[:, None] * np.ones((1, 3))  # real lighting change around the orbit
    flick = slow.copy()
    flick[10] += 0.2
    flick[20, 2] -= 0.15  # a white-balance jump in blue
    g = df.gains(flick, 1.0)
    fixed = flick + np.log(g)
    np.testing.assert_allclose(fixed[10], slow[10], atol=0.025)  # within one ramp step
    np.testing.assert_allclose(fixed[20], slow[20], atol=0.025)
    # elsewhere the slow ramp stays (gains ~1, not flattened to the mean)
    assert np.abs(np.log(g[[0, 5, 15, 25, 29]])).max() < 0.03
    # half strength: half the correction (in log)
    np.testing.assert_allclose(np.log(df.gains(flick, 0.5)[10]), 0.5 * np.log(g[10]), atol=1e-9)
    # capped, and unmeasured frames untouched
    big = np.zeros((9, 3))
    big[4] = 3.0
    big[2] = np.nan
    gb = df.gains(big, 1.0)
    assert np.isclose(gb[4], 1 / df.MAX_GAIN).all() and (gb[2] == 1).all()


def test_deflicker_images(tmp_path):
    paths = []
    for i in range(9):
        p = tmp_path / f"{i:05d}.png"
        level = 1.35 if i == 4 else 1.0
        tint = (1.0, 1.0, 0.8) if i == 6 else (1.0, 1.0, 1.0)
        alpha = np.zeros((32, 40), np.uint8)
        alpha[4:28, 5:35] = 255
        _write(p, _frame(level, tint=tint), alpha)
        paths.append(p)
    before = np.asarray(Image.open(paths[0])).copy()
    s = df.deflicker_images(paths, 1.0)
    assert s["frames"] == 9 and s["maxCorrection"] > 20
    ref = np.asarray(Image.open(paths[0]))
    for k in (4, 6):
        im = np.asarray(Image.open(paths[k]))
        assert im.shape == (32, 40, 4) and (im[..., 3] == ref[..., 3]).all()  # alpha untouched
        m = ref[..., 3] > 127
        ratio = im[..., :3][m].astype(float).mean(0) / ref[..., :3][m].astype(float).mean(0)
        np.testing.assert_allclose(ratio, 1.0, atol=0.03)
    assert (np.asarray(Image.open(paths[0])) == before).all()  # steady frames unchanged
    # off: nothing written
    assert df.deflicker_images(paths, 0.0)["maxCorrection"] == 0.0


def test_deflicker_rgb_and_unmeasurable(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.png"
        _write(p, _frame(1.3 if i == 2 else 1.0))
        paths.append(p)
    dark = tmp_path / "5.png"
    _write(dark, np.zeros((32, 40, 3), np.uint8))  # nothing to measure: left alone
    paths.append(dark)
    df.deflicker_images(paths, 1.0)
    a, b = (np.asarray(Image.open(paths[k])).astype(float).mean() for k in (1, 2))
    assert abs(b / a - 1) < 0.04 and np.asarray(Image.open(dark)).max() == 0
