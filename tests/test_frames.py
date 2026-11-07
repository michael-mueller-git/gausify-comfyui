import math

import numpy as np
import pytest

from gausify.frames import pick_frames, select_frames, sharpness, windows_of


def ref_sharpness(rgb):
    """Literal loop port of sharpness() in frames.ts."""
    h, w = rgb.shape[:2]
    lum = [[round(0.2126 * int(p[0]) + 0.7152 * int(p[1]) + 0.0722 * int(p[2])) for p in row] for row in rgb]
    s = sq = n = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            lap = lum[y][x - 1] + lum[y][x + 1] + lum[y - 1][x] + lum[y + 1][x] - 4 * lum[y][x]
            s += lap
            sq += lap * lap
            n += 1
    mean = s / n
    return sq / n - mean * mean


def test_windows_of():
    assert windows_of(10, 3).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert windows_of(5, 5).tolist() == [0, 1, 2, 3, 4]
    w = windows_of(480, 160)
    assert (np.bincount(w) == 3).all()


def test_sharpness_matches_loop():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(13, 17, 3), dtype=np.uint8)
    assert sharpness(img) == pytest.approx(ref_sharpness(img), rel=1e-12)
    assert sharpness(np.zeros((2, 10, 3), np.uint8)) == 0
    assert sharpness(np.full((8, 8, 3), 0.5, np.float32)) == 0


def _stack(n, h=24, w=32, sharp=()):
    """Flat gray frames; frames in `sharp` get a checkerboard (high Laplacian)."""
    f = np.full((n, h, w, 3), 0.5, np.float32)
    yy, xx = np.mgrid[:h, :w]
    for i, amp in dict(sharp).items():
        f[i] = (0.5 + amp * ((yy + xx) % 2 - 0.5))[..., None]
    return f


def test_video_one_candidate_per_frame_and_times():
    # 48 frames at 24 fps, 8 targets x oversample 3 = 12 fps < 24: 24 candidates
    frames = _stack(48, sharp={5: 0.4, 30: 0.4})
    picks = pick_frames(frames, 24.0, 8, max_side=1600, oversample=3)
    assert len(picks) == 8
    # candidate k -> t = k/12 + 1/48 -> frame floor(2k + 0.5) = 2k
    idx = [p["index"] for p in picks]
    assert all(i % 2 == 0 for i in idx)
    # ties: last of each window of 3 candidates (frames 0,2,4 -> 4)
    assert idx[0] == 4 and picks[0]["time"] == pytest.approx(2 / 12)


def test_video_sharpest_wins_and_seek_rounding():
    frames = _stack(30, sharp={4: 0.4})
    # 24 fps source, sampleFps = 24: candidate k lands exactly on frame k
    picks = pick_frames(frames, 24.0, 10, max_side=1600, oversample=10)
    total = round(30 / 24 * 24)
    assert total == 30
    assert [p["index"] for p in picks][1] == 4  # window 1 = candidates 3..5
    assert picks[1]["time"] == pytest.approx(4 / 24)


def test_video_trim():
    frames = _stack(240)  # 10 s at 24 fps
    out, times = select_frames(frames, 24.0, 10, 1600, oversample=1, trim_start=2.0, trim_end=4.0)
    assert out.shape[0] == 10 and not times.from_images
    # sampleFps = 5: times 2.0, 2.2, ...
    np.testing.assert_allclose(times.times, [2.0 + k / 5 for k in range(10)])
    with pytest.raises(ValueError):
        select_frames(frames, 24.0, 10, 1600, trim_start=5.0, trim_end=5.05)


def test_images_and_resize():
    frames = _stack(9, h=300, w=400, sharp={1: 0.2, 4: 0.4, 7: 0.3})
    out, times = select_frames(frames, None, 3, max_side=200)
    assert times.from_images and times.times == [1.0, 4.0, 7.0]
    assert out.shape == (3, 150, 200, 3) and out.dtype == np.float32
    # no upscaling
    out, _ = select_frames(frames, None, 3, max_side=4000)
    assert out.shape[1:3] == (300, 400)


def test_video_max_side_rounding():
    frames = _stack(48, h=101, w=333)
    out, _ = select_frames(frames, 24.0, 4, max_side=100)
    assert out.shape[1:3] == (math.floor(101 * 100 / 333 + 0.5), 100)


def test_filter_picks_blurry_and_duplicates():
    from gausify.frames import filter_picks

    rng = np.random.default_rng(0)
    frames = rng.random((6, 20, 20, 3), dtype=np.float32)
    frames[3] = frames[2]
    picks = [{"index": i, "time": float(i), "sharpness": s} for i, s in enumerate([10, 10, 10, 10, 2, 10])]
    kept, d = filter_picks(frames, picks)
    assert kept == picks and d == {"duplicates": 0, "blurry": 0}
    kept, d = filter_picks(frames, picks, skip_duplicates=True, min_sharpness=0.5)
    assert [p["index"] for p in kept] == [0, 1, 2, 5] and d == {"duplicates": 1, "blurry": 1}


def test_select_frames_reports_indices():
    from gausify.frames import select_frames, select_masks

    frames = np.random.default_rng(2).random((12, 20, 30, 3), dtype=np.float32)
    out, t = select_frames(frames, None, 6, 15, 1)
    assert len(t.indices) == len(out) == 6 and out.shape[1:3] == (10, 15)
    m = select_masks(np.ones((12, 20, 30), np.float32), t.indices, 15, 10)
    assert m.shape == (6, 10, 15) and m.min() == 1.0
