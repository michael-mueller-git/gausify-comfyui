"""The aliked_lightglue estimator: keypoint bookkeeping with a fake ONNX
session, and (slow, opt-in) the real model on the e2e frames."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from gausify import sfm_lightglue as lg
from gausify.pairs import orbit_pairs

GRID = np.array([[x, y] for y in (4.5, 12.5, 20.5) for x in (4.5, 20.5, 36.5, 52.5)], np.float32)


class _Input:
    name = "images"
    type = "tensor(float)"


class FakeSession:
    """Image i (encoded in its pixel value) has the GRID keypoints; every
    other run returns them reversed, as a GPU might reorder near-equal ones.
    Each pair matches every keypoint to the same grid point of the other image."""

    def __init__(self):
        self.calls = 0
        self.shapes = []

    def get_inputs(self):
        return [_Input()]

    def run(self, _out, feeds):
        x = feeds["images"]
        self.shapes.append(x.shape)
        self.calls += 1
        rev = self.calls % 2 == 0
        k = GRID[::-1] if rev else GRID
        kps = np.stack([k] * x.shape[0])
        rows = []
        for p in range(x.shape[0] // 2):
            for j in range(len(GRID)):
                rows.append([p, j, j])
        return kps, np.array(rows, np.int64), np.ones(len(rows), np.float32)


def _frames(tmp_path: Path, n: int, w=128, h=48):
    d = tmp_path / "images"
    d.mkdir()
    names = []
    for i in range(n):
        img = np.full((h, w, 3), i, np.uint8)
        names.append(f"{i:05d}.png")
        cv2.imwrite(str(d / names[-1]), img)
    return d, names


def test_model_keypoints_and_size():
    assert lg.model_keypoints(4000) == 3584 and lg.model_keypoints(3584) == 3584
    assert lg.model_keypoints(2500) == 2048 and lg.model_keypoints(500) == 1024
    assert lg.model_name(3000) == "aliked-lightglue-k2048"
    assert lg.infer_size(1920, 1080, 1600) == (1600, 896)
    assert lg.infer_size(480, 360, 1600) == (480, 352)
    w, h = lg.infer_size(1080, 1920, 1024)
    assert h == 1024 and w % 32 == 0 and h % 32 == 0


def test_remap():
    assert lg._remap(GRID, GRID.copy()) is None
    m = lg._remap(GRID, GRID[::-1] + 0.1)
    assert (GRID[m] == GRID[::-1]).all()
    far = GRID.copy()
    far[0] += 5
    assert lg._remap(GRID, far)[0] == -1


def _run(monkeypatch, tmp_path, masks_dir=None, n=6, batch=2):
    images, names = _frames(tmp_path, n)
    seen = {}

    def reconstruct(feats, matches, log=None, progress=None, closed_loop=True, cancel=None):
        from gausify.types import SfmModel

        seen["feats"], seen["matches"] = feats, matches
        return SfmModel(f=50.0, cx=64, cy=24, width=128, height=48, poses=[None] * n, points=[], names=[],
                        stats={"registered": 0, "points": 0, "meanError": 0.0})

    monkeypatch.setattr(lg, "reconstruct", reconstruct)
    sess = FakeSession()
    pairs = orbit_pairs(n, 2, True)
    model, matches = lg.lightglue_poses(images, names, masks_dir, pairs, max_side=64, session=sess, batch=batch)
    return sess, pairs, model, matches, seen


def test_lightglue_poses_bookkeeping(monkeypatch, tmp_path):
    sess, pairs, model, matches, seen = _run(monkeypatch, tmp_path)
    # 128x48 -> fits 64 -> 64x32 (multiples of 32); interleaved pairs, batch 2
    assert all(s[1:] == (3, 32, 64) and s[0] in (2, 4) for s in sess.shapes)
    assert sess.calls == -(-len(pairs) // 2)
    feats = seen["feats"]
    assert model.stats["mapper"] == "aliked_lightglue" and len(model.xy) == 6
    # full resolution: x * 128/64, y * 48/32
    np.testing.assert_allclose(feats[0].xy, GRID * (2.0, 1.5), rtol=1e-6)
    assert [(m["a"], m["b"]) for m in matches] == pairs
    for m in matches:
        ia, ib = m["idx"][:, 0], m["idx"][:, 1]
        assert len(ia) == len(GRID)
        # despite the reversed runs, matched indices are the same grid point
        np.testing.assert_allclose(feats[m["a"]].xy[ia], feats[m["b"]].xy[ib])
    # colours sampled from the full-resolution frame (value = image index)
    assert (feats[3].rgb == 3).all() and feats[3].rgb.shape == (len(GRID), 3)


def test_lightglue_poses_masks(monkeypatch, tmp_path):
    md = tmp_path / "masks"
    md.mkdir()
    for i in range(6):
        m = np.zeros((48, 128), np.uint8)
        m[:, :64] = 255  # left half: grid x 4.5, 20.5 (-> 9, 41 full res)
        cv2.imwrite(str(md / f"{i:05d}.png"), m)
    _, _, _, matches, seen = _run(monkeypatch, tmp_path, masks_dir=md)
    feats = seen["feats"]
    assert all(f.n == 6 and (f.xy[:, 0] < 64).all() for f in feats)
    for m in matches:
        assert len(m["idx"]) == 6 and m["idx"].max() < 6
        np.testing.assert_allclose(feats[m["a"]].xy[m["idx"][:, 0]], feats[m["b"]].xy[m["idx"][:, 1]])


def test_run_sfm_dispatches(monkeypatch, tmp_path):
    import gausify.sfm as sfm

    got = {}

    def fake(images_dir, names, masks_dir, pairs, max_side, max_features, **kw):
        from gausify.types import SfmModel

        got.update(max_side=max_side, max_features=max_features, closed=kw["closed_loop"])
        n = len(names)
        return SfmModel(f=1.0, cx=0, cy=0, width=2, height=2, poses=[None] * n, points=[], names=list(names),
                        xy=[np.zeros((0, 2), np.float32)] * n, stats={"registered": 0}), []

    monkeypatch.setattr(lg, "lightglue_poses", fake)
    sfm.run_sfm(tmp_path, ["a.png", "b.png", "c.png"], estimator="aliked_lightglue", max_side=1024,
                max_features=2048)
    assert got == {"max_side": 1024, "max_features": 2048, "closed": True}


E2E = Path(__file__).resolve().parents[1] / ".dev" / "e2e" / "src"


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("GAUSIFY_MODEL_ALIKED_LIGHTGLUE_K2048") or not E2E.is_dir(),
                    reason="set GAUSIFY_MODEL_ALIKED_LIGHTGLUE_K2048 to the ONNX file (and have .dev/e2e/src)")
def test_real_model_poses():
    from gausify.sfm import run_sfm

    names = sorted(p.name for p in E2E.glob("*.png"))[:24]
    model = run_sfm(E2E, names, estimator="aliked_lightglue", window=4, closed_loop=False,
                    max_side=480, max_features=2048)
    assert model.stats["registered"] >= len(names) * 0.8
    assert float(model.stats["meanError"]) < 2.0


class FakeExtractor:
    """Split mode: image i -> GRID keypoints, normalised keypoints and
    descriptors that encode the grid index (so the fake matcher can match)."""

    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [_Input()]

    def run(self, _out, feeds):
        x = feeds["images"]
        self.calls.append(x.shape[0])
        b = x.shape[0]
        kps = np.stack([GRID] * b)
        desc = np.zeros((b, len(GRID), 4), np.float32)
        desc[:, :, 0] = np.arange(len(GRID))
        return kps, np.stack([GRID / 64.0] * b), desc


class FakeMatcher:
    def __init__(self):
        self.calls = 0

    def run(self, _out, feeds):
        d = feeds[lg.DESC]
        assert feeds[lg.NORM].shape[:2] == d.shape[:2] and d.shape[0] % 2 == 0
        self.calls += 1
        rows = [[p, j, int(d[2 * p + 1, j, 0])] for p in range(d.shape[0] // 2) for j in range(d.shape[1])]
        scores = np.ones(len(rows), np.float32)
        scores[::3] = 0.05  # below MIN_SCORE: dropped
        return np.array(rows, np.int64), scores


def test_lightglue_poses_split(monkeypatch, tmp_path):
    images, names = _frames(tmp_path, 10)
    seen = {}

    def reconstruct(feats, matches, **kw):
        from gausify.types import SfmModel

        seen["feats"] = feats
        return SfmModel(f=50.0, cx=64, cy=24, width=128, height=48, poses=[None] * 10, points=[], names=[],
                        stats={"registered": 0, "points": 0, "meanError": 0.0})

    monkeypatch.setattr(lg, "reconstruct", reconstruct)
    ext, mat = FakeExtractor(), FakeMatcher()
    pairs = orbit_pairs(10, 3, True)
    _, matches = lg.lightglue_poses(images, names, None, pairs, max_side=64, split=(ext, mat), batch=4)
    assert sum(ext.calls) == 10 and ext.calls == [8, 2]  # every frame extracted once
    assert mat.calls == -(-len(pairs) // 4)
    feats = seen["feats"]
    np.testing.assert_allclose(feats[0].xy, GRID * (2.0, 1.5), rtol=1e-6)
    for m in matches:
        assert len(m["idx"]) == len(GRID) - 4  # 4 of 12 below the score threshold
        np.testing.assert_allclose(feats[m["a"]].xy[m["idx"][:, 0]], feats[m["b"]].xy[m["idx"][:, 1]])


def test_sessions_fall_back_to_pipeline(monkeypatch, tmp_path):
    import gausify.downloader
    import gausify.masking.onnx

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(gausify.downloader, "ensure_model", lambda name, progress=None: model)
    monkeypatch.setattr(gausify.masking.onnx, "session", lambda p, device, label: (str(p), label))
    monkeypatch.setattr(lg, "_SESSIONS", {})

    def broken(path):
        raise ImportError("No module named 'onnx'")

    monkeypatch.setattr(lg, "split_model", broken)
    warned = []
    s = lg._sessions(4000, "auto", log=warned.append)
    assert s["mode"] == "pipeline" and s["pipeline"][0] == str(model) and "pip install onnx" in warned[0]
    with pytest.raises(ImportError):
        lg._sessions(2048, "split")
    monkeypatch.setattr(lg, "split_model", lambda p: (tmp_path / "e.onnx", tmp_path / "x.onnx"))
    s = lg._sessions(2048, "auto")
    assert s["mode"] == "split" and s["extractor"][0].endswith("e.onnx") and s["matcher"][0].endswith("x.onnx")
    assert lg._sessions(2048, "auto") is s  # cached


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("GAUSIFY_MODEL_ALIKED_LIGHTGLUE_K2048") or not E2E.is_dir(),
                    reason="set GAUSIFY_MODEL_ALIKED_LIGHTGLUE_K2048 to the ONNX file (and have .dev/e2e/src)")
def test_real_split_matches_pipeline(tmp_path):
    import shutil

    src = Path(os.environ["GAUSIFY_MODEL_ALIKED_LIGHTGLUE_K2048"])
    model = tmp_path / src.name
    shutil.copyfile(src, model)
    ext_p, mat_p = lg.split_model(model)
    from gausify.masking.onnx import session

    names = sorted(p.name for p in E2E.glob("*.png"))[:6]
    pairs = orbit_pairs(6, 2, False)
    kw = dict(max_side=320, closed_loop=False)
    ma, a = lg.lightglue_poses(E2E, names, None, pairs, split=(session(ext_p, "cpu"), session(mat_p, "cpu")), **kw)
    mb, b = lg.lightglue_poses(E2E, names, None, pairs, session=session(model, "cpu"), **kw)

    def matched(model, ms):  # keypoint order may differ (tied scores): compare positions
        return [{tuple(np.round(np.r_[model.xy[m["a"]][i], model.xy[m["b"]][j]], 2)) for i, j in m["idx"]} for m in ms]

    assert matched(ma, a) == matched(mb, b) and sum(len(x["idx"]) for x in a) > 100
