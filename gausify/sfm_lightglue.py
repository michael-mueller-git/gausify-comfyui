"""The "aliked_lightglue" pose estimator (beyond the app): learned features
instead of AKAZE. RaCo keypoints + ALIKED descriptors, matched by LightGlue,
from the ONNX pipelines of LightGlue-ONNX (fabio-sim, Apache-2.0; RaCo
Apache-2.0, ALIKED BSD-3-Clause, LightGlue Apache-2.0). The reconstruction is
the "akaze" estimator's (incremental SfM, closed-loop handling, fill_missing).

The ONNX model is a pair pipeline: (2B, 3, H, W) interleaved [L0, R0, L1, R1,
...] -> keypoints (2B, K, 2) in input pixels (pixel centres at +0.5, the
COLMAP convention), matches (M, 3) [pair, i, j] and scores (M,). An image's
keypoints are recomputed for every pair it is in; the first set is kept and
later ones are mapped onto it by position (they agree to ~1e-5 px; the mapping
guards against reordered near-equal scores on the GPU).
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .sfm_akaze.features import ImageFeatures, _read_mask, _read_rgb
from .sfm_akaze.reconstruct import reconstruct
from .types import SfmModel

_log = logging.getLogger("gausify.sfm_lightglue")

#: Keypoints per image of the published models (models.json: aliked-lightglue-k<K>).
KEYPOINTS = (1024, 2048, 3072, 3584)
#: Pairs per model run.
PAIR_BATCH = 4
#: Resized frames kept in memory (others are read again from disk).
CACHE_IMAGES = 48
#: Matches below this LightGlue confidence are dropped (the model itself keeps >= 0.1).
MIN_SCORE = 0.1
#: Largest distance (input pixels) for mapping a recomputed keypoint onto the first set.
REMAP_TOLERANCE = 0.5

_SESSIONS: dict[str, tuple] = {}


def model_keypoints(max_features: int) -> int:
    """Keypoints of the model used for `max_features`: the largest K <= it (at least the smallest)."""
    fit = [k for k in KEYPOINTS if k <= int(max_features)]
    return fit[-1] if fit else KEYPOINTS[0]


def model_name(max_features: int) -> str:
    return f"aliked-lightglue-k{model_keypoints(max_features)}"


def infer_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """(w, h) the model runs at: scaled to fit `max_side` (no upscaling), each
    side rounded to a multiple of 32 (the network's stride)."""
    s = min(1.0, float(max_side) / max(width, height))
    return (max(32, int(round(width * s / 32)) * 32), max(32, int(round(height * s / 32)) * 32))


def _session(max_features: int, progress=None):
    from .downloader import ensure_model
    from .masking.onnx import session

    name = model_name(max_features)
    if name not in _SESSIONS:
        path = ensure_model(name, progress)
        _SESSIONS[name] = session(path, "auto", f"ALIKED-LightGlue k{model_keypoints(max_features)}")
    return _SESSIONS[name]


def _remap(first: np.ndarray, again: np.ndarray) -> np.ndarray | None:
    """Index into `first` of each keypoint of `again` (-1: none within
    REMAP_TOLERANCE), or None when both are the same list."""
    if first.shape == again.shape and np.abs(first - again).max(initial=0.0) <= 1e-3:
        return None
    from scipy.spatial import cKDTree

    d, i = cKDTree(first).query(again, distance_upper_bound=REMAP_TOLERANCE)
    return np.where(np.isfinite(d), i, -1).astype(np.int64)


def lightglue_poses(
    images_dir: Path,
    names: list[str],
    masks_dir: Path | None,
    pairs: list[tuple[int, int]],
    max_side: int = 1600,
    max_features: int = 4000,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    closed_loop: bool = True,
    log: Callable[[str], None] | None = None,
    session=None,
    batch: int = PAIR_BATCH,
    min_score: float = MIN_SCORE,
) -> tuple[SfmModel, list[dict]]:
    """As akaze_poses: (model, matches [{"a", "b", "idx": (M,2) int32}]).
    `session`: an InferenceSession (or (session, label)) to use instead of the
    downloaded model (tests). With masks, only keypoints on the subject are kept."""
    log = log or _log.info
    images_dir = Path(images_dir)
    prog = progress or (lambda _f, _m: None)

    def check():
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")

    n = len(names)
    t0 = time.perf_counter()
    if session is None:
        prog(0.0, "loading ALIKED-LightGlue")
        session = _session(max_features, lambda f, m: prog(0.0, m))
    sess, label = session if isinstance(session, tuple) else (session, "ALIKED-LightGlue")
    inp = sess.get_inputs()[0]
    dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

    sizes: dict[int, tuple[int, int]] = {}
    cache: OrderedDict[int, np.ndarray] = OrderedDict()
    first: dict[int, np.ndarray] = {}  # keypoints in model pixels, as first computed
    colors: dict[int, np.ndarray] = {}
    run_size: tuple[int, int] | None = None

    def load(i: int) -> np.ndarray:
        nonlocal run_size
        if i in cache:
            cache.move_to_end(i)
            return cache[i]
        rgb = _read_rgb(images_dir / names[i])
        H, W = rgb.shape[:2]
        sizes[i] = (W, H)
        if run_size is None:
            run_size = infer_size(W, H, max_side)
        w, h = run_size
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if w < W else cv2.INTER_LINEAR)
        x = (small.astype(np.float32) / 255.0).transpose(2, 0, 1).astype(dtype)
        cache[i] = x
        if i not in colors:
            colors[i] = rgb  # full resolution, until the keypoints are known
        while len(cache) > CACHE_IMAGES:
            cache.popitem(last=False)
        return x

    raw: list[np.ndarray] = []  # per pair: (M, 2) indices into first[a], first[b]
    pairs = [(int(a), int(b)) for a, b in pairs]
    prog(0.02, "ALIKED-LightGlue matching")
    done = 0
    for s in range(0, len(pairs), max(1, int(batch))):
        check()
        chunk = pairs[s : s + max(1, int(batch))]
        x = np.stack([load(i) for ab in chunk for i in ab])
        kps, matches, scores = sess.run(None, {inp.name: x})
        kps = np.asarray(kps, dtype=np.float32)
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 3)
        matches = matches[np.asarray(scores, dtype=np.float32).reshape(-1) >= float(min_score)]
        for p, (a, b) in enumerate(chunk):
            m = matches[matches[:, 0] == p][:, 1:3]
            for col, i in ((0, a), (1, b)):
                k = kps[2 * p + col]
                if i not in first:
                    first[i] = k
                    W, H = sizes[i]
                    w, h = run_size
                    xy = k.astype(np.float64) * (W / w, H / h)
                    px = np.clip(np.floor(xy[:, 0]), 0, W - 1).astype(np.int64)
                    py = np.clip(np.floor(xy[:, 1]), 0, H - 1).astype(np.int64)
                    colors[i] = colors[i][py, px].astype(np.uint8)
                else:
                    mp = _remap(first[i], k)
                    if mp is not None:
                        m = m.copy()
                        m[:, col] = mp[m[:, col]]
            raw.append(m[(m >= 0).all(1)])
        done += len(chunk)
        prog(0.02 + 0.48 * done / max(len(pairs), 1), f"ALIKED-LightGlue {done}/{len(pairs)} pairs")
    t_match = time.perf_counter() - t0
    check()

    # Images in no pair (n < 2 cannot happen here) still need features.
    for i in range(n):
        if i not in first:
            load(i)
            first[i] = np.zeros((0, 2), np.float32)
            colors[i] = np.zeros((0, 3), np.uint8)

    # Full-resolution keypoints, only on the subject with masks; reindex.
    feats: list[ImageFeatures] = []
    remap: list[np.ndarray] = []
    w, h = run_size
    for i in range(n):
        W, H = sizes[i]
        xy = first[i].astype(np.float64) * (W / w, H / h)
        keep = np.ones(len(xy), bool)
        if masks_dir is not None and len(xy):
            mpx = _read_mask(Path(masks_dir) / (Path(names[i]).stem + ".png"), W, H)
            mx = np.clip(np.floor(xy[:, 0]), 0, W - 1).astype(np.int64)
            my = np.clip(np.floor(xy[:, 1]), 0, H - 1).astype(np.int64)
            keep = mpx[my, mx] > 0
        new = np.full(len(xy), -1, np.int64)
        new[keep] = np.arange(int(keep.sum()))
        remap.append(new)
        feats.append(ImageFeatures(name=names[i], width=W, height=H, n=int(keep.sum()),
                                   xy=xy[keep].astype(np.float32), rgb=colors[i][keep].reshape(-1, 3)))
    out = []
    for (a, b), m in zip(pairs, raw):
        ia, ib = remap[a][m[:, 0]], remap[b][m[:, 1]]
        ok = (ia >= 0) & (ib >= 0)
        out.append({"a": a, "b": b, "idx": np.stack([ia[ok], ib[ok]], 1).astype(np.int32).reshape(-1, 2)})
    mean_feat = sum(f.n for f in feats) / max(n, 1)
    mean_m = sum(len(m["idx"]) for m in out) / max(len(out), 1)
    log(f"{label} at {w}x{h}: mean {mean_feat:.0f} keypoints/image, {len(pairs)} pairs, mean {mean_m:.0f} "
        f"matches in {t_match:.1f} s")

    t2 = time.perf_counter()
    model = reconstruct(feats, out, log=log, progress=lambda fr, msg: prog(0.5 + 0.5 * fr, msg),
                        closed_loop=closed_loop, cancel=cancel)
    model.names = list(names)
    model.xy = [f.xy for f in feats]
    model.stats["mapper"] = "aliked_lightglue"
    log(
        f"Reconstruction: {model.stats['registered']}/{n} cameras, {model.stats['points']} points, "
        f"mean error {float(model.stats['meanError']):.2f} px, f = {model.f:.1f} in {time.perf_counter() - t2:.1f} s"
    )
    return model, out
