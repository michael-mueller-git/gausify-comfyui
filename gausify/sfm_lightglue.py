"""The "aliked_lightglue" pose estimator (beyond the app): learned features
instead of AKAZE. RaCo keypoints + ALIKED descriptors, matched by LightGlue,
from the ONNX pipelines of LightGlue-ONNX (fabio-sim, Apache-2.0; RaCo
Apache-2.0, ALIKED BSD-3-Clause, LightGlue Apache-2.0). The reconstruction is
the "akaze" estimator's (incremental SfM, closed-loop handling, fill_missing).

The published ONNX model is a pair pipeline: (2B, 3, H, W) interleaved [L0,
R0, L1, R1, ...] -> keypoints (2B, K, 2) in input pixels (pixel centres at
+0.5, the COLMAP convention), matches (M, 3) [pair, i, j] and scores (M,).

Two ways to run it:

- "split" (default when the `onnx` package is installed): the pipeline is cut
  once, on first use, into its extractor (images -> keypoints, normalised
  keypoints, descriptors) and its LightGlue matcher (normalised keypoints,
  descriptors -> matches, scores), cached next to the model. Every frame is
  extracted once; only the matcher runs per pair. Same output as the
  pipeline; faster the more the extractor costs, i.e. at higher resolutions
  (a frame is in ~2 x match_window pairs; 1.5x at 480 px on CPU, where
  LightGlue dominates).
- "pipeline": the published model as is. An image's keypoints are recomputed
  for every pair it is in; the first set is kept and later ones are mapped
  onto it by position (they agree to ~1e-5 px, but near-equal scores can come
  back in another order).
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
#: Frames per extractor run (split mode).
EXTRACT_BATCH = 8
#: "auto" = split when possible, else pipeline.
MODES = ("auto", "split", "pipeline")
#: Interface tensors of the split models.
NORM, DESC = "norm_keypoints", "descriptors"

_SESSIONS: dict[tuple, dict] = {}


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


def _cut_tensors(graph) -> tuple[str, str]:
    """(normalised keypoints, descriptors): the (2B, K, 2) and (2B, K, D>2)
    tensors the matcher's first MatMuls take (its positional encoding and
    input projection). Their auto-generated names differ between the models."""
    shapes = {vi.name: [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]
              for vi in graph.value_info}
    kp_out = next(o for o in graph.output if o.name == "keypoints")
    k = kp_out.type.tensor_type.shape.dim[1].dim_value
    norm = desc = None
    for n in graph.node:
        if n.op_type != "MatMul":
            continue
        sh = shapes.get(n.input[0])
        if sh and len(sh) == 3 and sh[1] == k:
            if sh[2] == 2 and norm is None:
                norm = n.input[0]
            elif sh[2] and sh[2] > 2 and desc is None:
                desc = n.input[0]
    if norm is None or desc is None:
        raise ValueError("can't find the extractor/matcher boundary in this model")
    return norm, desc


def _rename(graph, old: str, new: str) -> None:
    for n in graph.node:
        for k, i in enumerate(n.input):
            if i == old:
                n.input[k] = new
        for k, o in enumerate(n.output):
            if o == old:
                n.output[k] = new
    for v in list(graph.input) + list(graph.output) + list(graph.value_info):
        if v.name == old:
            v.name = new


def split_model(path: Path) -> tuple[Path, Path]:
    """(extractor, matcher) ONNX files cut from the pipeline at `path`, made
    once and cached next to it (needs the `onnx` package)."""
    import onnx
    import onnx.utils

    from .downloader import file_lock

    path = Path(path)
    ext_p = path.with_name(path.stem + ".extractor.onnx")
    mat_p = path.with_name(path.stem + ".matcher.onnx")
    if ext_p.is_file() and mat_p.is_file():
        return ext_p, mat_p
    with file_lock(path.with_name(f".{path.stem}.split.lock")):
        if ext_p.is_file() and mat_p.is_file():
            return ext_p, mat_p
        m = onnx.load(str(path))
        norm, desc = _cut_tensors(m.graph)
        image = m.graph.input[0].name
        ex = onnx.utils.Extractor(m)
        parts = {
            ext_p: ex.extract_model([image], ["keypoints", norm, desc]),
            mat_p: ex.extract_model([norm, desc], ["matches", "mscores"]),
        }
        matcher = parts[mat_p].graph
        for node in matcher.node:  # the matcher only reads the batch size off the images
            for k, i in enumerate(node.input):
                if i == image:
                    if node.op_type != "Shape":
                        raise ValueError(f"the matcher uses the images beyond their shape ({node.op_type})")
                    node.input[k] = desc
        for part in parts.values():
            _rename(part.graph, norm, NORM)
            _rename(part.graph, desc, DESC)
            onnx.checker.check_model(part)
        for p, part in parts.items():
            tmp = p.with_name(p.name + ".part")
            onnx.save(part, str(tmp))
            tmp.replace(p)
    return ext_p, mat_p


def _sessions(max_features: int, mode: str = "auto", progress=None, log=None) -> dict:
    """{"mode": "split", "extractor": .., "matcher": ..} or {"mode":
    "pipeline", "pipeline": ..}, each a (session, label) pair; cached."""
    from .downloader import ensure_model
    from .masking.onnx import session

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    key = (model_name(max_features), mode)
    if key in _SESSIONS:
        return _SESSIONS[key]
    path = ensure_model(model_name(max_features), progress)
    label = f"ALIKED-LightGlue k{model_keypoints(max_features)}"
    out = None
    if mode in ("auto", "split"):
        try:
            ext_p, mat_p = split_model(path)
            out = {"mode": "split", "extractor": session(ext_p, "auto", label),
                   "matcher": session(mat_p, "auto", label + " matcher")}
        except Exception as e:  # noqa: BLE001 - fall back to the published pipeline
            if mode == "split":
                raise
            (log or _log.warning)(f"{label}: can't split the model ({e}); running the pair pipeline "
                                  "(slower; pip install onnx for the split)")
    if out is None:
        out = {"mode": "pipeline", "pipeline": session(path, "auto", label)}
    _SESSIONS[key] = out
    return out


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
    split=None,
    mode: str = "auto",
) -> tuple[SfmModel, list[dict]]:
    """As akaze_poses: (model, matches [{"a", "b", "idx": (M,2) int32}]).
    `session`: a pipeline InferenceSession (or (session, label)); `split`:
    (extractor, matcher) sessions; either replaces the downloaded model
    (tests). `mode`: see MODES. With masks, only keypoints on the subject are kept."""
    log = log or _log.info
    images_dir = Path(images_dir)
    prog = progress or (lambda _f, _m: None)

    def check():
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")

    n = len(names)
    t0 = time.perf_counter()
    if session is None and split is None:
        prog(0.0, "loading ALIKED-LightGlue")
        ss = _sessions(max_features, mode, lambda f, m: prog(0.0, m), log)
        if ss["mode"] == "split":
            split = (ss["extractor"], ss["matcher"])
        else:
            session = ss["pipeline"]

    def unpack(x, default):
        return x if isinstance(x, tuple) else (x, default)

    if split is not None:
        (sess, label), (msess, _) = unpack(split[0], "ALIKED-LightGlue"), unpack(split[1], "")
        label += " (split)"
    else:
        sess, label = unpack(session, "ALIKED-LightGlue")
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

    def first_sight(i: int, k: np.ndarray) -> None:
        first[i] = k
        W, H = sizes[i]
        w, h = run_size
        xy = k.astype(np.float64) * (W / w, H / h)
        px = np.clip(np.floor(xy[:, 0]), 0, W - 1).astype(np.int64)
        py = np.clip(np.floor(xy[:, 1]), 0, H - 1).astype(np.int64)
        colors[i] = colors[i][py, px].astype(np.uint8)

    def kept(matches, scores) -> np.ndarray:
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 3)
        return matches[np.asarray(scores, dtype=np.float32).reshape(-1) >= float(min_score)]

    raw: list[np.ndarray] = []  # per pair: (M, 2) indices into first[a], first[b]
    pairs = [(int(a), int(b)) for a, b in pairs]
    step = max(1, int(batch))
    if split is not None:
        # every frame once through the extractor ...
        norms: dict[int, np.ndarray] = {}
        descs: dict[int, np.ndarray] = {}
        for s0 in range(0, n, EXTRACT_BATCH):
            check()
            idx = list(range(s0, min(n, s0 + EXTRACT_BATCH)))
            kps, nk, ds = sess.run(None, {inp.name: np.stack([load(i) for i in idx])})
            for j, i in enumerate(idx):
                first_sight(i, np.asarray(kps[j], dtype=np.float32))
                norms[i], descs[i] = nk[j], ds[j]
                cache.pop(i, None)
            prog(0.02 + 0.18 * idx[-1] / max(n - 1, 1), f"ALIKED features {idx[-1] + 1}/{n}")
        # ... then only the matcher per pair
        for s0 in range(0, len(pairs), step):
            check()
            chunk = pairs[s0 : s0 + step]
            order = [i for ab in chunk for i in ab]
            matches, scores = msess.run(None, {NORM: np.stack([norms[i] for i in order]),
                                               DESC: np.stack([descs[i] for i in order])})
            matches = kept(matches, scores)
            for p in range(len(chunk)):
                raw.append(matches[matches[:, 0] == p][:, 1:3])
            done = s0 + len(chunk)
            prog(0.2 + 0.3 * done / max(len(pairs), 1), f"LightGlue {done}/{len(pairs)} pairs")
    else:
        prog(0.02, "ALIKED-LightGlue matching")
        done = 0
        for s0 in range(0, len(pairs), step):
            check()
            chunk = pairs[s0 : s0 + step]
            x = np.stack([load(i) for ab in chunk for i in ab])
            kps, matches, scores = sess.run(None, {inp.name: x})
            kps = np.asarray(kps, dtype=np.float32)
            matches = kept(matches, scores)
            for p, (a, b) in enumerate(chunk):
                m = matches[matches[:, 0] == p][:, 1:3]
                for col, i in ((0, a), (1, b)):
                    k = kps[2 * p + col]
                    if i not in first:
                        first_sight(i, k)
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
