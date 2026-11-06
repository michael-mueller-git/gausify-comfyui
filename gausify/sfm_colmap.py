"""The "colmap" pose estimator, on pycolmap in an isolated venv.

Port of gausify-orbit web/src/sfm/colmap-sfm.ts and the colmap-wasm package it
uses (github.com/obraia/colmap-wasm: native/entry/web.cc, src/worker.ts,
src/estimate.ts, src/sift-matcher.ts). colmap-wasm runs:

- SIFT with COLMAP's defaults on the CPU (FeatureExtractionOptions(SIFT):
  max_num_features 8192, first_octave -1, num_octaves 4, octave_resolution 3,
  peak_threshold 0.02/3, edge_threshold 10, max_num_orientations 2,
  L1_ROOT normalization; images larger than 3200 px are reduced and the
  keypoints scaled back), keypoints on mask pixels == 0 dropped (as COLMAP's
  ImageReader.mask_path);
- exact brute-force SIFT matching with COLMAP's defaults: max_ratio 0.8,
  max_distance 0.7, cross_check on, only the given pairs;
- one shared SIMPLE_PINHOLE camera (focal 1.2 x max side, the
  `feature_extractor` default);
- `geometric_verifier` (TwoViewGeometryOptions defaults, RANSAC seed),
  `view_graph_calibrator` (seeded) and `global_mapper` (GlobalPipelineOptions
  defaults), all with seed 0.

Here the same COLMAP version (4.2.1) does the same steps natively, but not in
this process: pycolmap-cuda12 needs CUDA 12 libraries that conflict with
ComfyUI's torch (CUDA 13), so `gausify/colmap_worker.py` runs it in a separate
venv (`colmap_env.py`). This module writes the job, spawns the worker, relays
its progress, and converts the result (read from an .npz) into an SfmModel,
sampling the point colours with OpenCV. The raw matches (before verification)
are returned, as colmapPoses does, for the orbit corrections
(gausify.orbit.fill_missing).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import colmap_env
from .types import Point3D, Pose, SfmModel

log = logging.getLogger("gausify.sfm_colmap")

#: Fixed seed: the same frames give the same cameras (colmap-sfm.ts SEED).
SEED = 0
#: colmap-wasm: one shared SIMPLE_PINHOLE camera.
CAMERA_MODEL = "SIMPLE_PINHOLE"
#: Images extracted per pycolmap call (progress granularity).
EXTRACT_CHUNK = 16
WORKER = Path(__file__).with_name("colmap_worker.py")
#: Seconds the worker gets to exit after terminate() before it is killed.
KILL_GRACE = 5.0

Progress = Callable[[float, str], None] | None
Cancel = Callable[[], bool] | None


class Cancelled(InterruptedError):
    """Raised when `cancel()` returned True; the worker process was stopped."""


class ColmapError(RuntimeError):
    pass


def write_masks(masks_dir: Path, names: Sequence[str], out_dir: Path) -> Path:
    """COLMAP's mask convention: `<mask_path>/<image name>.png` (0 = ignore).
    Our masks are `<stem>.png`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = Path(masks_dir) / (Path(n).stem + ".png")
        dst = out_dir / (n + ".png")
        if not src.is_file():
            raise FileNotFoundError(f"mask missing: {src}")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    return out_dir


def write_pairs(names: Sequence[str], pairs: Sequence[tuple[int, int]], path: Path) -> Path:
    path.write_text("".join(f"{names[a]} {names[b]}\n" for a, b in pairs))
    return path


def _rgb_u8(images_dir: Path, name: str) -> np.ndarray:
    import cv2

    im = cv2.imread(str(Path(images_dir) / name), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(Path(images_dir) / name)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_model(res, names, images_dir, width, height, mapper="colmap") -> SfmModel:
    """Worker result (dict of arrays, see colmap_worker.save_result) -> SfmModel,
    like toModel in colmap-sfm.ts (point colors from the image at the first
    track element's keypoint, as colmap-wasm)."""
    n = len(names)
    poses: list[Pose | None] = [None] * n
    has_rec = bool(res["has_rec"])
    if has_rec:
        f, cx, cy = (float(v) for v in res["camera"])
        for i in range(n):
            M = res["poses"][i]
            if not np.isnan(M).any():
                poses[i] = Pose(R=M[:, :3].copy(), t=M[:, 3].copy())
    else:
        f, cx, cy = 1.2 * max(width, height), width / 2, height / 2

    off = np.concatenate([[0], np.cumsum(res["xy_counts"])])
    xy = [np.ascontiguousarray(res["xy_flat"][off[i] : off[i + 1]], np.float32) for i in range(n)]

    cache: dict[int, np.ndarray] = {}
    points: list[Point3D] = []
    obs_total, err_sum = 0, 0.0
    t_off = np.concatenate([[0], np.cumsum(res["track_counts"])])
    for j in range(len(res["track_counts"])):
        sl = slice(t_off[j], t_off[j + 1])
        track = [(int(a), int(b)) for a, b in zip(res["track_img"][sl], res["track_kp"][sl])]
        i0, k0 = track[0]
        if i0 not in cache:
            cache[i0] = _rgb_u8(images_dir, names[i0])
        img = cache[i0]
        H, W = img.shape[:2]
        x = min(W - 1, max(0, int(np.floor(xy[i0][k0, 0]))))
        y = min(H - 1, max(0, int(np.floor(xy[i0][k0, 1]))))
        rgb = tuple(int(c) for c in img[y, x])
        err = float(res["error"][j])
        obs_total += len(track)
        err_sum += err * len(track)
        points.append(Point3D(X=np.asarray(res["xyz"][j], np.float64).copy(), rgb=rgb, obs=track, error=err))
    registered = sum(1 for q in poses if q is not None)
    stats = {
        "registered": registered,
        "total": n,
        "points": len(points),
        "observations": obs_total,
        "meanError": err_sum / obs_total if obs_total else 0.0,
        "focal": f,
        "mapper": "colmap",
    }
    if mapper != "colmap":
        stats["fallback"] = mapper
    return SfmModel(f=f, cx=cx, cy=cy, width=width, height=height, poses=poses, points=points,
                    names=list(names), xy=xy, stats=stats)


def _matches(res) -> list[dict]:
    off = np.concatenate([[0], np.cumsum(res["match_counts"])])
    return [
        {"a": int(a), "b": int(b), "idx": np.ascontiguousarray(res["match_idx"][off[k] : off[k + 1]], np.int32)}
        for k, (a, b) in enumerate(res["match_ab"])
    ]


def run_worker(
    job_path: Path, progress: Progress = None, cancel: Cancel = None, log_fn: Callable[[str], None] | None = None,
    python: str | os.PathLike | None = None,
) -> dict:
    """Run colmap_worker.py on `job_path` with the venv python; returns the
    worker's `result` event. Progress events go to `progress`, `log` events to
    `log_fn`; `cancel()` is polled and terminates the process (Cancelled)."""
    python = python or colmap_env.ensure_colmap_env(progress)
    cmd = [str(python), "-I", str(WORKER), str(job_path)]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=60)
    lines: queue.Queue[str | None] = queue.Queue()

    def pump_out():
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def pump_err():
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line.rstrip("\n"))

    threads = [threading.Thread(target=pump_out, daemon=True), threading.Thread(target=pump_err, daemon=True)]
    for t in threads:
        t.start()
    result: dict | None = None
    error: dict | None = None

    def stop():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(KILL_GRACE)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    try:
        while True:
            if cancel is not None and cancel():
                stop()
                raise Cancelled("COLMAP cancelled")
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                kind = ev["type"]
            except (ValueError, KeyError, TypeError):
                tail.append(line)
                continue
            if kind == "progress":
                if progress:
                    progress(float(ev["fraction"]), str(ev.get("message", "")))
            elif kind == "log":
                (log_fn or log.info)(str(ev.get("message", "")))
            elif kind == "result":
                result = ev
            elif kind == "error":
                error = ev
        rc = proc.wait()
    except BaseException:
        stop()
        raise
    for t in threads:
        t.join(timeout=5)
    if rc != 0 or result is None:
        msg = f"COLMAP worker failed (exit code {rc})"
        if error:
            msg += f": {error.get('message')}"
            if error.get("traceback"):
                msg += "\n" + str(error["traceback"]).rstrip()
        if tail:
            msg += "\n" + "\n".join(tail)
        raise ColmapError(msg)
    return result


# ---------------------------------------------------------------- entry point


def colmap_poses_with_matches(
    images_dir,
    names: Sequence[str],
    masks_dir=None,
    pairs: Sequence[tuple[int, int]] = (),
    progress: Progress = None,
    cancel: Cancel = None,
    log: Callable[[str], None] | None = None,
    work_dir=None,
) -> tuple[SfmModel, list[dict]]:
    """colmap_poses plus the raw matches ({"a","b","idx": (M,2) int32})."""
    say = log or (lambda s: None)
    names = list(names)
    images_dir = Path(images_dir).resolve()
    if len(names) < 2:
        raise ValueError("need at least 2 images")
    probe = _rgb_u8(images_dir, names[0])
    height, width = probe.shape[:2]
    python = colmap_env.ensure_colmap_env(progress)
    tmp = tempfile.TemporaryDirectory(prefix="gausify-colmap-", dir=work_dir)
    work = Path(tmp.name)
    try:
        masks = write_masks(Path(masks_dir), names, work / "masks") if masks_dir else None
        pairs_file = write_pairs(names, pairs, work / "pairs.txt")
        job = {
            "images_dir": str(images_dir),
            "names": names,
            "masks_dir": str(masks) if masks else None,
            "pairs_file": str(pairs_file),
            "n_pairs": len(pairs),
            "database": str(work / "database.db"),
            "work_dir": str(work),
            "result": str(work / "result.npz"),
            "seed": SEED,
            "camera_model": CAMERA_MODEL,
            "extract_chunk": EXTRACT_CHUNK,
        }
        job_path = work / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        t0 = time.perf_counter()
        res = run_worker(job_path, progress, cancel, say, python=python)
        with np.load(res["result"]) as z:
            arrays = {k: z[k] for k in z.files}
        mapper = res.get("mapper", "colmap")
        model = to_model(arrays, names, images_dir, width, height, mapper)
        model.stats["backend"] = res.get("backend")
        matches = _matches(arrays)
        s = model.stats
        say(
            f"COLMAP {'global' if mapper == 'colmap' else 'incremental'} mapper: {s['registered']}/{len(names)} cameras, "
            f"{s['points']} points, mean error {s['meanError']:.2f} px, f = {model.f:.1f} "
            f"in {res.get('seconds_mapping', time.perf_counter() - t0):.1f} s"
        )
        if progress:
            progress(1.0, "COLMAP done")
        return model, matches
    finally:
        tmp.cleanup()


def colmap_poses(images_dir, names, masks_dir=None, pairs=(), progress: Progress = None,
                 cancel: Cancel = None) -> SfmModel:
    """The "colmap" estimator. Raw matches are in `model.stats` only as counts;
    use colmap_poses_with_matches for them."""
    model, _ = colmap_poses_with_matches(images_dir, names, masks_dir, pairs, progress, cancel)
    return model
