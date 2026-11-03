"""The "colmap" pose estimator, on pycolmap (and the COLMAP CLI on Windows).

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

Here the same COLMAP version (4.2.1) does the same steps natively. The raw
matches (before verification) are returned, as colmapPoses does, for the
orbit corrections (gausify.orbit.fill_missing).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .types import Point3D, Pose, SfmModel

#: Fixed seed: the same frames give the same cameras (colmap-sfm.ts SEED).
SEED = 0
#: colmap-wasm: one shared SIMPLE_PINHOLE camera.
CAMERA_MODEL = "SIMPLE_PINHOLE"
#: Images extracted per pycolmap call (progress granularity).
EXTRACT_CHUNK = 16

Progress = Callable[[float, str], None] | None
Cancel = Callable[[], bool] | None


class Cancelled(Exception):
    pass


def _pycolmap():
    from .cuda_libs import import_pycolmap

    return import_pycolmap()


def use_gpu(pycolmap=None) -> bool:
    """CUDA build of pycolmap and at least one CUDA device."""
    pycolmap = pycolmap or _pycolmap()
    if not getattr(pycolmap, "has_cuda", False):
        return False
    try:
        return pycolmap.get_num_cuda_devices() > 0
    except Exception:
        return False


class _CancelWatch:
    """Turns `cancel()` polling into a pycolmap CancellationToken."""

    def __init__(self, pycolmap, cancel: Cancel):
        self.cancel = cancel
        self.token = pycolmap.CancellationToken() if cancel and hasattr(pycolmap, "CancellationToken") else None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.token is not None:
            def run():
                while not self._stop.wait(0.2):
                    if self.cancel():
                        self.token.cancel()
                        return

            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def check(self):
        if self.cancel and self.cancel():
            raise Cancelled("cancelled")


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


# ---------------------------------------------------------------- extraction / matching


def _reader_options(pycolmap, masks: Path | None, existing_camera_id: int = -1):
    o = pycolmap.ImageReaderOptions()
    o.camera_model = CAMERA_MODEL
    if masks is not None:
        o.mask_path = str(masks)
    if existing_camera_id >= 0:
        o.existing_camera_id = existing_camera_id
    return o


def _extract_pycolmap(pycolmap, db, images_dir, names, masks, gpu, watch, progress):
    ext = pycolmap.FeatureExtractionOptions()  # SIFT, COLMAP defaults
    ext.use_gpu = gpu
    device = pycolmap.Device.cuda if gpu else pycolmap.Device.cpu
    camera_id = -1
    for s in range(0, len(names), EXTRACT_CHUNK):
        watch.check()
        chunk = list(names[s : s + EXTRACT_CHUNK])
        kw = {"cancellation_token": watch.token} if watch.token is not None else {}
        pycolmap.extract_features(
            str(db),
            str(images_dir),
            image_names=chunk,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=_reader_options(pycolmap, masks, camera_id),
            extraction_options=ext,
            device=device,
            **kw,
        )
        if camera_id < 0:
            with pycolmap.Database.open(str(db)) as d:
                camera_id = d.read_all_cameras()[0].camera_id
        done = min(s + EXTRACT_CHUNK, len(names))
        if progress:
            progress(0.45 * done / len(names), f"COLMAP SIFT features {done}/{len(names)}")
    watch.check()


def _match_pycolmap(pycolmap, db, pairs_file, gpu, watch):
    mo = pycolmap.FeatureMatchingOptions()  # SIFT brute force: ratio 0.8, distance 0.7, cross check
    mo.use_gpu = gpu
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(pairs_file)
    vo = pycolmap.TwoViewGeometryOptions()
    vo.ransac.random_seed = SEED
    kw = {"cancellation_token": watch.token} if watch.token is not None else {}
    pycolmap.match_image_pairs(
        str(db),
        matching_options=mo,
        pairing_options=po,
        verification_options=vo,
        device=pycolmap.Device.cuda if gpu else pycolmap.Device.cpu,
        **kw,
    )
    watch.check()


def _colmap_cli(exe: Path, *args: str) -> None:
    r = subprocess.run([str(exe), *args], capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-20:])
        raise RuntimeError(f"colmap {args[0]} failed ({r.returncode}):\n{tail}")


def _extract_match_cli(db, images_dir, names, masks, pairs_file, work, progress):
    """Windows: the official COLMAP CUDA build for SIFT and matching (pycolmap's
    Windows wheel has no CUDA), same options."""
    from .downloader import ensure_binary

    exe = ensure_binary("colmap-windows")
    lst = work / "images.txt"
    lst.write_text("".join(n + "\n" for n in names))
    args = [
        "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images_dir),
        "--image_list_path", str(lst),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", CAMERA_MODEL,
        "--FeatureExtraction.use_gpu", "1",
    ]
    if masks is not None:
        args += ["--ImageReader.mask_path", str(masks)]
    _colmap_cli(exe, *args)
    if progress:
        progress(0.45, "COLMAP SIFT features done")
    _colmap_cli(
        exe,
        "matches_importer",
        "--database_path", str(db),
        "--match_list_path", str(pairs_file),
        "--match_type", "pairs",
        "--FeatureMatching.use_gpu", "1",
        "--TwoViewGeometry.random_seed", str(SEED),
    )


# ---------------------------------------------------------------- mapping


def _map(pycolmap, db, images_dir, out_dir, n_images, watch, log):
    vgc = pycolmap.ViewGraphCalibrationOptions()
    vgc.random_seed = SEED
    try:
        if not pycolmap.calibrate_view_graph(str(db), vgc):
            log("view graph calibration failed, continuing without it")
    except Exception as e:  # as colmap-wasm: continue without it
        log(f"view graph calibration failed, continuing without it ({e})")
    watch.check()
    opts = pycolmap.GlobalPipelineOptions()
    opts.random_seed = SEED
    out = Path(out_dir) / "global"
    out.mkdir(parents=True, exist_ok=True)
    recs = pycolmap.global_mapping(str(db), str(images_dir), str(out), opts)
    rec, mapper = _largest(recs), "colmap"
    if rec is None or rec.num_reg_images() < min(3, n_images):
        watch.check()
        log("global mapper registered fewer than 3 cameras; trying the incremental mapper")
        io = pycolmap.IncrementalPipelineOptions()
        io.random_seed = SEED
        out = Path(out_dir) / "incremental"
        out.mkdir(parents=True, exist_ok=True)
        kw = {"cancellation_token": watch.token} if watch.token is not None else {}
        recs2 = pycolmap.incremental_mapping(str(db), str(images_dir), str(out), io, **kw)
        rec2 = _largest(recs2)
        if rec2 is not None and (rec is None or rec2.num_reg_images() > rec.num_reg_images()):
            rec, mapper = rec2, "colmap-incremental"
    watch.check()
    return rec, mapper


def _largest(recs):
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


# ---------------------------------------------------------------- conversion


def _rgb_u8(images_dir: Path, name: str) -> np.ndarray:
    import cv2

    im = cv2.imread(str(Path(images_dir) / name), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(Path(images_dir) / name)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _read_db(pycolmap, db, names):
    """Image ids, keypoints (N,2) and raw matches by input index."""
    index = {n: i for i, n in enumerate(names)}
    with pycolmap.Database.open(str(db)) as d:
        images = d.read_all_images()
        cams = d.read_all_cameras()
        id_to_idx = {im.image_id: index[im.name] for im in images if im.name in index}
        xy: list[np.ndarray] = [np.zeros((0, 2), np.float32) for _ in names]
        for iid, i in id_to_idx.items():
            k = np.asarray(d.read_keypoints(iid), np.float32)
            xy[i] = np.ascontiguousarray(k[:, :2]) if k.size else np.zeros((0, 2), np.float32)
        pair_ids, mats = d.read_all_matches()
    matches = []
    for pid, m in zip(pair_ids, mats):
        id1, id2 = pycolmap.pair_id_to_image_pair(pid)
        if id1 not in id_to_idx or id2 not in id_to_idx:
            continue
        a, b = id_to_idx[id1], id_to_idx[id2]
        idx = np.asarray(m, np.int64).reshape(-1, 2).astype(np.int32)
        if a > b:
            a, b, idx = b, a, idx[:, ::-1].copy()
        matches.append({"a": a, "b": b, "idx": idx})
    matches.sort(key=lambda m: (m["a"], m["b"]))
    return id_to_idx, xy, matches, cams


def to_model(rec, names, id_to_idx, xy, images_dir, width, height, mapper="colmap") -> SfmModel:
    """Reconstruction -> SfmModel, like toModel in colmap-sfm.ts (point colors
    from the image at the first track element's keypoint, as colmap-wasm)."""
    n = len(names)
    poses: list[Pose | None] = [None] * n
    if rec is not None:
        cam = next(iter(rec.cameras.values()))
        f, cx, cy = float(cam.mean_focal_length()), float(cam.principal_point_x), float(cam.principal_point_y)
        for iid in rec.reg_image_ids():
            if iid not in id_to_idx:
                continue
            im = rec.images[iid]
            T = im.cam_from_world()
            M = np.asarray(T.matrix(), np.float64)
            poses[id_to_idx[iid]] = Pose(R=M[:, :3].copy(), t=M[:, 3].copy())
    else:
        f, cx, cy = 1.2 * max(width, height), width / 2, height / 2

    cache: dict[int, np.ndarray] = {}
    points: list[Point3D] = []
    obs_total, err_sum = 0, 0.0
    if rec is not None:
        for _, p in sorted(rec.points3D.items()):
            track = [(id_to_idx[e.image_id], int(e.point2D_idx)) for e in p.track.elements if e.image_id in id_to_idx]
            if not track:
                continue
            i0, k0 = track[0]
            if i0 not in cache:
                cache[i0] = _rgb_u8(images_dir, names[i0])
            img = cache[i0]
            H, W = img.shape[:2]
            x = min(W - 1, max(0, int(np.floor(xy[i0][k0, 0]))))
            y = min(H - 1, max(0, int(np.floor(xy[i0][k0, 1]))))
            rgb = tuple(int(c) for c in img[y, x])
            err = float(p.error)
            obs_total += len(track)
            err_sum += err * len(track)
            points.append(Point3D(X=np.asarray(p.xyz, np.float64).copy(), rgb=rgb, obs=track, error=err))
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
    log = log or (lambda s: None)
    pycolmap = _pycolmap()
    names = list(names)
    images_dir = Path(images_dir)
    if len(names) < 2:
        raise ValueError("need at least 2 images")
    probe = _rgb_u8(images_dir, names[0])
    height, width = probe.shape[:2]
    gpu = use_gpu(pycolmap)
    windows = sys.platform == "win32"
    tmp = tempfile.TemporaryDirectory(prefix="gausify-colmap-", dir=work_dir)
    work = Path(tmp.name)
    try:
        db = work / "database.db"
        masks = write_masks(Path(masks_dir), names, work / "masks") if masks_dir else None
        pairs_file = write_pairs(names, pairs, work / "pairs.txt")
        with _CancelWatch(pycolmap, cancel) as watch:
            t0 = time.perf_counter()
            if progress:
                progress(0.0, "COLMAP SIFT features")
            if windows:
                _extract_match_cli(db, images_dir, names, masks, pairs_file, work, progress)
                t1 = time.perf_counter()
            else:
                _extract_pycolmap(pycolmap, db, images_dir, names, masks, gpu, watch, progress)
                t1 = time.perf_counter()
                if progress:
                    progress(0.45, "COLMAP SIFT matching")
                _match_pycolmap(pycolmap, db, pairs_file, gpu, watch)
            id_to_idx, xy, matches, _ = _read_db(pycolmap, db, names)
            mean_feat = sum(len(k) for k in xy) / len(xy)
            log(f"COLMAP SIFT features: mean {mean_feat:.0f}/image in {t1 - t0:.1f} s ({'GPU' if gpu or windows else 'CPU'})")
            t2 = time.perf_counter()
            mean_m = sum(len(m["idx"]) for m in matches) / max(len(matches), 1)
            log(f"SIFT matching: {len(pairs)} pairs, mean {mean_m:.0f} matches in {t2 - t1:.1f} s")
            if progress:
                progress(0.55, "COLMAP global mapper")
            pycolmap.set_random_seed(SEED)
            rec, mapper = _map(pycolmap, db, images_dir, work, len(names), watch, log)
        model = to_model(rec, names, id_to_idx, xy, images_dir, width, height, mapper)
        s = model.stats
        log(
            f"COLMAP {'global' if mapper == 'colmap' else 'incremental'} mapper: {s['registered']}/{len(names)} cameras, "
            f"{s['points']} points, mean error {s['meanError']:.2f} px, f = {model.f:.1f} "
            f"in {time.perf_counter() - t2:.1f} s"
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
