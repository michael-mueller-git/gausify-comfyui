"""COLMAP worker: runs inside the isolated COLMAP venv (see colmap_env.py).

    <venv>/bin/python -I gausify/colmap_worker.py <job.json>

Self-contained apart from `cuda_libs.py` (pure stdlib, loaded by path because
`-I` keeps the package off sys.path): it preloads the CUDA runtime libraries,
imports pycolmap and runs feature extraction, matching and mapping, the port
of colmap-wasm described in sfm_colmap.py.

Job JSON: {"images_dir", "names", "masks_dir" (COLMAP layout, <name>.png) | null,
"pairs_file", "database", "work_dir", "result", "seed", "camera_model",
"extract_chunk"}. Protocol on stdout, one JSON object per line:
  {"type": "progress", "fraction": f, "message": s}
  {"type": "log", "message": s}
  {"type": "result", "backend": "GPU"|"CPU", "mapper": s, "result": path, ...}
  {"type": "error", "message": s, "traceback": s}
Everything else (pycolmap/glog output) goes to stderr. The result file is an
.npz read by sfm_colmap.py: camera, poses, keypoints, raw matches and points.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

_OUT = None  # protocol stream (the original stdout)


def emit(**ev) -> None:
    _OUT.write(json.dumps(ev) + "\n")
    _OUT.flush()


def log(msg: str) -> None:
    emit(type="log", message=msg)


def progress(frac: float, msg: str) -> None:
    emit(type="progress", fraction=float(frac), message=msg)


def load_cuda_libs():
    spec = importlib.util.spec_from_file_location("gausify_cuda_libs", Path(__file__).with_name("cuda_libs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def use_gpu(pycolmap) -> bool:
    """CUDA build of pycolmap and at least one CUDA device."""
    if not getattr(pycolmap, "has_cuda", False):
        return False
    try:
        return pycolmap.get_num_cuda_devices() > 0
    except Exception:
        return False


def reader_options(pycolmap, job, masks, existing_camera_id=-1):
    o = pycolmap.ImageReaderOptions()
    o.camera_model = job["camera_model"]
    if masks is not None:
        o.mask_path = str(masks)
    if existing_camera_id >= 0:
        o.existing_camera_id = existing_camera_id
    return o


def extract(pycolmap, job, gpu):
    names, chunk_n = job["names"], int(job["extract_chunk"])
    ext = pycolmap.FeatureExtractionOptions()  # SIFT, COLMAP defaults
    ext.use_gpu = gpu
    device = pycolmap.Device.cuda if gpu else pycolmap.Device.cpu
    camera_id = -1
    for s in range(0, len(names), chunk_n):
        pycolmap.extract_features(
            job["database"],
            job["images_dir"],
            image_names=list(names[s : s + chunk_n]),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options(pycolmap, job, job.get("masks_dir"), camera_id),
            extraction_options=ext,
            device=device,
        )
        if camera_id < 0:
            with pycolmap.Database.open(job["database"]) as d:
                camera_id = d.read_all_cameras()[0].camera_id
        done = min(s + chunk_n, len(names))
        progress(0.45 * done / len(names), f"COLMAP SIFT features {done}/{len(names)}")


def match(pycolmap, job, gpu):
    mo = pycolmap.FeatureMatchingOptions()  # SIFT brute force: ratio 0.8, distance 0.7, cross check
    mo.use_gpu = gpu
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = job["pairs_file"]
    vo = pycolmap.TwoViewGeometryOptions()
    vo.ransac.random_seed = job["seed"]
    pycolmap.match_image_pairs(
        job["database"],
        matching_options=mo,
        pairing_options=po,
        verification_options=vo,
        device=pycolmap.Device.cuda if gpu else pycolmap.Device.cpu,
    )


def largest(recs):
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


def map_images(pycolmap, job):
    seed, db, images_dir, work = job["seed"], job["database"], job["images_dir"], Path(job["work_dir"])
    vgc = pycolmap.ViewGraphCalibrationOptions()
    vgc.random_seed = seed
    try:
        if not pycolmap.calibrate_view_graph(db, vgc):
            log("view graph calibration failed, continuing without it")
    except Exception as e:  # as colmap-wasm: continue without it
        log(f"view graph calibration failed, continuing without it ({e})")
    opts = pycolmap.GlobalPipelineOptions()
    opts.random_seed = seed
    out = work / "global"
    out.mkdir(parents=True, exist_ok=True)
    rec, mapper = largest(pycolmap.global_mapping(db, images_dir, str(out), opts)), "colmap"
    if rec is None or rec.num_reg_images() < min(3, len(job["names"])):
        log("global mapper registered fewer than 3 cameras; trying the incremental mapper")
        io = pycolmap.IncrementalPipelineOptions()
        io.random_seed = seed
        out = work / "incremental"
        out.mkdir(parents=True, exist_ok=True)
        rec2 = largest(pycolmap.incremental_mapping(db, images_dir, str(out), io))
        if rec2 is not None and (rec is None or rec2.num_reg_images() > rec.num_reg_images()):
            rec, mapper = rec2, "colmap-incremental"
    return rec, mapper


def read_db(pycolmap, job):
    """Keypoints (N,2) per input index and raw matches by input index."""
    import numpy as np

    names = job["names"]
    index = {n: i for i, n in enumerate(names)}
    with pycolmap.Database.open(job["database"]) as d:
        id_to_idx = {im.image_id: index[im.name] for im in d.read_all_images() if im.name in index}
        xy = [np.zeros((0, 2), np.float32) for _ in names]
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
        matches.append((a, b, idx))
    matches.sort(key=lambda m: (m[0], m[1]))
    return id_to_idx, xy, matches


def save_result(path, rec, id_to_idx, xy, matches, n):
    import numpy as np

    poses = np.full((n, 3, 4), np.nan, np.float64)
    cam = np.zeros(3)
    if rec is not None:
        c = next(iter(rec.cameras.values()))
        cam = np.array([c.mean_focal_length(), c.principal_point_x, c.principal_point_y])
        for iid in rec.reg_image_ids():
            if iid in id_to_idx:
                poses[id_to_idx[iid]] = np.asarray(rec.images[iid].cam_from_world().matrix(), np.float64)
    xyz, err, counts, t_img, t_kp = [], [], [], [], []
    if rec is not None:
        for _, p in sorted(rec.points3D.items()):
            track = [(id_to_idx[e.image_id], int(e.point2D_idx)) for e in p.track.elements if e.image_id in id_to_idx]
            if not track:
                continue
            xyz.append(np.asarray(p.xyz, np.float64))
            err.append(float(p.error))
            counts.append(len(track))
            t_img += [a for a, _ in track]
            t_kp += [b for _, b in track]
    np.savez(
        path,
        camera=cam,
        poses=poses,
        has_rec=np.array(rec is not None),
        xy_flat=np.concatenate(xy) if xy else np.zeros((0, 2), np.float32),
        xy_counts=np.array([len(k) for k in xy], np.int64),
        match_ab=np.array([(a, b) for a, b, _ in matches], np.int64).reshape(-1, 2),
        match_counts=np.array([len(m) for _, _, m in matches], np.int64),
        match_idx=np.concatenate([m for _, _, m in matches]) if matches else np.zeros((0, 2), np.int32),
        xyz=np.array(xyz, np.float64).reshape(-1, 3),
        error=np.array(err, np.float64),
        track_counts=np.array(counts, np.int64),
        track_img=np.array(t_img, np.int64),
        track_kp=np.array(t_kp, np.int64),
    )


def run(job) -> dict:
    load_cuda_libs().preload()
    import pycolmap

    gpu = use_gpu(pycolmap)
    backend = "GPU" if gpu else "CPU"
    names = job["names"]
    t0 = time.perf_counter()
    progress(0.0, "COLMAP SIFT features")
    extract(pycolmap, job, gpu)
    t1 = time.perf_counter()
    progress(0.45, "COLMAP SIFT matching")
    match(pycolmap, job, gpu)
    id_to_idx, xy, matches = read_db(pycolmap, job)
    t2 = time.perf_counter()
    mean_feat = sum(len(k) for k in xy) / len(xy)
    log(f"COLMAP SIFT features: mean {mean_feat:.0f}/image in {t1 - t0:.1f} s ({backend})")
    mean_m = sum(len(m) for _, _, m in matches) / max(len(matches), 1)
    log(f"SIFT matching: {job['n_pairs']} pairs, mean {mean_m:.0f} matches in {t2 - t1:.1f} s")
    progress(0.55, "COLMAP global mapper")
    pycolmap.set_random_seed(job["seed"])
    rec, mapper = map_images(pycolmap, job)
    save_result(job["result"], rec, id_to_idx, xy, matches, len(names))
    return {
        "type": "result",
        "backend": backend,
        "mapper": mapper,
        "result": job["result"],
        "seconds_mapping": time.perf_counter() - t2,
        "pycolmap": getattr(pycolmap, "__version__", None),
    }


def main(argv) -> int:
    global _OUT
    # Keep fd 1 for the protocol only: C++ code (glog) writes to fd 1/2.
    _OUT = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        with open(argv[1], encoding="utf-8") as f:
            job = json.load(f)
        emit(**run(job))
        return 0
    except BaseException as e:  # noqa: BLE001
        emit(type="error", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
