"""The "akaze" pose estimator of gausify-orbit: AKAZE features, brute-force Hamming
matching (torch, CUDA when available) and the incremental reconstruction.

Port of `akazePoses` in web/src/sfm/index.ts (+ features.worker, matcher, geometry,
linalg, reconstruct, ba).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..types import SfmModel
from .features import ImageFeatures, extract
from .matcher import match
from .reconstruct import reconstruct

__all__ = ["akaze_poses", "extract", "match", "reconstruct", "ImageFeatures"]

_log = logging.getLogger("gausify.sfm_akaze")


def akaze_poses(
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
) -> tuple[SfmModel, list[dict]]:
    """Returns the model (poses world->cam, full-resolution keypoints in `xy`) and the
    matches of every pair (mutual + ratio test, before geometric verification, as
    index.ts passes them to correspondences/fillMissing):
    [{"a": int, "b": int, "idx": (M,2) int32 keypoint indices}]."""
    log = log or _log.info
    images_dir = Path(images_dir)
    prog = progress or (lambda _f, _m: None)

    def check():
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")

    # ---- features ----
    t0 = time.perf_counter()
    prog(0.0, "AKAZE features")
    n = len(names)
    feats: list[ImageFeatures | None] = [None] * n
    done = 0

    def job(i: int) -> ImageFeatures:
        check()
        mask = None if masks_dir is None else Path(masks_dir) / (Path(names[i]).stem + ".png")
        return extract(images_dir / names[i], max_side, max_features, mask, name=names[i])

    workers = max(1, min(8, (os.cpu_count() or 4) - 1))
    with ThreadPoolExecutor(workers) as ex:
        for i, ft in zip(range(n), ex.map(job, range(n))):
            feats[i] = ft
            done += 1
            prog(0.3 * done / n, f"AKAZE features {done}/{n}")
    t_feat = time.perf_counter() - t0
    mean_feat = sum(f.n for f in feats) / max(n, 1)
    log(f"AKAZE features: mean {mean_feat:.0f}/image in {t_feat:.1f} s")

    # ---- matching ----
    t1 = time.perf_counter()
    matches = match(
        [f.desc for f in feats],
        [(int(a), int(b)) for a, b in pairs],
        progress=lambda d, k: prog(0.3 + 0.2 * d / k, f"matching {d}/{k}"),
        cancel=cancel,
    )
    t_match = time.perf_counter() - t1
    mean_m = sum(len(m["idx"]) for m in matches) / max(len(matches), 1)
    log(f"Matching: {len(pairs)} pairs, mean {mean_m:.0f} matches in {t_match:.1f} s")
    check()

    # ---- reconstruction ----
    t2 = time.perf_counter()
    model = reconstruct(
        feats,
        matches,
        log=log,
        progress=lambda fr, msg: prog(0.5 + 0.5 * fr, msg),
        closed_loop=closed_loop,
        cancel=cancel,
    )
    t_rec = time.perf_counter() - t2
    model.names = list(names)
    model.xy = [f.xy for f in feats]
    model.stats["mapper"] = "akaze"
    log(
        f"Reconstruction: {model.stats['registered']}/{n} cameras, {model.stats['points']} points, "
        f"mean error {float(model.stats['meanError']):.2f} px, f = {model.f:.1f} in {t_rec:.1f} s"
    )
    for m in matches:
        m["idx"] = np.asarray(m["idx"], dtype=np.int32).reshape(-1, 2)
    return model, matches
