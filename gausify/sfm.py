"""Poses of an orbit: pairs -> estimator -> orbit corrections -> COLMAP model.

Port of `runSfm` in gausify-orbit web/src/sfm/index.ts (and the poses stage of
web/src/host/job.ts for the stats and the "too few cameras" error).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .presets import SFM_MAX_FEATURES, SFM_MAX_SIDE
from .types import SfmModel

log = logging.getLogger("gausify.sfm")

ESTIMATORS = ("akaze", "aliked_lightglue", "colmap")


def _times_map(names: Sequence[str], times) -> dict[str, float] | None:
    if times is None:
        return None
    if isinstance(times, Mapping):
        return {str(k): float(v) for k, v in times.items()}
    times = list(getattr(times, "times", times))
    if len(times) != len(names):
        return None
    return {n: float(t) for n, t in zip(names, times)}


def run_sfm(
    images_dir,
    names: Sequence[str],
    masks_dir=None,
    estimator: str = "akaze",
    window: int = 8,
    closed_loop: bool = True,
    flat_orbit: bool = False,
    fill_missing: bool = False,
    times=None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    sparse_dir=None,
    log_fn: Callable[[str], None] | None = None,
    max_side: int = SFM_MAX_SIDE,
    max_features: int = SFM_MAX_FEATURES,
) -> SfmModel:
    """`times`: timestamps by name (dict), or a list/OrbitTimes parallel to
    `names`; None = name order. Writes the model to `sparse_dir` when given.
    Timings land in model.stats["timings"] (seconds)."""
    from . import orbit
    from .pairs import orbit_pairs, orbit_times

    say = log_fn or log.info
    names = list(names)
    if len(names) < 3:
        raise ValueError("Too few images (at least 3).")
    t0 = time.perf_counter()
    pairs = orbit_pairs(len(names), int(window), bool(closed_loop))
    est = estimator or "akaze"
    if est == "colmap":
        from .sfm_colmap import colmap_poses_with_matches

        model, matches = colmap_poses_with_matches(
            images_dir, names, masks_dir, pairs, progress=progress, cancel=cancel, log=say
        )
    elif est == "akaze":
        from .sfm_akaze import akaze_poses

        model, matches = akaze_poses(
            images_dir, names, masks_dir, pairs, max_side, max_features, progress=progress, cancel=cancel,
            closed_loop=bool(closed_loop), log=say,
        )
    elif est == "aliked_lightglue":
        from .sfm_lightglue import lightglue_poses

        model, matches = lightglue_poses(
            images_dir, names, masks_dir, pairs, max_side, max_features, progress=progress, cancel=cancel,
            closed_loop=bool(closed_loop), log=say,
        )
    else:
        raise ValueError(f"unknown estimator {estimator!r} (expected one of {ESTIMATORS})")
    if cancel and cancel():
        raise InterruptedError("cancelled")
    if not model.names:
        model.names = names
    xy = model.xy

    # Orbit corrections, as in colmap.rs (flat orbit) and pipeline.rs.
    if flat_orbit:
        if progress:
            progress(0.99, "flattening the orbit")
        dev = orbit.flatten_orbit(model, xy)
        if dev is None:
            say("flat orbit not applied: the cameras barely form an arc")
        else:
            model.stats["flat"] = 1
            say(
                f"flat orbit: largest height deviation corrected {dev * 100:.2f}% of the radius, "
                f"error {float(model.stats.get('meanError', 0)):.2f} px"
            )
    if fill_missing:
        at = orbit_times(names, _times_map(names, times))
        r = orbit.fill_missing(model, matches, at, flat=bool(flat_orbit), xy=xy)
        inferred, refined = list(r.get("inferred", [])), list(r.get("refined", []))
        model.stats["inferred"] = len(inferred)
        model.stats["refined"] = len(refined)
        if inferred:
            say("poses inferred from neighbours: " + ", ".join(names[i] for i in inferred))
        for i, n in refined:
            say(f"pose of {names[i]} refined against {n} 3D points")
    model.stats["registered"] = int(model.stats.get("registered", sum(p is not None for p in model.poses)))
    model.stats["timings"] = {"total": time.perf_counter() - t0}
    if sparse_dir is not None:
        from .colmap_io import write_model

        Path(sparse_dir).mkdir(parents=True, exist_ok=True)
        write_model(model, sparse_dir, names, xy)
    return model


def model_stats(model: SfmModel) -> dict:
    """ModelStats of job.ts (poses stage)."""
    s = model.stats
    mapper = "COLMAP global mapper" if s.get("mapper") == "colmap" else "incremental SfM"
    return {
        "registered": int(s.get("registered", 0)),
        "total": int(s.get("total", len(model.poses))),
        "points": int(s.get("points", len(model.points))),
        "reprojectionError": float(s.get("meanError", 0.0)),
        "mapper": " · ".join(x for x in (mapper, "flat orbit" if s.get("flat") else None) if x),
        "inferred": int(s.get("inferred", 0)),
        "refined": int(s.get("refined", 0)),
    }
