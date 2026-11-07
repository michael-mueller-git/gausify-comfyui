"""Pose consistency score: each frame against the silhouette of a model that
never saw it (the other fold). Where the model sticks out of the frame's mask,
that frame's pose disagrees with the others'.

Port of web/src/posecheck/score.ts (gausify-orbit).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .image import open_disk


def xor_score(alpha: np.ndarray, mask: np.ndarray, radius: int) -> dict:
    """XOR of the render's silhouette (alpha > 0.5) and the frame's mask, split
    into "out" (model outside the mask) and "in", each opened with a disk of
    radius `radius`. Returns {"out", "in", "subject"} pixel counts."""
    r = np.asarray(alpha) > 0.5
    g = np.asarray(mask) != 0
    out = (r & ~g).astype(np.uint8)
    inn = (g & ~r).astype(np.uint8)
    return {
        "out": int(open_disk(out, radius).sum()),
        "in": int(open_disk(inn, radius).sum()),
        "subject": int(g.sum()),
    }


def frame_value(s: dict) -> float:
    """"Out" area over the subject's (0 for an empty mask)."""
    return s["out"] / s["subject"] if s["subject"] > 0 else 0.0


def orbit_score(values: Sequence[float]) -> float:
    """Mean of the frames' values: the orbit's score."""
    values = list(values)
    if not values:
        return 0.0
    acc = 0.0
    for v in values:
        acc += v
    return acc / len(values)


def segments(values: Sequence[float], size: int) -> list[dict]:
    """Mean of each run of `size` consecutive frames (the last may be shorter)."""
    values = list(values)
    out = []
    for i in range(0, len(values), size):
        run = values[i : i + size]
        out.append({"start": i, "end": i + len(run) - 1, "value": orbit_score(run)})
    return out


def opening_radius(max_side: int) -> int:
    """Opening radius at an analysis size (longer side): ~2 px of a 1280 px
    frame, at least 1 (JS Math.round: halves round up)."""
    return max(1, int(np.floor(max_side / 640 + 0.5)))


#: inconsistent_frames: a frame is only dropped above this x the median value.
OUTLIER_RATIO = 1.5


def inconsistent_frames(names: Sequence[str], values: Sequence[float], fraction: float,
                        min_ratio: float = OUTLIER_RATIO, keep_at_least: int = 8) -> list[str]:
    """Beyond the app: the frames to leave out of training, worst first. At
    most `fraction` of the frames (the highest values), and of those only
    outliers (value > min_ratio x the median and > 0), never leaving fewer
    than `keep_at_least` frames."""
    names, values = list(names), [float(v) for v in values]
    if fraction <= 0 or len(names) != len(values) or not values:
        return []
    med = float(np.median(values))
    budget = min(int(len(values) * float(fraction)), max(0, len(values) - keep_at_least))
    order = sorted(range(len(values)), key=lambda i: -values[i])[:budget]
    return [names[i] for i in order if values[i] > 0 and values[i] > min_ratio * med]
