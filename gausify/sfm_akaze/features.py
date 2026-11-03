"""AKAZE feature extraction (OpenCV 4.12, same build of the algorithm as
@techstark/opencv-js 4.12). Port of web/src/sfm/features.worker.ts.

Returns keypoints in original-image pixels (COLMAP convention, +0.5), binary
descriptors (61 bytes = 486 bits) and each keypoint's color.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

THRESHOLDS = (0.0008, 0.0002, 0.00005)


@dataclass
class ImageFeatures:
    name: str
    width: int
    height: int
    n: int
    xy: np.ndarray  # (n,2) float32, full-resolution pixels
    rgb: np.ndarray  # (n,3) uint8
    desc: np.ndarray | None = None  # (n,61) uint8


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path, W: int, H: int) -> np.ndarray:
    """First channel of the mask drawn at W x H (drawImage(mb, 0, 0, W, H))."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"cannot read mask {path}")
    if m.ndim == 3:
        m = m[:, :, 2] if m.shape[2] >= 3 else m[:, :, 0]  # BGR(A) -> R
    if m.dtype != np.uint8:
        m = (m.astype(np.float64) / np.iinfo(m.dtype).max * 255 + 0.5).astype(np.uint8)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return m


def _js_round(x: float) -> int:
    return math.floor(x + 0.5)


def extract(
    image: Path | np.ndarray,
    max_side: int = 1600,
    max_features: int = 4000,
    mask: Path | np.ndarray | None = None,
    name: str = "",
) -> ImageFeatures:
    """AKAZE on the grayscale image resized to `max_side` (INTER_AREA), adaptive
    threshold, features only where mask > 0, the `max_features` strongest by response."""
    rgb = _read_rgb(Path(image)) if not isinstance(image, np.ndarray) else image
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, max_side / max(W, H))
    if scale < 1:
        gray = cv2.resize(gray, (_js_round(W * scale), _js_round(H * scale)), interpolation=cv2.INTER_AREA)

    akaze = cv2.AKAZE_create()
    kps, desc = (), None
    for thr in THRESHOLDS:
        akaze.setThreshold(thr)
        kps, desc = akaze.detectAndCompute(gray, None)
        if len(kps) >= max_features * 0.6:
            break
    total = len(kps)
    pt = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
    resp = np.array([k.response for k in kps], dtype=np.float32)

    order = np.arange(total)
    if mask is not None:
        mpx = _read_mask(Path(mask), W, H) if not isinstance(mask, np.ndarray) else mask
        if mpx.dtype != np.uint8:
            mpx = np.clip(mpx * 255 + 0.5, 0, 255).astype(np.uint8) if mpx.dtype.kind == "f" else mpx.astype(np.uint8)
        mx = np.minimum(W - 1, np.floor((pt[:, 0] + 0.5) / scale)).astype(np.int64)
        my = np.minimum(H - 1, np.floor((pt[:, 1] + 0.5) / scale)).astype(np.int64)
        order = order[mpx[my, mx] > 0] if total else order
    # stable sort by decreasing response (Array.prototype.sort is stable)
    order = order[np.argsort(-resp[order], kind="stable")]
    n = min(len(order), max_features)
    sel = order[:n]
    xy64 = (pt[sel] + 0.5) / scale
    xy = xy64.astype(np.float32)
    px = np.clip(np.floor(xy64[:, 0]), 0, W - 1).astype(np.int64)
    py = np.clip(np.floor(xy64[:, 1]), 0, H - 1).astype(np.int64)
    cols = rgb[py, px].astype(np.uint8).reshape(-1, 3)
    d = np.zeros((n, 61), dtype=np.uint8) if desc is None else np.ascontiguousarray(desc[sel])
    return ImageFeatures(name=name, width=W, height=H, n=n, xy=xy, rgb=cols, desc=d)
