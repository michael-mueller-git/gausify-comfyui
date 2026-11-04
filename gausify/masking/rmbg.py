"""RMBG-1.4 background removal (BRIA AI, non-commercial licence) on ONNX Runtime.

Port of gausify-orbit `web/src/pipeline/segment.ts` (MODELS.rmbg, inputSize,
preprocess, postprocess, Segmenter.mask, resizeMask) and the RMBG branch of
`web/src/pipeline/masks.ts` (makeMasks). The model spec lives in
`gausify/models.json` (key "rmbg").
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from gausify.alpha import _js_round
from gausify.downloader import ensure_model, model_spec

LICENSE = "BRIA RMBG-1.4, non-commercial use"
LABEL = "RMBG-1.4"
SIZE = 1024
MEAN = (0.5, 0.5, 0.5)
STD = (1.0, 1.0, 1.0)


def model() -> dict:
    """{label, file, url, sha256, bytes} from models.json, plus mean/std/license."""
    return {**model_spec("rmbg"), "license": LICENSE, "mean": MEAN, "std": STD}


def input_size(w: int, h: int) -> tuple[int, int]:
    """1024x1024 (squashed, aspect ratio not preserved)."""
    return SIZE, SIZE


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    down = w * h < img.shape[1] * img.shape[0]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3) at input size -> normalized float32 NCHW (1, 3, H, W)."""
    x = rgb[..., :3].astype(np.float32) / 255.0
    x = (x - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def postprocess(out: np.ndarray) -> np.ndarray:
    """Model output -> uint8 at input size: min-max stretch unless the frame is
    nearly constant (range <= 0.5), clamp, x255 with JS rounding."""
    out = np.asarray(out, dtype=np.float32)
    lo, hi = float(out.min()), float(out.max())
    lo_, scale = (lo, 1.0 / (hi - lo)) if hi - lo > 0.5 else (0.0, 1.0)
    v = np.clip((out.astype(np.float64) - lo_) * scale, 0.0, 1.0) * 255.0
    return _js_round(v).astype(np.uint8)


def frame_mask(sess, frame: np.ndarray) -> np.ndarray:
    """Mask (255 = subject) at the size of one RGB uint8 frame (H, W, 3)."""
    h, w = frame.shape[:2]
    iw, ih = input_size(w, h)
    x = preprocess(_resize(frame, iw, ih))
    outs = sess.run([sess.get_outputs()[-1].name], {sess.get_inputs()[0].name: x})
    small = postprocess(np.asarray(outs[0]).reshape(ih, iw))
    return _resize(small, w, h)


def rmbg_masks(
    frames: np.ndarray,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    session=None,
    device: str = "auto",
) -> tuple[np.ndarray, dict]:
    """Masks uint8 (N, H, W) for frames uint8 (N, H, W, 3), and the stats dict.
    `session`: an InferenceSession or a (session, backend label) pair; default:
    the downloaded model on `device`."""
    if len(frames) == 0:
        raise ValueError("no frames to cut out")
    if session is None:
        from .onnx import session as open_session

        path = ensure_model("rmbg", (lambda f, m: progress(0.0, m)) if progress else None)
        session = open_session(path, device, LABEL)
    sess, backend = session if isinstance(session, tuple) else (session, LABEL)
    n = len(frames)
    masks = np.zeros(frames.shape[:3], dtype=np.uint8)
    coverage = 0.0
    for i in range(n):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        masks[i] = frame_mask(sess, frames[i])
        coverage += float(masks[i].sum(dtype=np.float64)) / (255.0 * masks[i].size)
        if progress:
            progress((i + 1) / n, f"RMBG-1.4 {i + 1}/{n}")
    stats = {"count": n, "noSubject": 0, "meanCoverage": coverage / n, "backend": backend, "keyColor": None}
    return masks, stats
