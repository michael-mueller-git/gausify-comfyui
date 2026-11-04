"""SAM 2.1 tiny for video (Meta, Apache-2.0) on ONNX Runtime: a prompt (box and/or
points) on one frame, and the mask of the same object propagated through the others
with the model's memory.

Port of gausify-orbit `web/src/pipeline/sam2.ts` (Sam2Tracker) and `sam2Masks` of
`web/src/pipeline/masks.ts`. The four graphs (encoder, decoder, memory attention,
memory encoder) are listed in `gausify/models.json` ("sam2-<variant>-<graph>"); the
memory bank is not a graph, it lives here, mirroring _prepare_memory_conditioned_features
from Transformers 4.57, with the parameters it uses outside the graphs in
`assets/sam2-params.bin`.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from gausify.downloader import ensure_model, model_spec

from .sam2_math import (
    HIDDEN,
    LOW,
    MAX_POINTERS,
    MEAN,
    MEM,
    NUM_MASKMEM,
    SIZE,
    STD,
    TOKENS,
    Params,
    fix_vision_shape,
    load_params,
    logits_to_mask,
    pointer_pos_enc,
    prompt_pixels,
    round_bf16,
    to_nchw,
    to_seq,
)

LICENSE = "SAM 2.1 (Meta), Apache-2.0"
VARIANTS = ("fp32", "fp16")
GRAPHS = ("vision_encoder", "mask_decoder", "memory_attention", "memory_encoder")


def model_names(variant: str) -> list[str]:
    """models.json keys of the four graphs of a variant, in order."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    return [f"sam2-{variant}-{g}" for g in GRAPHS]


def total_bytes(variant: str) -> int:
    return sum(model_spec(n)["bytes"] for n in model_names(variant))


_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


class _Graph:
    """An InferenceSession that casts the feeds to the dtypes it declares and
    returns float32 outputs by name."""

    def __init__(self, sess):
        self.sess = sess
        self.dtypes = {i.name: _DTYPES.get(getattr(i, "type", None)) for i in sess.get_inputs()}

    def __call__(self, outputs: list[str], **feeds) -> dict[str, np.ndarray]:
        f = {}
        for k, v in feeds.items():
            dt = self.dtypes.get(k)
            f[k] = np.ascontiguousarray(v if dt is None else np.asarray(v).astype(dt, copy=False))
        res = self.sess.run(outputs, f)
        return {k: np.asarray(r) for k, r in zip(outputs, res)}


def frame_tensor(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 frame -> normalized float32 (1, 3, 1024, 1024) (squashed)."""
    h, w = frame.shape[:2]
    if (w, h) != (SIZE, SIZE):
        interp = cv2.INTER_AREA if w * h > SIZE * SIZE else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (SIZE, SIZE), interpolation=interp)
    x = (frame[..., :3].astype(np.float32) / 255.0 - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def prompt_arrays(prompt: dict, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Box (labels 2 and 3) and points in 1024^2 input coordinates:
    float32 (1, 1, N, 2) and int32 (1, 1, N)."""
    pts: list[float] = []
    lab: list[int] = []
    sx, sy = SIZE / w, SIZE / h
    if prompt.get("box"):
        x0, y0, x1, y1 = prompt["box"]
        pts += [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
        lab += [2, 3]
    for p in prompt.get("points") or []:
        pts += [p["x"] * sx, p["y"] * sy]
        lab.append(int(p["label"]))
    if not lab:
        raise ValueError("SAM 2.1: empty prompt")
    return np.asarray(pts, np.float32).reshape(1, 1, len(lab), 2), np.asarray(lab, np.int32).reshape(1, 1, len(lab))


class Sam2Tracker:
    """Per-video SAM 2.1 state: the four graphs and the memory bank."""

    def __init__(self, sessions, params: Params | None = None, backend: str = "SAM 2.1", num_frames: int = 1):
        self.vision, self.decoder, self.attention, self.memory = (_Graph(s) for s in sessions)
        self.params = params or load_params()
        self.backend = backend
        self.num_frames = num_frames  # caps the pointers, like num_total_frames in Transformers
        self.cond: dict[int, dict] = {}
        self.non_cond: dict[int, dict] = {}
        self.mem_pos: np.ndarray | None = None  # spatial encoding of the memories: depends only on position
        self.vision_pos: np.ndarray | None = None  # same, for the frame features (seq)

    def frame(self, idx: int, image: np.ndarray, prompt: dict | None = None, reverse: bool = False):
        """Segments frame `idx` (RGB uint8 (H, W, 3)). With `prompt` (pixels), the frame becomes
        a conditioning frame (no memory, like a first click); without, it uses the memory of the
        frames already seen. `reverse`: propagating backwards. Returns (logits (256, 256) float32,
        object score)."""
        if prompt is None and not self.cond:
            raise ValueError("SAM 2.1: the first frame needs a prompt")
        h, w = image.shape[:2]
        vis = self.vision(
            ["feats0", "feats1", "feats2", "pos2"], pixel_values=frame_tensor(image)
        )
        feats2 = vis["feats2"].astype(np.float32)
        if self.vision_pos is None:
            self.vision_pos = to_seq(vis["pos2"].astype(np.float32), HIDDEN)

        if prompt is not None:
            pix = self._without_memory(feats2)
            points, labels = prompt_arrays(prompt, w, h)
        else:
            pix = to_nchw(self._conditioned(idx, feats2, reverse), HIDDEN)
            points = np.zeros((1, 1, 1, 2), np.float32)
            labels = np.asarray([-1], np.int32).reshape(1, 1, 1)

        dec = self.decoder(
            ["pred_masks", "object_pointer", "object_score_logits", "high_res_masks"],
            feats0=vis["feats0"],
            feats1=vis["feats1"],
            feats2=pix,
            input_points=points,
            input_labels=labels,
        )
        logits = dec["pred_masks"].astype(np.float32).reshape(-1)
        if logits.size != LOW * LOW:
            raise RuntimeError(f"SAM 2.1: unexpected decoder mask size {dec['pred_masks'].shape}")
        pointer = dec["object_pointer"].astype(np.float32).reshape(-1)
        object_score = float(dec["object_score_logits"].astype(np.float32).reshape(-1)[0])

        mem = self.memory(
            ["memory_tokens", "memory_pos_enc"],
            vision_features=feats2,
            pred_masks_high_res=dec["high_res_masks"],
            object_score_logits=np.asarray([[object_score]], np.float32),
            is_mask_from_pts=np.asarray([prompt is not None], np.bool_),
        )
        feat = round_bf16(mem["memory_tokens"].astype(np.float32).reshape(-1))
        if self.mem_pos is None:
            self.mem_pos = mem["memory_pos_enc"].astype(np.float32).reshape(TOKENS, MEM).copy()
        entry = {"feat": feat, "pointer": pointer.copy()}
        (self.cond if prompt is not None else self.non_cond)[idx] = entry
        self._prune(idx, reverse)
        return logits.reshape(LOW, LOW).copy(), object_score

    def _without_memory(self, feats2: np.ndarray) -> np.ndarray:
        """feats2 + no_memory_embedding: the conditioning frame uses no memory."""
        f = feats2.reshape(1, HIDDEN, 64, 64) + self.params.no_memory.reshape(1, HIDDEN, 1, 1)
        return f.astype(np.float32)

    def _conditioned(self, idx: int, feats2: np.ndarray, reverse: bool) -> np.ndarray:
        """Frame features conditioned on memory (memory_attention); returns seq (4096, 256)."""
        tpos = self.params.tpos
        blocks: list[dict] = []
        block_tpos: list[np.ndarray] = []
        # Conditioning frames (all of them), with the offset-0 temporal encoding (the last one).
        for m in self.cond.values():
            blocks.append(m)
            block_tpos.append(tpos[NUM_MASKMEM - 1])
        for off in range(NUM_MASKMEM - 1, 0, -1):
            m = self.non_cond.get(idx + off if reverse else idx - off)
            if m is not None:
                blocks.append(m)
                block_tpos.append(tpos[off - 1])
        spatial = np.concatenate([m["feat"].reshape(TOKENS, MEM) for m in blocks])
        spatial_pos = np.concatenate([self.mem_pos + tp[None, :] for tp in block_tpos])

        # Pointers: conditioning frames up to the current one, then the most recent tracked ones.
        max_ptrs = min(self.num_frames, MAX_POINTERS)
        sign = -1 if reverse else 1
        ptrs: list[tuple[int, dict]] = []
        for t, m in self.cond.items():
            if (t >= idx) if reverse else (t <= idx):
                ptrs.append(((idx - t) * sign, m))
        for d in range(1, max_ptrs):
            t = idx + d if reverse else idx - d
            if t < 0 or t >= self.num_frames:
                break
            m = self.non_cond.get(t)
            if m is not None:
                ptrs.append((d, m))
        split = HIDDEN // MEM  # each 256-d pointer becomes 4 tokens of 64
        pointer = np.concatenate([m["pointer"] for _, m in ptrs]).reshape(len(ptrs) * split, 1, MEM)
        pointer_pos = np.concatenate(
            [np.tile(pointer_pos_enc(diff / max(max_ptrs - 1, 1), self.params), (split, 1)) for diff, _ in ptrs]
        ).reshape(len(ptrs) * split, 1, MEM)

        res = self.attention(
            ["conditioned_features"],
            current_vision_features=to_seq(feats2, HIDDEN),
            current_vision_position_embeddings=self.vision_pos,
            spatial_memory=spatial.reshape(len(blocks) * TOKENS, 1, MEM).astype(np.float32),
            spatial_memory_position_embeddings=spatial_pos.reshape(len(blocks) * TOKENS, 1, MEM).astype(np.float32),
            pointer_memory=pointer.astype(np.float32),
            pointer_memory_position_embeddings=pointer_pos.astype(np.float32),
        )
        return res["conditioned_features"].astype(np.float32).reshape(TOKENS, HIDDEN)  # (1, 1, 4096, 256) = seq

    def _prune(self, idx: int, reverse: bool) -> None:
        """Keeps only memories that can still be used: those of the latest frames and
        those near the conditioning frames (the backward pass starts there)."""
        for t in list(self.non_cond):
            recent = (t - idx < MAX_POINTERS and t >= idx) if reverse else (idx - t < MAX_POINTERS and t <= idx)
            near_cond = any(abs(t - c) < MAX_POINTERS for c in self.cond)
            if not recent and not near_cond:
                del self.non_cond[t]


def open_sessions(variant: str, device: str = "auto", progress=None) -> tuple[list, str]:
    """Downloads (if needed) and opens the four graphs of a variant. The fp32 encoder
    is patched with `fix_vision_shape` first."""
    from .onnx import session

    names = model_names(variant)
    paths = []
    for i, n in enumerate(names):
        paths.append(ensure_model(n, (lambda f, m, i=i: progress(0.0, m)) if progress else None))
    sessions, backend = [], "SAM 2.1"
    for i, p in enumerate(paths):
        src = p
        if i == 0 and variant == "fp32":
            src = fix_vision_shape(p.read_bytes())
        s, b = session(src, device, "SAM 2.1")
        sessions.append(s)
        backend = b
    return sessions, backend + (" fp16" if variant == "fp16" else "")


def sam2_masks(
    frames: np.ndarray,
    prompt: dict | None,
    prompt_frame: int = 0,
    variant: str = "fp32",
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    sessions=None,
    device: str = "auto",
    params: Params | None = None,
) -> tuple[np.ndarray, dict]:
    """Masks uint8 (N, H, W) for frames uint8 (N, H, W, 3): segments `prompt_frame` from the
    normalized `prompt` ({"box": [x0, y0, x1, y1] | None, "points": [{"x","y","label"}]}; None =
    the object at the centre) and propagates forward, then backward. `sessions`: the four
    graphs' sessions, or (sessions, backend label)."""
    n, h, w = frames.shape[:3]
    if n == 0:
        raise ValueError("no frames to cut out")
    if not 0 <= prompt_frame < n:
        raise ValueError(f"prompt_frame {prompt_frame} is outside 0..{n - 1}")
    if sessions is None:
        sessions, backend = open_sessions(variant, device, progress)
    elif isinstance(sessions, tuple):
        sessions, backend = sessions
    else:
        backend = "SAM 2.1"
    tracker = Sam2Tracker(sessions, params, backend, n)
    start = prompt_frame
    order = list(range(start, n)) + list(range(start - 1, -1, -1))
    masks = np.zeros((n, h, w), dtype=np.uint8)
    coverage, no_subject = 0.0, 0
    for k, i in enumerate(order):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        px = prompt_pixels(prompt, w, h) if i == start else None
        logits, score = tracker.frame(i, frames[i], px, i < start)
        masks[i] = logits_to_mask(logits, w, h)
        if score <= 0:
            no_subject += 1
        coverage += float(masks[i].sum(dtype=np.float64)) / (255.0 * masks[i].size)
        if progress:
            progress((k + 1) / n, f"SAM 2.1 {k + 1}/{n}")
    stats = {"count": n, "noSubject": no_subject, "meanCoverage": coverage / n, "backend": tracker.backend, "keyColor": None}
    return masks, stats
