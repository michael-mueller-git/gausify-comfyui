"""Shared helpers of the Gausify Orbit nodes: ComfyUI glue (guarded so the
tests run without ComfyUI), presets, tensors <-> files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from gausify.presets import DEFAULT_PRESET, PRESET_NAMES, PRESETS

CATEGORY = "Gausify Orbit"

try:  # ComfyUI
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - outside ComfyUI
    folder_paths = None


# ---------------------------------------------------------------- ComfyUI glue


def temp_dir() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_temp_directory())
    import tempfile

    return Path(tempfile.gettempdir()) / "comfyui-gausify"


def output_dir() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return Path.cwd() / "output"


def interrupted() -> bool:
    try:
        import comfy.model_management as mm  # type: ignore
    except Exception:
        return False
    return bool(mm.processing_interrupted())


def raise_if_interrupted() -> None:
    try:
        import comfy.model_management as mm  # type: ignore
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()


class Progress:
    """`progress(frac, msg)` callback on a ComfyUI ProgressBar (0..1000), over
    the sub-range [lo, hi]."""

    STEPS = 1000

    def __init__(self, lo: float = 0.0, hi: float = 1.0, bar=None):
        self.lo, self.hi = lo, hi
        self.last_msg = ""
        if bar is None:
            try:
                import comfy.utils  # type: ignore

                bar = comfy.utils.ProgressBar(self.STEPS)
            except Exception:
                bar = None
        self.bar = bar

    def sub(self, lo: float, hi: float) -> Progress:
        span = self.hi - self.lo
        return Progress(self.lo + span * lo, self.lo + span * hi, self.bar)

    def __call__(self, frac: float, msg: str = "") -> None:
        self.last_msg = msg or self.last_msg
        if self.bar is None:
            return
        f = self.lo + (self.hi - self.lo) * min(max(float(frac), 0.0), 1.0)
        self.bar.update_absolute(int(round(f * self.STEPS)), self.STEPS)


def run_cancellable(fn, *args, **kwargs):
    """Calls fn; a cancellation raised inside (InterruptedError or our
    Cancelled) becomes ComfyUI's interrupt when the user interrupted."""
    from gausify import isolated

    try:
        if isolated.enabled() and isolated.isolatable(fn):
            # Compute code runs in the pack's own venv (OpenCV 4, ONNX Runtime GPU, ...), not in
            # ComfyUI's shared one.
            return isolated.call(fn, *args, should_cancel=interrupted, one_way=(Progress,), **kwargs)
        return fn(*args, **kwargs)
    except Exception as e:
        if interrupted() or isinstance(e, InterruptedError) or type(e).__name__ == "Cancelled":
            raise_if_interrupted()
        raise


# ---------------------------------------------------------------- presets


def preset_input(stage: str | None = None):
    return (PRESET_NAMES, {"default": DEFAULT_PRESET, "tooltip": "Quality preset (src/lib/presets.ts). "
                           "Anything but 'custom' overrides the preset's widgets."})


#: Node widget -> (preset stage, preset key, conversion widget <- preset value)
PRESET_WIDGETS = {
    "frames": {
        "frames": ("target_frames", lambda v: int(v)),
        "max_side": ("max_side", lambda v: int(v)),
        "candidates": ("oversample", lambda v: int(v)),
    },
    "poses": {
        "match_window": ("match_window", lambda v: int(v)),
        "closed_loop": ("closed_loop", lambda v: bool(v)),
    },
    "train": {
        "steps": ("total_steps", lambda v: int(v)),
        "sh_degree": ("sh_degree", lambda v: int(v)),
        "max_splats": ("max_splats", lambda v: round(v / 1e6, 1)),
        "max_resolution": ("max_resolution", lambda v: int(v)),
        "ssim": ("ssim_weight", lambda v: v > 0),
        "psnr": ("eval_split_every", lambda v: v > 0),
    },
}


def apply_preset(stage: str, preset: str, values: dict) -> dict:
    """The widget values with the preset's applied (unless preset == custom)."""
    out = dict(values)
    if preset == "custom" or preset not in PRESETS:
        return out
    p = PRESETS[preset][stage]
    for widget, (key, conv) in PRESET_WIDGETS[stage].items():
        out[widget] = conv(p[key])
    return out


def presets_for_js() -> dict:
    """Widget values per preset and node stage (the table web/js/presets.js uses)."""
    return {
        name: {stage: apply_preset(stage, name, {}) for stage in PRESET_WIDGETS}
        for name in PRESETS
    }


# ---------------------------------------------------------------- tensors <-> files


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def images_to_u8(images) -> np.ndarray:
    """IMAGE (B,H,W,C) float [0,1] -> uint8 (B,H,W,3)."""
    a = to_numpy(images)
    if a.ndim == 3:
        a = a[None]
    a = a[..., :3]
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def masks_to_u8(masks, shape: tuple[int, int]) -> np.ndarray:
    """MASK (B,H,W) float [0,1] -> uint8 (B,H,W), 255 = subject, resized to shape."""
    import cv2

    a = to_numpy(masks)
    if a.ndim == 2:
        a = a[None]
    if a.ndim == 4:
        a = a[..., 0]
    u = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = shape
    if u.shape[1:] != (h, w):
        u = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in u])
    return u


def content_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.shape).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def frame_names(n: int) -> list[str]:
    return [f"{i:05d}.png" for i in range(n)]


def write_pngs(arrays: np.ndarray, out_dir: Path, names: list[str], rgb: bool = True) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    for a, n in zip(arrays, names):
        img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if rgb else a
        if not cv2.imwrite(str(out_dir / n), img):
            raise OSError(f"could not write {out_dir / n}")


_COUNTER = re.compile(r"_(\d{5})\.ply$")


def next_ply_path(prefix: str, root: Path | None = None) -> tuple[Path, str]:
    """<output>/<prefix>_NNNNN.ply (prefix may hold subfolders). Returns the
    absolute path and the path relative to the output directory (POSIX)."""
    root = Path(root) if root is not None else output_dir()
    prefix = (prefix or "gausify").strip().replace("\\", "/").strip("/") or "gausify"
    full = (root / prefix).resolve()
    if root.resolve() not in full.parents and full != root.resolve():
        raise ValueError("filename_prefix must stay inside the output directory")
    folder, stem = full.parent, full.name
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in folder.glob(f"{stem}_*.ply"):
        m = _COUNTER.search(f.name)
        if m and f.name == f"{stem}_{m.group(1)}.ply":
            n = max(n, int(m.group(1)))
    path = folder / f"{stem}_{n + 1:05d}.ply"
    return path, path.relative_to(root.resolve()).as_posix()
