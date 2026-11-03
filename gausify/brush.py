"""Training and rendering with `brush-cli` (Brush + gausify-orbit's patches +
`brush/patches/gausify-cli.patch`).

Reproduces what gausify-orbit does with brush-js (web/src/brush.ts,
web/src/posecheck/job.ts):

- `train()` runs `brush-cli <dataset> <args>` and parses the patch's progress
  protocol: one JSON object per stdout line prefixed with `GAUSIFY `
  (`config`, `dataset`, `step`, `refine`, `eval`, `warning`, `error`, `done`).
  An initial splat is passed the way brush-js gets it: as `<dataset>/init.ply`
  (Brush's dataset loader prefers it over the COLMAP points).
- `render()` runs `brush-cli render`, the same code as brush-js
  `Training.renderColor` (render.patch) / `Training.renderDepth` (geom.patch),
  over a ply exported by `train()`.
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
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .downloader import ensure_binary

log = logging.getLogger("gausify.brush")

PREFIX = "GAUSIFY "
# Brush exports on `iter % export_every == 0` and always at the last step:
# a huge period leaves only the final export.
NO_EXPORT_EVERY = 1_000_000_000
_OWNED_FLAGS = {"--export-path": 1, "--export-name": 1, "--export-every": 1}

Progress = Callable[[float, str], None] | None
Cancel = Callable[[], bool] | None


class Cancelled(InterruptedError):
    """Raised when `cancel()` returned True; the brush-cli process was killed."""


class BrushError(RuntimeError):
    pass


def brush_exe() -> Path:
    return ensure_binary("brush-cli")


def _strip_owned(args: Sequence[str]) -> list[str]:
    """Drop the export flags train() controls (`--flag value` and `--flag=value`)."""
    out: list[str] = []
    skip = 0
    for a in args:
        a = str(a)
        if skip:
            skip -= 1
            continue
        name = a.split("=", 1)[0]
        if name in _OWNED_FLAGS:
            if "=" not in a:
                skip = _OWNED_FLAGS[name]
            continue
        out.append(a)
    return out


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    val = None
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


def _run(cmd: list[str], on_event: Callable[[dict], None], cancel: Cancel) -> tuple[int, list[str], dict | None]:
    """Run `cmd`, feeding each GAUSIFY event to `on_event`. Returns
    (exit code, last output lines, last `error` event)."""
    log.debug("running %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
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
    error: dict | None = None
    try:
        while True:
            if cancel is not None and cancel():
                proc.kill()
                proc.wait()
                raise Cancelled("brush-cli cancelled")
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break
            line = line.rstrip("\r\n")
            if not line.startswith(PREFIX):
                if line.strip():
                    tail.append(line)
                continue
            try:
                ev = json.loads(line[len(PREFIX) :])
            except json.JSONDecodeError:
                tail.append(line)
                continue
            if ev.get("event") == "error":
                error = ev
            elif ev.get("event") == "warning":
                log.warning("brush: %s", ev.get("message"))
            on_event(ev)
        rc = proc.wait()
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    for t in threads:
        t.join(timeout=5)
    return rc, list(tail), error


def _fail(what: str, rc: int, tail: list[str], error: dict | None) -> BrushError:
    msg = f"brush-cli {what} failed (exit code {rc})"
    if error and error.get("message"):
        msg += f": {error['message']}"
    if tail:
        msg += "\n" + "\n".join(tail[-30:])
    return BrushError(msg)


def _stray_plys(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.rglob("*.ply") if p.is_file())


def train(
    dataset_dir: str | os.PathLike,
    args: Sequence[str],
    out_ply: str | os.PathLike,
    init_ply: str | os.PathLike | None = None,
    progress: Progress = None,
    cancel: Cancel = None,
) -> dict:
    """Train a splat on a COLMAP dataset folder; writes the export made at the
    last step to `out_ply`. Returns {"psnr": float|None (last eval), "splats":
    int, "ssim", "seconds"}.

    `args`: Brush CLI flags (`train_config.to_cli_args`). The export flags are
    replaced (the export goes to a temporary folder outside the dataset: Brush
    takes any .ply in the dataset folder as its initial splat).
    """
    dataset_dir = Path(dataset_dir).resolve()
    out_ply = Path(out_ply)
    args = _strip_owned(args)
    total = int(_flag_value(args, "--total-train-iters") or 30000)

    work = Path(tempfile.mkdtemp(prefix="gausify-brush-"))
    init_target = dataset_dir / "init.ply"
    backup: Path | None = None
    placed = False
    try:
        if init_ply is not None:
            if init_target.exists():
                backup = work / "init.ply.orig"
                shutil.move(init_target, backup)
            shutil.copyfile(init_ply, init_target)
            placed = True
        else:
            stray = _stray_plys(dataset_dir)
            if stray:
                log.warning("Brush will start from %s (a .ply in the dataset folder)", stray[0])

        export_dir = work / "export"
        cmd = [
            str(brush_exe()),
            str(dataset_dir),
            *args,
            "--export-path", str(export_dir),
            "--export-name", "splat_{iter}.ply",
            "--export-every", str(NO_EXPORT_EVERY),
        ]  # fmt: skip
        state: dict = {"total": total, "done": None}

        def on_event(ev: dict) -> None:
            kind = ev.get("event")
            if kind == "config" and ev.get("total"):
                state["total"] = int(ev["total"])
            elif kind == "step" and progress:
                tot = max(int(ev.get("total") or state["total"]), 1)
                progress(min(int(ev["iter"]) / tot, 1.0), f"training step {ev['iter']}/{tot}")
            elif kind == "eval":
                log.info("brush eval at %s: PSNR %.2f", ev.get("iter"), ev.get("psnr") or float("nan"))
            elif kind == "done":
                state["done"] = ev

        rc, tail, error = _run(cmd, on_event, cancel)
        done = state["done"]
        if rc != 0 or done is None:
            raise _fail("training", rc, tail, error)
        if int(done.get("iter") or 0) != state["total"]:
            raise _fail(f"training stopped at step {done.get('iter')} of {state['total']}", rc, tail, error)
        ply = Path(done["ply"]) if done.get("ply") else None
        if ply is None or not ply.is_file():
            digits = len(str(state["total"]))
            ply = export_dir / f"splat_{state['total']:0{digits}d}.ply"
        if not ply.is_file():
            raise _fail(f"training produced no export at step {state['total']}", rc, tail, error)
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ply), out_ply)
        if progress:
            progress(1.0, "training done")
        return {
            "psnr": done.get("psnr"),
            "splats": int(done.get("splats") or 0),
            "ssim": done.get("ssim"),
            "seconds": done.get("seconds"),
        }
    finally:
        if placed:
            init_target.unlink(missing_ok=True)
        if backup is not None and backup.exists():
            shutil.move(backup, init_target)
        shutil.rmtree(work, ignore_errors=True)


def _camera_json(cam: dict) -> dict:
    R = np.asarray(cam["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
    out = {
        "R": R.tolist(),
        "t": t.tolist(),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
        "width": int(cam["width"]),
        "height": int(cam["height"]),
    }
    for k in ("f", "fx", "fy"):
        if cam.get(k) is not None:
            out[k] = float(cam[k])
    if "fx" not in out and "f" not in out:
        raise ValueError("camera needs f or fx")
    return out


def render(
    ply: str | os.PathLike,
    cameras: Sequence[dict],
    color: bool = True,
    progress: Progress = None,
    cancel: Cancel = None,
) -> list[dict]:
    """Render `ply` from COLMAP cameras ({R (world->cam), t, f or fx/fy, cx, cy,
    width, height}) in one brush-cli process. Per camera, in order:
    {"rgba": (H,W,4) float32 premultiplied over black (None if not `color`),
     "depth": (H,W) float32 expected camera z (0 where alpha <= 1e-4),
     "alpha": (H,W) float32}."""
    cams = [_camera_json(c) for c in cameras]
    if not cams:
        return []
    with tempfile.TemporaryDirectory(prefix="gausify-render-") as tmp:
        tmp = Path(tmp)
        (tmp / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")
        out_dir = tmp / "out"
        cmd = [
            str(brush_exe()), "render",
            "--ply", str(Path(ply).resolve()),
            "--cameras", str(tmp / "cameras.json"),
            "--out-dir", str(out_dir),
        ]  # fmt: skip
        if not color:
            cmd.append("--no-color")
        state = {"done": False}

        def on_event(ev: dict) -> None:
            if ev.get("event") == "render" and progress:
                progress((int(ev["index"]) + 1) / len(cams), f"rendering {int(ev['index']) + 1}/{len(cams)}")
            elif ev.get("event") == "done":
                state["done"] = True

        rc, tail, error = _run(cmd, on_event, cancel)
        if rc != 0 or not state["done"]:
            raise _fail("render", rc, tail, error)
        results = []
        for i, c in enumerate(cams):
            h, w = c["height"], c["width"]
            rgba = np.load(out_dir / f"{i:05d}.rgba.npy") if color else None
            depth = np.load(out_dir / f"{i:05d}.depth.npy")
            alpha = np.load(out_dir / f"{i:05d}.alpha.npy")
            if depth.shape != (h, w) or (rgba is not None and rgba.shape != (h, w, 4)):
                raise BrushError(f"brush-cli render: unexpected output shape for camera {i}")
            results.append({"rgba": rgba, "depth": depth, "alpha": alpha})
        return results
