"""Run library functions in the isolated core venv (see core_env.py).

`call(fn, *args, **kwargs)` runs `fn` (a module-level function of the `gausify`
package) in a subprocess of the core venv and returns its result. Arguments
and the result travel as pickles. What does not pickle works anyway:

* callables (progress, cancel, log, per-image loaders, ...) stay in the caller
  and are called back over a pipe. Progress/log style callbacks (the
  `progress`, `log`, `log_fn` keywords, or anything in `fire_and_forget`) are
  one-way; the others return their value to the worker;
* dict/list arguments the function fills in (stats, caches) are updated in place.

Exceptions are re-raised in the caller as the same type when they pickle, with
the worker's traceback as a note. `should_cancel()` is polled; when true the
worker is terminated and InterruptedError is raised.

`enabled()` is true inside ComfyUI (override: `GAUSIFY_ISOLATE=0|1`); the unit
tests run the functions in-process.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import queue
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger("gausify.isolated")

WORKER = Path(__file__).with_name("isolated_worker.py")
KILL_GRACE = 5.0
ONE_WAY_KWARGS = {"progress", "log", "log_fn", "say", "on_line"}
#: Modules that never run in the venv (no OpenCV/ONNX; brush-cli is a plain subprocess).
IN_PROCESS_MODULES = ("gausify.brush", "gausify.isolated", "gausify.colmap_env", "gausify.core_env")


class CbRef:
    """Placeholder of a caller-side callable inside the job pickle."""

    def __init__(self, id: int, reply: bool):
        self.id, self.reply = id, reply


def enabled() -> bool:
    v = os.environ.get("GAUSIFY_ISOLATE")
    if v is not None:
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    try:
        import folder_paths  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def isolatable(fn) -> bool:
    mod = getattr(fn, "__module__", "") or ""
    qn = getattr(fn, "__qualname__", "") or ""
    return mod.startswith("gausify.") and mod not in IN_PROCESS_MODULES and "<locals>" not in qn and "." not in qn


def _env() -> dict:
    from . import downloader

    env = dict(os.environ)
    env.setdefault("GAUSIFY_BIN_DIR", str(downloader.bin_dir()))
    env.setdefault("GAUSIFY_MODEL_DIR", str(downloader.model_dir()))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _b64(obj) -> str:
    return base64.b64encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _unb64(s: str):
    return pickle.loads(base64.b64decode(s))


def call(
    fn,
    *args,
    should_cancel: Callable[[], bool] | None = None,
    one_way: tuple = (),
    python: str | os.PathLike | None = None,
    **kwargs,
):
    """Runs `fn(*args, **kwargs)` in the core venv. `one_way`: types whose
    instances are one-way callbacks (e.g. the nodes' Progress)."""
    from .core_env import ensure_core_env

    callbacks: dict[int, Callable] = {}

    def ref(v, name=None):
        if callable(v) and not isinstance(v, type):
            cid = len(callbacks)
            callbacks[cid] = v
            return CbRef(cid, reply=not (name in ONE_WAY_KWARGS or isinstance(v, tuple(one_way))))
        return v

    jargs = [ref(a) for a in args]
    jkwargs = {k: ref(v, k) for k, v in kwargs.items()}
    # A one-way hint also covers the first positional progress-like callable
    # of types in `one_way`; everything else callable is synchronous.
    one_progress = next((v for v in callbacks.values() if isinstance(v, tuple(one_way))), None)
    progress_cb = one_progress or kwargs.get("progress")
    python = python or ensure_core_env(progress_cb if callable(progress_cb) else None)

    with tempfile.TemporaryDirectory(prefix="gausify-job-") as tmp:
        job_path, result_path = Path(tmp) / "job.pkl", Path(tmp) / "result.pkl"
        with open(job_path, "wb") as f:
            pickle.dump(
                {"module": fn.__module__, "qualname": fn.__qualname__, "args": jargs, "kwargs": jkwargs,
                 "result": str(result_path)},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )
        proc = subprocess.Popen(
            [str(python), "-I", str(WORKER), str(job_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=_env(),
        )
        tail: deque[str] = deque(maxlen=60)
        lines: queue.Queue = queue.Queue()

        def pump_out():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        def pump_err():
            for line in proc.stderr:
                tail.append(line.rstrip("\n"))

        threads = [threading.Thread(target=pump_out, daemon=True), threading.Thread(target=pump_err, daemon=True)]
        for t in threads:
            t.start()

        def stop():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(KILL_GRACE)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        done = False
        error: dict | None = None
        try:
            while True:
                if should_cancel is not None and should_cancel():
                    stop()
                    raise InterruptedError("cancelled")
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    kind = ev["type"]
                except (ValueError, KeyError, TypeError):
                    tail.append(line)
                    continue
                if kind == "cb":
                    cb = callbacks[ev["id"]]
                    a, kw = _unb64(ev["args"]), _unb64(ev["kwargs"])
                    if not ev["reply"]:
                        try:
                            cb(*a, **kw)
                        except InterruptedError:
                            raise
                        except Exception:  # noqa: BLE001 - a broken progress bar must not kill the job
                            log.debug("callback failed", exc_info=True)
                        continue
                    try:
                        reply = {"ok": _b64(cb(*a, **kw))}
                    except InterruptedError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        reply = {"err": f"{type(e).__name__}: {e}"}
                    try:
                        proc.stdin.write(json.dumps(reply) + "\n")
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                elif kind == "result":
                    done = True
                elif kind == "error":
                    error = ev
            rc = proc.wait()
        except BaseException:
            stop()
            raise
        for t in threads:
            t.join(timeout=5)

        if error is not None:
            exc = None
            if error.get("exc"):
                try:
                    exc = _unb64(error["exc"])
                except Exception:  # noqa: BLE001
                    exc = None
            if isinstance(exc, BaseException):
                if hasattr(exc, "add_note"):
                    exc.add_note("in the Gausify environment:\n" + str(error.get("traceback", "")).rstrip())
                raise exc
            raise RuntimeError(str(error.get("message")) + "\n" + str(error.get("traceback", "")).rstrip())
        if rc != 0 or not done or not result_path.exists():
            msg = f"Gausify worker for {fn.__qualname__} failed (exit code {rc})"
            if tail:
                msg += "\n" + "\n".join(tail)
            raise RuntimeError(msg)
        with open(result_path, "rb") as f:
            back = pickle.load(f)

    for orig, new in zip(args, back["args"]):  # containers filled in by the worker
        _sync(orig, new)
    for k, new in back["kwargs"].items():
        _sync(kwargs.get(k), new)
    return back["result"]


def _sync(orig, new) -> None:
    if new is None:
        return
    if isinstance(orig, dict) and isinstance(new, dict):
        orig.clear()
        orig.update(new)
    elif isinstance(orig, list) and isinstance(new, list):
        orig[:] = new
