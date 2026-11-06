"""The isolated Python environment COLMAP (pycolmap) runs in.

ComfyUI's venv has torch built for CUDA 13 (which pins `cuda-toolkit==13.x`),
while `pycolmap-cuda12` needs `cuda-toolkit<13`: installing it next to torch
breaks the environment. So pycolmap lives in a separate venv, created on first
use, and `gausify/colmap_worker.py` runs inside it as a subprocess. The main
process never imports pycolmap.

`ensure_colmap_env()` returns the venv's python. The venv is created next to
the other caches: `$GAUSIFY_COLMAP_VENV`, else ComfyUI's
`models/gausify/colmap-venv`, else `~/.cache/gausify/colmap-venv`. It is
rebuilt when the package list or the Python version changes (stamp file).
Extra pip arguments: `$GAUSIFY_COLMAP_PIP_ARGS` (e.g. `--index-url ...`); the
user's pip configuration and `PIP_*` variables apply as usual.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .downloader import file_lock

log = logging.getLogger("gausify.colmap_env")

#: Pinned packages of the worker venv (cuda-toolkit<13 comes with pycolmap-cuda12).
PACKAGES = (
    "pycolmap-cuda12==4.2.1",
    "numpy",
    "nvidia-cuda-runtime-cu12",
    "nvidia-curand-cu12",
)
STAMP = "gausify-colmap-env.json"
ENV_VENV = "GAUSIFY_COLMAP_VENV"
ENV_PIP_ARGS = "GAUSIFY_COLMAP_PIP_ARGS"


class ColmapEnvError(RuntimeError):
    pass


def venv_dir() -> Path:
    env = os.environ.get(ENV_VENV)
    if env:
        return Path(env).expanduser()
    try:  # Inside ComfyUI.
        import folder_paths  # type: ignore[import-not-found]

        return Path(folder_paths.models_dir) / "gausify" / "colmap-venv"
    except Exception:  # noqa: BLE001 - any import problem means "not in ComfyUI"
        pass
    return Path.home() / ".cache" / "gausify" / "colmap-venv"


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def pip_args() -> list[str]:
    return shlex.split(os.environ.get(ENV_PIP_ARGS, ""))


def spec_hash(packages=PACKAGES, extra_args: list[str] | None = None) -> str:
    """Hash of what the venv should contain: packages, extra pip args and the
    Python version (major.minor) of the interpreter that creates it."""
    spec = {
        "packages": list(packages),
        "pip_args": list(pip_args() if extra_args is None else extra_args),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def is_ready(venv: Path, packages=PACKAGES) -> bool:
    try:
        stamp = json.loads((venv / STAMP).read_text())
    except (OSError, ValueError):
        return False
    return stamp.get("hash") == spec_hash(packages) and venv_python(venv).is_file()


def manual_command(venv: Path, packages=PACKAGES) -> str:
    return (
        f"{shlex.quote(sys.executable)} -m venv {shlex.quote(str(venv))} && "
        f"{shlex.quote(str(venv_python(venv)))} -m pip install {' '.join(shlex.quote(p) for p in packages)}"
    )


def _run_logged(cmd: list[str], tail: deque, on_line: Callable[[str], None]) -> int:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                on_line(line)
        return proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise


def _create_venv(path: Path, tail: deque, on_line) -> None:
    rc = _run_logged([sys.executable, "-m", "venv", str(path)], tail, on_line)
    if rc == 0 and venv_python(path).is_file():
        # ensurepip may be missing (Debian without python3-venv): check pip.
        if subprocess.run([str(venv_python(path)), "-m", "pip", "--version"], capture_output=True).returncode == 0:
            return
    uv = shutil.which("uv")
    if uv is None:
        raise ColmapEnvError("could not create a venv with `python -m venv` (and `uv` is not available)")
    shutil.rmtree(path, ignore_errors=True)
    on_line("python -m venv failed; trying uv")
    if _run_logged([uv, "venv", "--python", sys.executable, "--seed", str(path)], tail, on_line) != 0:
        raise ColmapEnvError("could not create a venv with `python -m venv` or `uv venv`")


def ensure_colmap_env(
    progress: Callable[[float, str], None] | None = None,
    packages=PACKAGES,
) -> Path:
    """Path of the worker venv's python, creating/updating the venv if needed."""
    venv = venv_dir()
    if is_ready(venv, packages):
        return venv_python(venv)
    venv.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(venv.parent / f".{venv.name}.lock"):
        if is_ready(venv, packages):  # Another process finished it while we waited.
            return venv_python(venv)
        msg = f"setting up the COLMAP environment in {venv} (first use; needs network access to pip)"
        log.info(msg)
        if progress:
            progress(0.0, "setting up the COLMAP environment")
        tail: deque[str] = deque(maxlen=40)

        def on_line(line: str) -> None:
            log.info("colmap env: %s", line)
            if progress:
                progress(0.0, "COLMAP environment: " + line[:80])

        work = Path(tempfile.mkdtemp(prefix=f".{venv.name}-", dir=venv.parent))
        try:
            _create_venv(work, tail, on_line)
            py = venv_python(work)
            cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check", *pip_args(), *packages]
            rc = _run_logged(cmd, tail, on_line)
            if rc != 0:
                raise ColmapEnvError(
                    f"installing the COLMAP packages failed (pip exit code {rc}):\n"
                    + "\n".join(tail)
                    + f"\nTo set it up manually: {manual_command(venv, packages)}"
                )
            (work / STAMP).write_text(json.dumps({"hash": spec_hash(packages), "packages": list(packages)}))
            # Renaming is fine: bin/python finds pyvenv.cfg next to itself; only the
            # console-script shebangs hold the build path and we only run `python -m`.
            if venv.exists():
                shutil.rmtree(venv)
            os.replace(work, venv)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    if not venv_python(venv).is_file():
        raise ColmapEnvError(f"the COLMAP environment at {venv} is broken; delete it and retry")
    return venv_python(venv)
