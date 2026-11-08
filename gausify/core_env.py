"""The isolated Python environment the Gausify compute code runs in.

ComfyUI's venv is shared by every custom node, and some of them replace
packages this pack depends on (OpenCV 5 has no AKAZE, a plain `onnxruntime`
hides the CUDA provider, ...). So the compute code (`gausify/*`: frames, masks,
poses, cleaning, ...) runs in its own venv, created on first use, through
`gausify/isolated.py`. The ComfyUI process only keeps the node glue.

The venv holds the packages pinned in `PACKAGES` (OpenCV 4, ONNX Runtime GPU,
...). torch is not duplicated: a `.pth` file appends ComfyUI's site-packages
*after* the venv's own, so torch comes from ComfyUI and everything listed
here always wins over whatever other nodes installed there.

Location: `$GAUSIFY_CORE_VENV`, else ComfyUI's `models/gausify/core-venv`, else
`~/.cache/gausify/core-venv`. Rebuilt when the package list or the Python
version changes (stamp file). Extra pip arguments: `$GAUSIFY_CORE_PIP_ARGS`
(falls back to `$GAUSIFY_COLMAP_PIP_ARGS`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import site
import sys
import sysconfig
import tempfile
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .colmap_env import ColmapEnvError, _create_venv, _run_logged, venv_python
from .downloader import file_lock

log = logging.getLogger("gausify.core_env")

PACKAGES = (
    "numpy",
    "scipy",
    "opencv-python-headless>=4.12,<5",
    "onnxruntime-gpu>=1.20",
    "onnx",
    "pillow",
)
STAMP = "gausify-core-env.json"
PTH = "gausify-comfyui-site.pth"
ENV_VENV = "GAUSIFY_CORE_VENV"
ENV_PIP_ARGS = "GAUSIFY_CORE_PIP_ARGS"


class CoreEnvError(ColmapEnvError):
    pass


def venv_dir() -> Path:
    env = os.environ.get(ENV_VENV)
    if env:
        return Path(env).expanduser()
    try:  # Inside ComfyUI.
        import folder_paths  # type: ignore[import-not-found]

        return Path(folder_paths.models_dir) / "gausify" / "core-venv"
    except Exception:  # noqa: BLE001 - any import problem means "not in ComfyUI"
        pass
    return Path.home() / ".cache" / "gausify" / "core-venv"


def pip_args() -> list[str]:
    return shlex.split(os.environ.get(ENV_PIP_ARGS) or os.environ.get("GAUSIFY_COLMAP_PIP_ARGS", ""))


def spec_hash(packages=PACKAGES) -> str:
    spec = {
        "packages": list(packages),
        "pip_args": pip_args(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def parent_site_dirs() -> list[str]:
    """site-packages of the interpreter running ComfyUI (where torch lives)."""
    dirs: list[str] = []
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p and p not in dirs:
            dirs.append(p)
    try:
        for p in site.getsitepackages():
            if p not in dirs:
                dirs.append(p)
    except Exception:  # noqa: BLE001 - virtualenvs without getsitepackages
        pass
    return [d for d in dirs if os.path.isdir(d)]


def _venv_site(venv: Path) -> Path | None:
    for p in sorted(venv.glob("lib/python*/site-packages")):
        return p
    return None


def _write_pth(venv: Path) -> None:
    """Append ComfyUI's site-packages after the venv's own (.pth lines are appended)."""
    sp = _venv_site(venv)
    if sp is None:
        raise CoreEnvError(f"the core environment at {venv} has no site-packages; delete it and retry")
    text = "\n".join(parent_site_dirs()) + "\n"
    pth = sp / PTH
    if not pth.exists() or pth.read_text() != text:
        pth.write_text(text)


def is_ready(venv: Path, packages=PACKAGES) -> bool:
    try:
        stamp = json.loads((venv / STAMP).read_text())
    except (OSError, ValueError):
        return False
    return stamp.get("hash") == spec_hash(packages) and venv_python(venv).is_file()


def manual_command(venv: Path) -> str:
    root = Path(__file__).resolve().parent.parent
    code = "from gausify.core_env import ensure_core_env; print(ensure_core_env())"
    return (
        f"cd {shlex.quote(str(root))} && {ENV_VENV}={shlex.quote(str(venv))} "
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    )


def ensure_core_env(progress: Callable[[float, str], None] | None = None, packages=PACKAGES) -> Path:
    """Path of the core venv's python, creating/updating the venv if needed."""
    venv = venv_dir()
    if is_ready(venv, packages):
        _write_pth(venv)
        return venv_python(venv)
    venv.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(venv.parent / f".{venv.name}.lock"):
        if is_ready(venv, packages):
            _write_pth(venv)
            return venv_python(venv)
        log.info("setting up the Gausify environment in %s (first use; needs network access to pip)", venv)
        if progress:
            progress(0.0, "setting up the Gausify environment")
        tail: deque[str] = deque(maxlen=40)

        def on_line(line: str) -> None:
            log.info("core env: %s", line)
            if progress:
                progress(0.0, "Gausify environment: " + line[:80])

        work = Path(tempfile.mkdtemp(prefix=f".{venv.name}-", dir=venv.parent))
        try:
            _create_venv(work, tail, on_line)
            py = venv_python(work)
            cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check", *pip_args(), *packages]
            rc = _run_logged(cmd, tail, on_line)
            if rc != 0:
                raise CoreEnvError(
                    f"installing the Gausify packages ({' '.join(packages)}) failed (pip exit code {rc}):\n"
                    + "\n".join(tail)
                    + f"\nTo set it up manually: {manual_command(venv)}"
                )
            (work / STAMP).write_text(json.dumps({"hash": spec_hash(packages), "packages": list(packages)}))
            if venv.exists():
                shutil.rmtree(venv)
            os.replace(work, venv)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    if not venv_python(venv).is_file():
        raise CoreEnvError(f"the core environment at {venv} is broken; delete it and retry")
    _write_pth(venv)
    return venv_python(venv)
