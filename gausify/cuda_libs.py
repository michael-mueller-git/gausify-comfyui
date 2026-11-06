"""Preload the CUDA runtime libraries that pycolmap-cuda12 links against.

The pycolmap-cuda12 wheel expects libcudart.so.12 and libcurand.so.10 on the
loader path; pip installs them in the nvidia-cuda-runtime-cu12 and
nvidia-curand-cu12 wheels (site-packages/nvidia/*/lib), which the loader does
not search. Loading them with RTLD_GLOBAL first makes `import pycolmap` work
without touching LD_LIBRARY_PATH. No-op outside Linux.

Only used inside the COLMAP worker (gausify/colmap_worker.py, run by the isolated
venv of colmap_env.py); the main ComfyUI process never imports pycolmap.
"""

from __future__ import annotations

import ctypes
import site
import sys
import sysconfig
from pathlib import Path

LIBS = (
    ("cuda_runtime", "libcudart.so.12"),
    ("curand", "libcurand.so.10"),
)

_loaded: list[str] | None = None


def _site_dirs() -> list[Path]:
    dirs: list[str] = []
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            dirs.append(p)
    try:
        dirs += site.getsitepackages()
    except Exception:  # virtualenvs without getsitepackages
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:
        pass
    dirs += [p for p in sys.path if p.endswith("site-packages")]
    out, seen = [], set()
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(Path(d))
    return out


def find_lib(package: str, soname: str) -> Path | None:
    """Path of nvidia/<package>/lib/<soname> in any site-packages, or None."""
    for d in _site_dirs():
        p = d / "nvidia" / package / "lib" / soname
        if p.is_file():
            return p
    return None


def preload() -> list[str]:
    """Load the CUDA libraries globally (once). Returns the paths loaded.
    Missing libraries are skipped: the system ones may be found instead."""
    global _loaded
    if _loaded is not None:
        return _loaded
    _loaded = []
    if not sys.platform.startswith("linux"):
        return _loaded
    for package, soname in LIBS:
        p = find_lib(package, soname)
        if p is None:
            continue
        try:
            ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            _loaded.append(str(p))
        except OSError:
            pass
    return _loaded


def import_pycolmap():
    """`import pycolmap` after preload()."""
    preload()
    import pycolmap

    return pycolmap
