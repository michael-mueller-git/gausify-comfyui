"""Download and cache the external executables listed in `gausify/binaries.json`.

`ensure_binary(name)` returns the path of a ready-to-run executable:

1. `$GAUSIFY_<NAME>` (e.g. `GAUSIFY_BRUSH_CLI=/path/to/brush-cli`, for local
   builds) wins if set.
2. Otherwise the archive for this platform is downloaded once into the cache
   (`$GAUSIFY_BIN_DIR`, else ComfyUI's `models/gausify/bin`, else
   `~/.cache/gausify/bin`), sha256-verified, extracted next to it and renamed
   into place atomically. A lock file serialises concurrent first uses.

`ensure_model(name)` does the same for the ONNX models of `gausify/models.json`
(`$GAUSIFY_MODEL_<NAME>` overrides with a local file; cache in
`$GAUSIFY_MODEL_DIR`, else ComfyUI's `models/gausify/models`, else
`~/.cache/gausify/models`; size and sha256 are checked).

No ComfyUI dependency: `folder_paths` is only used when it can be imported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("gausify.downloader")

MANIFEST = Path(__file__).with_name("binaries.json")
CHUNK = 1 << 20


def platform_key() -> str:
    """`linux-x64`, `linux-arm64`, ... (only `linux-x64` binaries are published)."""
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    osname = "linux" if sys.platform.startswith("linux") else sys.platform
    return f"{osname}-{arch}"


def env_override_var(name: str) -> str:
    return "GAUSIFY_" + name.upper().replace("-", "_")


def bin_dir() -> Path:
    env = os.environ.get("GAUSIFY_BIN_DIR")
    if env:
        return Path(env).expanduser()
    try:  # Inside ComfyUI.
        import folder_paths  # type: ignore[import-not-found]

        return Path(folder_paths.models_dir) / "gausify" / "bin"
    except Exception:  # noqa: BLE001 - any import problem means "not in ComfyUI"
        pass
    return Path.home() / ".cache" / "gausify" / "bin"


def load_manifest(path: Path | None = None) -> dict:
    with open(path or MANIFEST, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def file_lock(path: Path):
    """Exclusive inter-process lock on `path` (created if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")  # noqa: SIM115 - held for the duration of the context
    try:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def _download(url: str, dest: Path, progress: Callable[[float, str], None] | None) -> str:
    """Stream `url` into `dest`; returns the sha256 hex digest."""
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "comfyui-gausify"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0.0
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            frac = done / total if total else 0.0
            if total and frac >= next_report:
                log.info("downloading %s: %.0f%% of %.1f MB", dest.name, 100 * frac, total / 1e6)
                next_report = frac + 0.1
            if progress:
                progress(frac, f"downloading {dest.name}")
    if total and done != total:
        raise OSError(f"incomplete download of {url}: {done} of {total} bytes")
    return h.hexdigest()


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"archive member escapes the target directory: {member}")
    return target


def _extract(archive: Path, dest: Path) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():
                _safe_target(dest, m)
            z.extractall(dest)
    elif name.endswith((".tar.xz", ".txz", ".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                _safe_target(dest, m.name)
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python without extraction filters (< 3.10.12).
                t.extractall(dest)
    else:  # A bare executable.
        shutil.copy2(archive, dest / archive.name)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def ensure_binary(
    name: str,
    progress: Callable[[float, str], None] | None = None,
    manifest: dict | None = None,
) -> Path:
    """Path to the executable `name` from binaries.json, downloading it on first use."""
    override = os.environ.get(env_override_var(name))
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{env_override_var(name)}={override} does not exist")
        return p

    manifest = manifest if manifest is not None else load_manifest()
    if name not in manifest:
        raise KeyError(f"unknown binary {name!r} (binaries.json has {sorted(manifest)})")
    entry = manifest[name]
    key = platform_key()
    plat = entry.get("platforms", {}).get(key)
    if plat is None:
        raise RuntimeError(f"{name} is not available for {key} (only {sorted(entry.get('platforms', {}))})")
    version = str(entry.get("version", "0"))
    url = plat["url"].format(version=version)
    sha256 = plat.get("sha256")
    exe_rel = plat["exe"].format(version=version)

    root = bin_dir() / name
    final = root / f"{version}-{key}"
    exe = final / exe_rel
    if exe.is_file():
        return exe

    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / f".{version}-{key}.lock"):
        if exe.is_file():  # Another process finished it while we waited.
            return exe
        work = Path(tempfile.mkdtemp(prefix=f".{version}-{key}-", dir=root))
        try:
            archive = work / url.rsplit("/", 1)[-1].split("?", 1)[0]
            log.info("downloading %s %s for %s from %s", name, version, key, url)
            digest = _download(url, archive, progress)
            if sha256:
                if digest.lower() != sha256.lower():
                    raise ValueError(f"sha256 mismatch for {url}: got {digest}, expected {sha256}")
            else:
                log.warning(
                    "no sha256 for %s %s (%s) in binaries.json; not verified (got %s)", name, version, key, digest
                )
            extracted = work / "x"
            extracted.mkdir()
            _extract(archive, extracted)
            if not (extracted / exe_rel).is_file():
                raise FileNotFoundError(f"{exe_rel} not found in {archive.name}")
            _make_executable(extracted / exe_rel)
            if final.exists():  # Stale/incomplete leftover.
                shutil.rmtree(final)
            os.replace(extracted, final)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    log.info("%s ready at %s", name, exe)
    return exe


# ------------------------------------------------------------------ models

MODELS_MANIFEST = Path(__file__).with_name("models.json")


def model_env_var(name: str) -> str:
    return "GAUSIFY_MODEL_" + name.upper().replace("-", "_")


def model_dir() -> Path:
    env = os.environ.get("GAUSIFY_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    try:  # Inside ComfyUI.
        import folder_paths  # type: ignore[import-not-found]

        return Path(folder_paths.models_dir) / "gausify" / "models"
    except Exception:  # noqa: BLE001
        pass
    return Path.home() / ".cache" / "gausify" / "models"


def model_spec(name: str, manifest: dict | None = None) -> dict:
    """Entry `name` of models.json: label, file, url, sha256, bytes."""
    manifest = manifest if manifest is not None else load_manifest(MODELS_MANIFEST)
    if name not in manifest:
        raise KeyError(f"unknown model {name!r} (models.json has {sorted(manifest)})")
    return manifest[name]


def model_cached(name: str, manifest: dict | None = None) -> bool:
    """Is the model file already in place (right size)? Env overrides count."""
    if os.environ.get(model_env_var(name)):
        return True
    spec = model_spec(name, manifest)
    f = model_dir() / spec["file"]
    return f.is_file() and f.stat().st_size == spec["bytes"]


def ensure_model(
    name: str,
    progress: Callable[[float, str], None] | None = None,
    manifest: dict | None = None,
) -> Path:
    """Path of the model file `name` from models.json, downloading it on first use
    (size and sha256 checked, atomic rename, lock file)."""
    override = os.environ.get(model_env_var(name))
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{model_env_var(name)}={override} does not exist")
        return p
    spec = model_spec(name, manifest)
    root = model_dir()
    final = root / spec["file"]
    if final.is_file() and final.stat().st_size == spec["bytes"]:
        return final
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / f".{spec['file']}.lock"):
        if final.is_file() and final.stat().st_size == spec["bytes"]:
            return final
        tmp = Path(tempfile.mkdtemp(prefix=f".{spec['file']}-", dir=root))
        try:
            part = tmp / spec["file"]
            log.info("downloading %s from %s", spec.get("label", name), spec["url"])
            digest = _download(spec["url"], part, progress)
            size = part.stat().st_size
            if size != spec["bytes"]:
                raise ValueError(f"size mismatch for {spec['url']}: got {size} bytes, expected {spec['bytes']}")
            if digest.lower() != spec["sha256"].lower():
                raise ValueError(f"sha256 mismatch for {spec['url']}: got {digest}, expected {spec['sha256']}")
            os.replace(part, final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    log.info("%s ready at %s", name, final)
    return final
