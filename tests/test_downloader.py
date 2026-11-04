import hashlib
import http.server
import io
import json
import logging
import os
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from gausify import downloader


class _Handler(http.server.SimpleHTTPRequestHandler):
    hits: dict = {}

    def do_GET(self):  # noqa: N802
        _Handler.hits[self.path] = _Handler.hits.get(self.path, 0) + 1
        super().do_GET()

    def log_message(self, *a):
        pass


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    _Handler.hits = {}
    handler = lambda *a, **k: _Handler(*a, directory=str(root), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GAUSIFY_BIN_DIR", str(tmp_path / "bin"))
    monkeypatch.delenv("GAUSIFY_FAKE_TOOL", raising=False)
    monkeypatch.setattr(downloader, "platform_key", lambda: "linux-x64")


def _tar_xz(path: Path, files: dict[str, bytes]):
    with tarfile.open(path, "w:xz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            t.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zip(path: Path, files: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(url, sha, exe="fake-tool", plat="linux-x64"):
    return {"fake-tool": {"version": "1.2.3", "platforms": {plat: {"url": url, "sha256": sha, "exe": exe}}}}


def test_download_verify_extract(server, tmp_path):
    root, base = server
    sha = _tar_xz(root / "fake-tool-1.2.3.tar.xz", {"fake-tool": b"#!/bin/sh\necho hi\n", "LICENSE": b"x"})
    m = _manifest(base + "/fake-tool-{version}.tar.xz", sha)
    exe = downloader.ensure_binary("fake-tool", manifest=m)
    assert exe.read_bytes().startswith(b"#!/bin/sh")
    assert exe.parent == tmp_path / "bin" / "fake-tool" / "1.2.3-linux-x64"
    assert os.access(exe, os.X_OK)
    # Cached: no second request.
    assert downloader.ensure_binary("fake-tool", manifest=m) == exe
    assert _Handler.hits == {"/fake-tool-1.2.3.tar.xz": 1}
    # No temp dirs left behind.
    assert sorted(p.name for p in exe.parent.parent.iterdir() if not p.name.endswith(".lock")) == ["1.2.3-linux-x64"]


def test_sha_mismatch(server, tmp_path):
    root, base = server
    _tar_xz(root / "a.tar.xz", {"fake-tool": b"x"})
    with pytest.raises(ValueError, match="sha256 mismatch"):
        downloader.ensure_binary("fake-tool", manifest=_manifest(base + "/a.tar.xz", "0" * 64))
    assert not (tmp_path / "bin" / "fake-tool" / "1.2.3-linux-x64").exists()


def test_null_sha_warns(server, caplog):
    root, base = server
    _tar_xz(root / "a.tar.xz", {"fake-tool": b"x"})
    with caplog.at_level(logging.WARNING, logger="gausify.downloader"):
        exe = downloader.ensure_binary("fake-tool", manifest=_manifest(base + "/a.tar.xz", None))
    assert exe.is_file()
    assert "not verified" in caplog.text


def test_zip_nested_exe(server):
    root, base = server
    sha = _zip(root / "c.zip", {"bin/tool": b"#!/bin/sh\n", "README": b"x"})
    exe = downloader.ensure_binary("fake-tool", manifest=_manifest(base + "/c.zip", sha, "bin/tool"))
    assert exe.name == "tool" and exe.read_bytes() == b"#!/bin/sh\n"
    assert os.access(exe, os.X_OK)


def test_missing_exe_in_archive(server):
    root, base = server
    sha = _tar_xz(root / "a.tar.xz", {"other": b"x"})
    with pytest.raises(FileNotFoundError):
        downloader.ensure_binary("fake-tool", manifest=_manifest(base + "/a.tar.xz", sha))


def test_path_traversal_rejected(server):
    root, base = server
    sha = _zip(root / "evil.zip", {"../evil": b"x", "fake-tool": b"x"})
    with pytest.raises(ValueError, match="escapes"):
        downloader.ensure_binary("fake-tool", manifest=_manifest(base + "/evil.zip", sha))


def test_concurrent_single_download(server):
    root, base = server
    sha = _tar_xz(root / "a.tar.xz", {"fake-tool": os.urandom(3 << 20)})
    m = _manifest(base + "/a.tar.xz", sha)
    results, errors = [], []

    def worker():
        try:
            results.append(downloader.ensure_binary("fake-tool", manifest=m))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(results)) == 1
    assert _Handler.hits == {"/a.tar.xz": 1}


def test_env_override(tmp_path, monkeypatch):
    exe = tmp_path / "my-tool"
    exe.write_text("x")
    monkeypatch.setenv("GAUSIFY_FAKE_TOOL", str(exe))
    assert downloader.ensure_binary("fake-tool", manifest={}) == exe
    monkeypatch.setenv("GAUSIFY_FAKE_TOOL", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        downloader.ensure_binary("fake-tool", manifest={})


def test_unsupported_platform(monkeypatch):
    monkeypatch.setattr(downloader, "platform_key", lambda: "macos-arm64")
    with pytest.raises(RuntimeError, match="not available"):
        downloader.ensure_binary("fake-tool", manifest=_manifest("http://x/a.tar.xz", None))


def test_bin_dir_default(monkeypatch):
    monkeypatch.delenv("GAUSIFY_BIN_DIR")
    d = downloader.bin_dir()
    assert d.parts[-2:] == ("gausify", "bin")


def test_manifest_shape():
    m = downloader.load_manifest()
    assert set(m) == {"brush-cli"}
    assert set(m["brush-cli"]["platforms"]) == {"linux-x64"}
    for entry in m.values():
        for p in entry["platforms"].values():
            assert p["url"].startswith("https://") and p["exe"]
            assert p["sha256"] is None or len(p["sha256"]) == 64
    json.dumps(m)


# ------------------------------------------------------------------ ensure_model


def _model_manifest(data: bytes, **over):
    return {"tiny-model": {"label": "Tiny", "file": "tiny.onnx", "url": "http://x/tiny.onnx",
                           "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), **over}}


@pytest.fixture
def model_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GAUSIFY_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("GAUSIFY_MODEL_TINY_MODEL", raising=False)
    calls = []

    def fake_download(url, dest, progress):
        calls.append(url)
        dest.write_bytes(fake_download.data)
        if progress:
            progress(1.0, "downloading")
        return hashlib.sha256(fake_download.data).hexdigest()

    fake_download.data = b"onnx-bytes" * 10
    monkeypatch.setattr(downloader, "_download", fake_download)
    return fake_download, calls, tmp_path / "models"


def test_ensure_model_downloads_once(model_env):
    fd, calls, root = model_env
    man = _model_manifest(fd.data)
    seen = []
    p = downloader.ensure_model("tiny-model", lambda f, m: seen.append(f), man)
    assert p == root / "tiny.onnx" and p.read_bytes() == fd.data and seen == [1.0]
    assert downloader.ensure_model("tiny-model", None, man) == p
    assert len(calls) == 1  # cache hit
    assert [f.name for f in root.iterdir() if f.is_file() and not f.name.startswith(".")] == ["tiny.onnx"]
    assert downloader.model_cached("tiny-model", man)


def test_ensure_model_mismatch(model_env):
    fd, calls, root = model_env
    with pytest.raises(ValueError, match="sha256 mismatch"):
        downloader.ensure_model("tiny-model", None, _model_manifest(fd.data, sha256="0" * 64))
    with pytest.raises(ValueError, match="size mismatch"):
        downloader.ensure_model("tiny-model", None, _model_manifest(fd.data, bytes=5))
    assert not (root / "tiny.onnx").exists()
    assert not [p for p in root.iterdir() if p.name.startswith(".tiny.onnx-")]  # temp dirs removed


def test_ensure_model_env_override_and_unknown(model_env, tmp_path, monkeypatch):
    fd, calls, _ = model_env
    local = tmp_path / "mine.onnx"
    local.write_bytes(b"x")
    monkeypatch.setenv("GAUSIFY_MODEL_TINY_MODEL", str(local))
    assert downloader.ensure_model("tiny-model", None, _model_manifest(fd.data)) == local
    assert not calls
    monkeypatch.setenv("GAUSIFY_MODEL_TINY_MODEL", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        downloader.ensure_model("tiny-model", None, _model_manifest(fd.data))
    monkeypatch.delenv("GAUSIFY_MODEL_TINY_MODEL")
    with pytest.raises(KeyError):
        downloader.ensure_model("nope", None, _model_manifest(fd.data))


def test_models_json():
    man = downloader.load_manifest(downloader.MODELS_MANIFEST)
    assert "rmbg" in man and len([k for k in man if k.startswith("sam2-fp32-")]) == 4
    assert len([k for k in man if k.startswith("sam2-fp16-")]) == 4
    for k, e in man.items():
        assert len(e["sha256"]) == 64 and e["bytes"] > 0 and e["url"].startswith("https://huggingface.co/"), k


def test_model_dir_default(monkeypatch, tmp_path):
    monkeypatch.delenv("GAUSIFY_MODEL_DIR", raising=False)
    monkeypatch.setattr(downloader.Path, "home", classmethod(lambda cls: tmp_path))
    d = downloader.model_dir()
    assert d.name == "models" and d.parent.name == "gausify"
