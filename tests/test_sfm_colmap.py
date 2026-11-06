"""End-to-end test of the "colmap" estimator on a small synthetic orbit
(textured boxes on a textured floor, rendered with OpenCV homographies)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
# Import the library first: pytest would otherwise import the repo root __init__.py as "gausify".
import gausify.sfm_colmap  # noqa: E402,F401

REPO_DIR = Path(__file__).resolve().parents[1]
W, H, F = 480, 360, 420.0
N = 24
RADIUS, HEIGHT = 4.0, 1.4


@pytest.fixture(scope="module")
def colmap_venv(tmp_path_factory):
    """The real worker venv (built on first use: needs network). Set
    GAUSIFY_COLMAP_VENV to reuse one between runs."""
    mp = pytest.MonkeyPatch()
    if not os.environ.get("GAUSIFY_COLMAP_VENV"):
        mp.setenv("GAUSIFY_COLMAP_VENV", str(tmp_path_factory.mktemp("colmap") / "venv"))
    yield
    mp.undo()


needs_pycolmap = pytest.mark.slow


def _texture(seed: int, size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size, 3), np.float32)
    for k, s in enumerate((4, 8, 16, 32, 64)):
        n = rng.random((s, s, 3)).astype(np.float32)
        acc += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC) / (k + 1)
    acc -= acc.min()
    acc /= acc.max()
    # some crisp shapes too
    img = (acc * 255).astype(np.uint8)
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.circle(img, p, int(rng.integers(4, 24)), c, -1)
    return img


def _look_at(c: np.ndarray, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    fwd = target - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world -> cam rows
    return R, -R @ c


def _box_faces(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1, y0, y1, z0, z1 = cx - sx, cx + sx, cy - sy, cy + sy, cz - sz, cz + sz
    v = lambda x, y, z: np.array([x, y, z], float)  # noqa: E731
    return [
        ([v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1)], np.array([0, 0, 1.0])),
        ([v(x1, y0, z0), v(x1, y1, z0), v(x1, y1, z1), v(x1, y0, z1)], np.array([1.0, 0, 0])),
        ([v(x0, y1, z0), v(x0, y0, z0), v(x0, y0, z1), v(x0, y1, z1)], np.array([-1.0, 0, 0])),
        ([v(x1, y1, z0), v(x0, y1, z0), v(x0, y1, z1), v(x1, y1, z1)], np.array([0, 1.0, 0])),
        ([v(x0, y0, z0), v(x1, y0, z0), v(x1, y0, z1), v(x0, y0, z1)], np.array([0, -1.0, 0])),
    ]


def render_orbit(out_dir, n: int = N):
    """Frames 00000.png ... of a circular orbit. Returns (names, centers)."""
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    floor = ([np.array(p, float) for p in ((-2.2, -2.2, 0), (2.2, -2.2, 0), (2.2, 2.2, 0), (-2.2, 2.2, 0))],
             np.array([0, 0, 1.0]))
    boxes = [((0.0, 0.0, 0.5), (0.5, 0.5, 0.5)), ((0.9, -0.8, 0.3), (0.3, 0.35, 0.3)), ((-0.8, 0.9, 0.4), (0.35, 0.3, 0.4))]
    quads = [(floor, 0)] + [(f, 1 + 5 * b + k) for b, (c, s) in enumerate(boxes) for k, f in enumerate(_box_faces(c, s))]
    tex = {i: _texture(100 + i) for _, i in quads}
    T = 511.0
    src = np.float32([[0, 0], [T, 0], [T, T], [0, T]])
    names, centers = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        th = 2 * np.pi * i / n
        c = np.array([RADIUS * np.cos(th), RADIUS * np.sin(th), HEIGHT])
        R, t = _look_at(c, np.array([0, 0, 0.3]))
        img = np.full((H, W, 3), 128, np.uint8)
        visible = []
        for (pts, normal), ti in quads:
            mid = np.mean(pts, axis=0)
            if np.dot(normal, c - mid) <= 0:
                continue
            visible.append((np.linalg.norm(c - mid) if ti else 1e9, pts, ti))
        for _, pts, ti in sorted(visible, key=lambda v: -v[0]):
            Xc = np.array([R @ p + t for p in pts])
            uv = (K @ Xc.T).T
            dst = np.float32(uv[:, :2] / uv[:, 2:3])
            Hm = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(tex[ti], Hm, (W, H), flags=cv2.INTER_LINEAR)
            m = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST)
            img[m > 0] = warped[m > 0]
        name = f"{i:05d}.png"
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        names.append(name)
        centers.append(c)
    return names, np.array(centers)


def circle_residual(C: np.ndarray) -> float:
    """Relative RMS distance of points from their best-fit plane circle."""
    m = C.mean(axis=0)
    _, _, Vt = np.linalg.svd(C - m)
    P = (C - m) @ Vt[:2].T
    off = (C - m) @ Vt[2]
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P**2).sum(1), rcond=None)
    ctr = sol[:2]
    r = np.sqrt(sol[2] + ctr @ ctr)
    rad = np.linalg.norm(P - ctr, axis=1) - r
    return float(np.sqrt(np.mean(rad**2 + off**2)) / r)


def _orbit_pairs(n, window, closed):
    try:
        from gausify.pairs import orbit_pairs

        return orbit_pairs(n, window, closed)
    except ImportError:
        out = set()
        for i in range(n):
            for o in range(1, window + 1):
                if o <= n // 2:
                    j = (i + o) % n
                    out.add((min(i, j), max(i, j)))
        return sorted(out)


@pytest.fixture(scope="module")
def orbit(tmp_path_factory):
    d = tmp_path_factory.mktemp("orbit")
    names, centers = render_orbit(d / "images")
    return d, names, centers


def test_write_masks_convention(tmp_path):
    from gausify.sfm_colmap import write_masks, write_pairs

    md = tmp_path / "m"
    md.mkdir()
    cv2.imwrite(str(md / "00000.png"), np.full((4, 4), 255, np.uint8))
    out = write_masks(md, ["00000.png"], tmp_path / "cm")
    assert (out / "00000.png.png").is_file()
    p = write_pairs(["a.png", "b.png", "c.png"], [(0, 1), (0, 2)], tmp_path / "pairs.txt")
    assert p.read_text() == "a.png b.png\na.png c.png\n"


def test_circle_residual_sanity():
    th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    C = np.c_[np.cos(th), np.sin(th), np.zeros_like(th)] * 3 + 1
    assert circle_residual(C) < 1e-9
    C[:, 2] += np.random.default_rng(0).normal(0, 0.3, len(th))
    assert circle_residual(C) > 0.03


@needs_pycolmap
def test_colmap_poses_synthetic_orbit(orbit, colmap_venv):
    from gausify.sfm_colmap import colmap_poses_with_matches

    d, names, _ = orbit
    pairs = _orbit_pairs(len(names), 8, True)
    steps = []
    model, matches = colmap_poses_with_matches(d / "images", names, None, pairs, progress=lambda f, m: steps.append(f))
    s = model.stats
    assert s["total"] == N and s["mapper"] == "colmap" and s["backend"] in ("CPU", "GPU")
    assert s["registered"] >= int(0.8 * N), s
    assert len(model.poses) == N and len(model.xy) == N
    assert abs(model.f - F) / F < 0.1, model.f
    assert s["points"] > 100 and s["meanError"] < 2.0
    C = np.array([p.center for p in model.poses if p is not None])
    assert circle_residual(C) < 0.05
    # points reference valid keypoints
    for p in model.points[:200]:
        for i, k in p.obs:
            assert 0 <= k < len(model.xy[i])
    assert matches and all(m["a"] < m["b"] and m["idx"].dtype == np.int32 for m in matches)
    assert steps[0] == 0.0 and steps[-1] == 1.0 and steps == sorted(steps)


@needs_pycolmap
def test_colmap_poses_with_masks(orbit, tmp_path, colmap_venv):
    from gausify.sfm_colmap import colmap_poses

    d, names, _ = orbit
    md = tmp_path / "masks"
    md.mkdir()
    for n in names:  # ignore the left quarter of every frame
        m = np.full((H, W), 255, np.uint8)
        m[:, : W // 4] = 0
        cv2.imwrite(str(md / n), m)
    model = colmap_poses(d / "images", names, md, _orbit_pairs(len(names), 8, True))
    assert model.stats["registered"] >= int(0.8 * N)
    for xy in model.xy:
        assert (xy[:, 0] >= W // 4 - 1).all()


@needs_pycolmap
def test_run_sfm_colmap(orbit, tmp_path, colmap_venv):
    pytest.importorskip("gausify.pairs")
    pytest.importorskip("gausify.orbit")
    pytest.importorskip("gausify.colmap_io")
    from gausify.colmap_io import read_model
    from gausify.sfm import run_sfm

    d, names, _ = orbit
    sparse = tmp_path / "sparse" / "0"
    lines = []
    model = run_sfm(d / "images", names, None, "colmap", 8, True, True, True, None,
                    sparse_dir=sparse, log_fn=lines.append)
    assert (sparse / "images.bin").is_file()
    back = read_model(sparse)
    assert sum(p is not None for p in back.poses) == sum(p is not None for p in model.poses)
    assert any("COLMAP global mapper" in s for s in lines)
    assert model.stats.get("flat") == 1


# ---------------------------------------------------------------- fast tests (fake worker)

FAKE_WORKER = r"""
import json, sys, time, os
job = json.load(open(sys.argv[1]))
mode = job["mode"]
out = os.fdopen(os.dup(1), "w"); os.dup2(2, 1)
def emit(**e): out.write(json.dumps(e) + "\n"); out.flush()
print("glog noise", flush=True)
if mode == "ok":
    emit(type="progress", fraction=0.25, message="a")
    emit(type="log", message="hello")
    emit(type="progress", fraction=0.5, message="b")
    emit(type="result", backend="CPU", mapper="colmap", result="r.npz")
elif mode == "error":
    emit(type="error", message="ValueError: boom", traceback="Traceback...\nValueError: boom")
    sys.exit(1)
elif mode == "crash":
    print("segfault-ish stderr line", file=sys.stderr, flush=True)
    sys.exit(139)
elif mode == "hang":
    open(job["pidfile"], "w").write(str(os.getpid()))
    emit(type="progress", fraction=0.1, message="started")
    time.sleep(60)
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    from gausify import sfm_colmap

    w = tmp_path / "fake_worker.py"
    w.write_text(FAKE_WORKER)
    monkeypatch.setattr(sfm_colmap, "WORKER", w)

    def run(mode, **kw):
        job = tmp_path / "job.json"
        import json

        job.write_text(json.dumps({"mode": mode, "pidfile": str(tmp_path / "pid")}))
        return sfm_colmap.run_worker(job, python=sys.executable, **kw)

    return run


def test_worker_protocol_progress_and_log(fake):
    steps, logs = [], []
    res = fake("ok", progress=lambda f, m: steps.append((f, m)), log_fn=logs.append)
    assert steps == [(0.25, "a"), (0.5, "b")]
    assert logs == ["hello"]
    assert res["backend"] == "CPU" and res["result"] == "r.npz"


def test_worker_error_event(fake):
    from gausify.sfm_colmap import ColmapError

    with pytest.raises(ColmapError, match="ValueError: boom") as e:
        fake("error")
    assert "Traceback" in str(e.value) and "glog noise" in str(e.value)


def test_worker_crash_reports_stderr_tail(fake):
    from gausify.sfm_colmap import ColmapError

    with pytest.raises(ColmapError, match="exit code 139") as e:
        fake("crash")
    assert "segfault-ish stderr line" in str(e.value)


def test_worker_cancel_kills_process(fake, tmp_path):
    import time

    from gausify.sfm_colmap import Cancelled

    t0 = time.time()
    state = {"n": 0}

    def cancel():
        pid = tmp_path / "pid"
        return pid.exists() and pid.read_text() != "" and (state.update(n=state["n"] + 1) or state["n"] > 3)

    with pytest.raises(Cancelled):
        fake("hang", cancel=cancel)
    assert time.time() - t0 < 30
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(OSError):  # reaped: no such process
        os.kill(pid, 0)


def test_colmap_poses_does_not_import_pycolmap():
    import subprocess

    code = "import sys, gausify.sfm_colmap, gausify.sfm; assert 'pycolmap' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(REPO_DIR))


def test_to_model_from_arrays(tmp_path):
    from gausify.sfm_colmap import _matches, to_model

    cv2.imwrite(str(tmp_path / "a.png"), np.full((10, 20, 3), (10, 20, 30), np.uint8))  # BGR
    cv2.imwrite(str(tmp_path / "b.png"), np.zeros((10, 20, 3), np.uint8))
    poses = np.full((2, 3, 4), np.nan)
    poses[0] = np.c_[np.eye(3), [1, 2, 3]]
    res = {
        "has_rec": np.array(True),
        "camera": np.array([24.0, 10.0, 5.0]),
        "poses": poses,
        "xy_flat": np.array([[3.7, 4.2], [1, 1]], np.float32),
        "xy_counts": np.array([1, 1]),
        "xyz": np.array([[0, 0, 1.0]]),
        "error": np.array([0.5]),
        "track_counts": np.array([2]),
        "track_img": np.array([0, 1]),
        "track_kp": np.array([0, 0]),
        "match_ab": np.array([[0, 1]]),
        "match_counts": np.array([1]),
        "match_idx": np.array([[0, 0]], np.int32),
    }
    m = to_model(res, ["a.png", "b.png"], tmp_path, 20, 10)
    assert m.poses[1] is None and m.poses[0].t.tolist() == [1, 2, 3]
    assert m.points[0].rgb == (30, 20, 10) and m.points[0].obs == [(0, 0), (1, 0)]
    assert m.stats["registered"] == 1 and m.stats["observations"] == 2 and m.f == 24.0
    mm = _matches(res)
    assert mm[0]["a"] == 0 and mm[0]["idx"].dtype == np.int32


# ---------------------------------------------------------------- colmap_env


@pytest.fixture
def env(tmp_path, monkeypatch):
    from gausify import colmap_env

    venv = tmp_path / "models" / "venv"
    monkeypatch.setenv("GAUSIFY_COLMAP_VENV", str(venv))
    monkeypatch.delenv("GAUSIFY_COLMAP_PIP_ARGS", raising=False)
    calls = []

    def fake_create(path, tail, on_line):
        (path / "bin").mkdir(parents=True)
        (path / "bin" / "python").write_text("")

    def fake_run(cmd, tail, on_line):
        calls.append(cmd)
        on_line("Collecting pycolmap-cuda12")
        return 0

    monkeypatch.setattr(colmap_env, "_create_venv", fake_create)
    monkeypatch.setattr(colmap_env, "_run_logged", fake_run)
    return colmap_env, venv, calls


def test_env_location_precedence(tmp_path, monkeypatch):
    from gausify import colmap_env

    monkeypatch.setenv("GAUSIFY_COLMAP_VENV", str(tmp_path / "x"))
    assert colmap_env.venv_dir() == tmp_path / "x"
    monkeypatch.delenv("GAUSIFY_COLMAP_VENV")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert colmap_env.venv_dir() == tmp_path / ".cache" / "gausify" / "colmap-venv"


def test_env_builds_once_and_reuses(env):
    colmap_env, venv, calls = env
    steps = []
    py = colmap_env.ensure_colmap_env(lambda f, m: steps.append(m))
    assert py == venv / "bin" / "python" and py.is_file()
    assert len(calls) == 1 and calls[0][1:5] == ["-m", "pip", "install", "--disable-pip-version-check"]
    assert "pycolmap-cuda12==4.2.1" in calls[0] and "nvidia-curand-cu12" in calls[0]
    assert any("Collecting" in m for m in steps)
    assert not [p for p in venv.parent.iterdir() if p.name.startswith(".venv-")]  # temp dir renamed away
    colmap_env.ensure_colmap_env()
    assert len(calls) == 1


def test_env_rebuilds_when_spec_changes(env, monkeypatch):
    colmap_env, venv, calls = env
    colmap_env.ensure_colmap_env()
    (venv / "marker").write_text("old")
    monkeypatch.setenv("GAUSIFY_COLMAP_PIP_ARGS", "--index-url https://example.invalid/simple")
    colmap_env.ensure_colmap_env()
    assert len(calls) == 2 and "--index-url" in calls[1]
    assert not (venv / "marker").exists()
    colmap_env.ensure_colmap_env(packages=(*colmap_env.PACKAGES, "extra"))
    assert len(calls) == 3


def test_env_failure_message(env, monkeypatch):
    colmap_env, venv, _ = env

    def bad(cmd, tail, on_line):
        tail.append("ERROR: no matching distribution")
        return 1

    monkeypatch.setattr(colmap_env, "_run_logged", bad)
    with pytest.raises(colmap_env.ColmapEnvError) as e:
        colmap_env.ensure_colmap_env()
    msg = str(e.value)
    assert "no matching distribution" in msg and "-m pip install" in msg and "pycolmap-cuda12==4.2.1" in msg
    assert not venv.exists()
