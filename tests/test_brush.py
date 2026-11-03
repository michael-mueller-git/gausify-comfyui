"""gausify/brush.py against a fake brush-cli speaking the GAUSIFY protocol."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gausify import brush

FAKE = r"""
import json, os, sys, time
from pathlib import Path
import numpy as np

def emit(**ev):
    print("GAUSIFY " + json.dumps(ev), flush=True)

argv = sys.argv[1:]
mode = os.environ.get("FAKE_MODE", "ok")
log = os.environ.get("FAKE_LOG")
if log:
    Path(log).write_text(json.dumps(argv))

def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

if argv[0] == "render":
    cams = json.loads(Path(flag("--cameras")).read_text())
    out = Path(flag("--out-dir")); out.mkdir(parents=True, exist_ok=True)
    emit(event="loaded", splats=10, cameras=len(cams))
    for i, c in enumerate(cams):
        h, w = c["height"], c["width"]
        if "--no-color" not in argv:
            np.save(out / f"{i:05d}.rgba.npy", np.full((h, w, 4), c["fx"] if "fx" in c else c["f"], np.float32))
        np.save(out / f"{i:05d}.depth.npy", np.full((h, w), i, np.float32))
        np.save(out / f"{i:05d}.alpha.npy", np.ones((h, w), np.float32))
        emit(event="render", index=i, total=len(cams))
    emit(event="done", cameras=len(cams))
    sys.exit(0)

ds = Path(argv[0])
total = int(flag("--total-train-iters", "30000"))
if os.environ.get("FAKE_EXPECT_INIT") and (ds / "init.ply").read_text() != os.environ["FAKE_EXPECT_INIT"]:
    print("bad init", file=sys.stderr); sys.exit(3)
emit(event="config", total=total, export_every=int(flag("--export-every")), eval_every=1000)
emit(event="dataset", train=4, eval=1)
if mode == "fail":
    print("thread 'main' panicked: boom", file=sys.stderr)
    emit(event="error", message="Failed to load dataset")
    sys.exit(1)
stop = total // 2 if mode == "short" else total
for it in range(5, stop + 1, 5):
    emit(event="step", iter=it, total=total, splats=100 + it, seconds=it * 0.01)
    print(f"[INFO] log line {it}", flush=True)
    if mode == "slow":
        time.sleep(0.05)
emit(event="eval", iter=stop, psnr=27.5, ssim=0.9)
digits = len(str(total))
exp = Path(flag("--export-path")); exp.mkdir(parents=True, exist_ok=True)
ply = exp / flag("--export-name").replace("{iter}", f"{total:0{digits}d}")
if mode != "short":
    ply.write_text("ply " + str(total))
emit(event="done", iter=stop, total=total, splats=100 + stop, psnr=27.5, ssim=0.9, ply=str(ply), seconds=1.0)
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "brush-cli"
    exe.write_text(f"#!{sys.executable}\n" + FAKE)
    exe.chmod(0o755)
    monkeypatch.setenv("GAUSIFY_BRUSH_CLI", str(exe))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "argv.json"))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("FAKE_EXPECT_INIT", raising=False)
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    return tmp_path, ds


def _argv(tmp_path) -> list[str]:
    return json.loads((tmp_path / "argv.json").read_text())


def test_strip_owned():
    assert brush._strip_owned(["--a", "1", "--export-path", "x", "--export-every=5", "--export-name", "n", "--b"]) == [
        "--a",
        "1",
        "--b",
    ]


def test_train_ok(fake):
    tmp, ds = fake
    seen = []
    out = tmp / "res" / "splat.ply"
    r = brush.train(
        ds,
        ["--total-train-iters", "100", "--export-every", "7", "--export-path", "/bad"],
        out,
        progress=lambda f, m: seen.append(f),
    )
    assert r["psnr"] == 27.5 and r["splats"] == 200
    assert out.read_text() == "ply 100"
    argv = _argv(tmp)
    assert argv[0] == str(ds.resolve())
    assert "/bad" not in argv and argv.count("--export-every") == 1
    assert argv[argv.index("--export-every") + 1] == str(brush.NO_EXPORT_EVERY)
    assert not Path(argv[argv.index("--export-path") + 1]).is_relative_to(ds)
    assert seen[-1] == 1.0 and seen == sorted(seen) and len(seen) >= 20


def test_train_init_ply(fake, monkeypatch):
    tmp, ds = fake
    init = tmp / "prev.ply"
    init.write_text("previous splat")
    monkeypatch.setenv("FAKE_EXPECT_INIT", "previous splat")
    brush.train(ds, ["--total-train-iters", "10"], tmp / "o.ply", init_ply=init)
    assert not (ds / "init.ply").exists()
    # An existing init.ply is restored afterwards.
    (ds / "init.ply").write_text("mine")
    brush.train(ds, ["--total-train-iters", "10"], tmp / "o.ply", init_ply=init)
    assert (ds / "init.ply").read_text() == "mine"


def test_train_failure_has_stderr(fake, monkeypatch):
    tmp, ds = fake
    monkeypatch.setenv("FAKE_MODE", "fail")
    with pytest.raises(brush.BrushError) as e:
        brush.train(ds, ["--total-train-iters", "10"], tmp / "o.ply")
    assert "exit code 1" in str(e.value) and "Failed to load dataset" in str(e.value) and "boom" in str(e.value)


def test_train_must_reach_total(fake, monkeypatch):
    tmp, ds = fake
    monkeypatch.setenv("FAKE_MODE", "short")
    with pytest.raises(brush.BrushError, match="stopped at step 50 of 100"):
        brush.train(ds, ["--total-train-iters", "100"], tmp / "o.ply")
    assert not (tmp / "o.ply").exists()


def test_train_cancel(fake, monkeypatch):
    tmp, ds = fake
    monkeypatch.setenv("FAKE_MODE", "slow")
    t0 = time.monotonic()
    state = {"n": 0}

    def progress(f, m):
        state["n"] += 1

    with pytest.raises(brush.Cancelled):
        brush.train(
            ds, ["--total-train-iters", "10000"], tmp / "o.ply", progress=progress, cancel=lambda: state["n"] >= 3
        )
    assert time.monotonic() - t0 < 20


def test_render(fake):
    tmp, _ = fake
    ply = tmp / "s.ply"
    ply.write_text("x")
    cams = [
        {"R": np.eye(3), "t": np.zeros(3), "f": 50.0, "cx": 8, "cy": 6, "width": 16, "height": 12},
        {
            "R": np.eye(3).tolist(),
            "t": [0, 0, 1],
            "f": 50.0,
            "fx": 51.0,
            "fy": 49.0,
            "cx": 4,
            "cy": 4,
            "width": 8,
            "height": 8,
        },
    ]
    out = brush.render(ply, cams)
    assert len(out) == 2
    assert out[0]["rgba"].shape == (12, 16, 4) and out[0]["rgba"].dtype == np.float32
    assert out[0]["depth"].shape == (12, 16) and out[1]["depth"][0, 0] == 1
    assert out[1]["rgba"][0, 0, 0] == 51.0  # fx wins over f
    assert _argv(tmp)[0] == "render"
    no_color = brush.render(ply, cams, color=False)
    assert no_color[0]["rgba"] is None and "--no-color" in _argv(tmp)
    assert brush.render(ply, []) == []
