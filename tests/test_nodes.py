"""ComfyUI node definitions and wiring, without ComfyUI (folder_paths and the
gausify stages are stubbed)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gausify.presets import DEFAULT_PRESET, PRESET_NAMES, PRESETS, REDUCE_HAZE_WEIGHT
from gausify.types import OrbitCameras, OrbitTimes, Pose, SfmModel

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

import nodes  # noqa: E402
from nodes import common  # noqa: E402
from nodes.frame_select import GausifyFrameSelect  # noqa: E402
from nodes.poses import GausifyPoses  # noqa: E402
from nodes.train import GausifyTrainSplat, train_options  # noqa: E402


@pytest.fixture
def comfy_dirs(tmp_path, monkeypatch):
    out, tmp = tmp_path / "output", tmp_path / "temp"
    out.mkdir()
    tmp.mkdir()
    fp = types.SimpleNamespace(get_output_directory=lambda: str(out), get_temp_directory=lambda: str(tmp))
    monkeypatch.setattr(common, "folder_paths", fp)
    return out, tmp


def _defaults(cls):
    req = cls.INPUT_TYPES()["required"]
    return {k: (v[1]["default"] if len(v) > 1 and "default" in v[1] else None) for k, v in req.items()}


# ---------------------------------------------------------------- definitions


def test_mappings():
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["GausifyOrbitFrameSelect"] == "Gausify Orbit Frame Select"
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["GausifyOrbitPoses"] == "Gausify Orbit Poses"
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["GausifyOrbitTrainSplat"] == "Gausify Orbit Train Splat"
    for cls in nodes.NODE_CLASS_MAPPINGS.values():
        assert cls.CATEGORY == "Gausify Orbit"
        assert callable(getattr(cls, cls.FUNCTION))
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
        it = cls.INPUT_TYPES()
        assert it["required"]["preset"][0] == PRESET_NAMES
        assert it["required"]["preset"][1]["default"] == DEFAULT_PRESET


def test_root_package_loads():
    spec = importlib.util.spec_from_file_location(
        "comfyui_gausify_orbit_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        assert mod.WEB_DIRECTORY == "./web/js"
        assert set(mod.NODE_CLASS_MAPPINGS) == set(nodes.NODE_CLASS_MAPPINGS)
    finally:
        for k in [k for k in sys.modules if k.startswith(spec.name)]:
            del sys.modules[k]


def test_input_types_match_app_ranges():
    f = GausifyFrameSelect.INPUT_TYPES()
    assert f["required"]["frames"][1] | {} == {**f["required"]["frames"][1], "min": 20, "max": 600, "step": 10}
    assert (f["required"]["max_side"][1]["min"], f["required"]["max_side"][1]["max"]) == (480, 3840)
    assert f["required"]["candidates"][1]["max"] == 8
    assert f["optional"]["video"][0] == "VIDEO" and f["optional"]["images"][0] == "IMAGE"
    assert GausifyFrameSelect.RETURN_TYPES == ("IMAGE", "ORBIT_TIMES")

    p = GausifyPoses.INPUT_TYPES()
    assert p["required"]["estimator"][0] == ["akaze", "colmap"]
    assert p["required"]["estimator"][1]["default"] == "akaze"
    assert p["required"]["pose_check"][0] == ["off", "auto", "always"]
    assert (p["required"]["match_window"][1]["min"], p["required"]["match_window"][1]["max"]) == (3, 40)
    assert p["optional"]["masks"][0] == "MASK" and p["optional"]["frame_times"][0] == "ORBIT_TIMES"
    assert GausifyPoses.RETURN_TYPES == ("GAUSIFY_CAMERAS", "STRING", "STRING")

    t = GausifyTrainSplat.INPUT_TYPES()["required"]
    assert (t["steps"][1]["min"], t["steps"][1]["max"], t["steps"][1]["step"]) == (1000, 100000, 1000)
    assert (t["sh_degree"][1]["min"], t["sh_degree"][1]["max"]) == (0, 3)
    assert t["max_splats"][0] == "FLOAT" and t["max_splats"][1]["default"] == 2.0
    assert t["seed"][1]["default"] == 42
    assert GausifyTrainSplat.RETURN_TYPES == ("STRING", "INT", "FLOAT")


def test_defaults_are_balanced_preset():
    b = PRESETS["balanced"]
    d = _defaults(GausifyFrameSelect)
    assert (d["frames"], d["max_side"], d["candidates"]) == (160, 1600, 3)
    d = _defaults(GausifyPoses)
    assert (d["match_window"], d["closed_loop"]) == (b["poses"]["match_window"], True)
    d = _defaults(GausifyTrainSplat)
    assert d["steps"] == b["train"]["total_steps"] and d["ssim"] is False and d["psnr"] is True


# ---------------------------------------------------------------- presets


def test_apply_preset():
    v = {"steps": 1000, "sh_degree": 3, "max_splats": 0.1, "max_resolution": 480, "ssim": False, "psnr": False}
    assert common.apply_preset("train", "custom", v) == v
    q = common.apply_preset("train", "quality", v)
    assert q == {"steps": 30000, "sh_degree": 2, "max_splats": 3.0, "max_resolution": 1920, "ssim": True, "psnr": True}
    p = common.apply_preset("frames", "preview", {"frames": 600, "max_side": 3840, "candidates": 8})
    assert p == {"frames": 90, "max_side": 1280, "candidates": 3}
    assert common.apply_preset("poses", "preview", {"match_window": 3, "closed_loop": False}) == {
        "match_window": 8, "closed_loop": True}


def test_presets_js_table_matches_python():
    js = (ROOT / "web" / "js" / "presets.js").read_text()
    m = re.search(r"const PRESETS = (\{.*?\});\n// END PRESETS", js, re.S)
    assert m, "PRESETS table not found in presets.js"
    assert json.loads(m.group(1)) == common.presets_for_js()
    assert "registerExtension" in js and "nodeCreated" in js


def test_train_options():
    v = {"steps": 7000, "sh_degree": 2, "max_splats": 1.5, "max_resolution": 960, "ssim": True, "psnr": False,
         "reduce_haze": True}
    o = train_options(v, masked=True)
    assert o == {"total_steps": 7000, "sh_degree": 2, "max_splats": 1_500_000, "max_resolution": 960,
                 "export_every": 7000, "eval_split_every": 0, "ssim_weight": 0.2,
                 "depth_distortion_weight": REDUCE_HAZE_WEIGHT}
    assert "depth_distortion_weight" not in train_options(v, masked=False)


def test_next_ply_path(tmp_path):
    p, rel = common.next_ply_path("gausify/orbit", tmp_path)
    assert rel == "gausify/orbit_00001.ply" and p.parent.is_dir()
    p.write_text("x")
    _, rel = common.next_ply_path("gausify/orbit", tmp_path)
    assert rel == "gausify/orbit_00002.ply"
    with pytest.raises(ValueError):
        common.next_ply_path("../escape", tmp_path)


# ---------------------------------------------------------------- wiring


def _fake_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg = sys.modules["gausify"]
    monkeypatch.setattr(pkg, name.split(".")[-1], mod, raising=False)
    return mod


def test_frame_select_images(monkeypatch, comfy_dirs):
    calls = {}

    def select_frames(frames, fps, target, max_side, oversample, trim_start, trim_end, progress=None, cancel=None):
        calls.update(shape=frames.shape, fps=fps, target=target, max_side=max_side, oversample=oversample,
                     trim=(trim_start, trim_end))
        return frames[::2], OrbitTimes([float(i) for i in range(0, len(frames), 2)], from_images=fps is None)

    _fake_module(monkeypatch, "gausify.frames", select_frames=select_frames)
    imgs = torch.rand(6, 8, 10, 3)
    out, times = GausifyFrameSelect().select("preview", 600, 3840, 8, 0.0, -1.0, images=imgs, fps=0.0)
    assert calls == {"shape": (6, 8, 10, 3), "fps": None, "target": 90, "max_side": 1280, "oversample": 3,
                     "trim": (0.0, None)}
    assert tuple(out.shape) == (3, 8, 10, 3) and out.dtype == torch.float32
    assert times.from_images


def test_frame_select_video(monkeypatch, comfy_dirs):
    from fractions import Fraction

    seen = {}
    _fake_module(monkeypatch, "gausify.frames",
                 select_frames=lambda f, fps, *a, **kw: (seen.update(fps=fps, a=a) or f, OrbitTimes([0.0] * len(f))))

    class Video:
        def get_components(self):
            return types.SimpleNamespace(images=torch.rand(4, 6, 6, 3), audio=None, frame_rate=Fraction(30000, 1001))

    GausifyFrameSelect().select("custom", 40, 640, 2, 1.5, 3.0, video=Video())
    assert abs(seen["fps"] - 29.97) < 0.01 and seen["a"] == (40, 640, 2, 1.5, 3.0)
    with pytest.raises(ValueError):
        GausifyFrameSelect().select("custom", 40, 640, 2, 0.0, -1.0)


def _fake_model(names):
    n = len(names)
    poses = [Pose(np.eye(3), np.array([0.0, 0.0, float(i)])) for i in range(n)]
    return SfmModel(f=100.0, cx=5.0, cy=4.0, width=10, height=8, poses=poses, points=[], names=list(names),
                    xy=[np.zeros((0, 2), np.float32)] * n,
                    stats={"registered": n, "total": n, "points": 0, "meanError": 0.5, "mapper": "colmap"})


def test_poses_wiring_and_cache(monkeypatch, comfy_dirs):
    import gausify.posecheck
    import gausify.sfm

    calls = []

    def run_sfm(images_dir, names, masks_dir, estimator, window, closed_loop, flat_orbit, fill_missing, times,
                progress=None, cancel=None, sparse_dir=None, log_fn=None):
        calls.append(dict(images_dir=Path(images_dir), names=names, masks_dir=masks_dir, estimator=estimator,
                          window=window, closed=closed_loop, flat=flat_orbit, fill=fill_missing, times=times))
        assert all((Path(images_dir) / n).is_file() for n in names)
        Path(sparse_dir).mkdir(parents=True)
        (Path(sparse_dir) / "images.bin").write_bytes(b"x")
        log_fn("COLMAP global mapper: ok")
        progress(1.0, "done")
        assert cancel() is False
        return _fake_model(names)

    checks = []
    monkeypatch.setattr(gausify.sfm, "run_sfm", run_sfm)
    monkeypatch.setattr(gausify.posecheck, "check_poses",
                        lambda cams, mode, progress=None, cancel=None, **kw: checks.append((cams, mode, kw)) or {"score": 0.9})

    imgs = torch.rand(5, 8, 10, 3)
    masks = torch.zeros(5, 8, 10)
    masks[:, 2:6, 3:7] = 1.0
    times = OrbitTimes([0.0, 0.5, 1.0, 1.5, 2.0])
    node = GausifyPoses()
    cams, path, stats = node.poses(imgs, "quality", "colmap", 3, False, True, True, True, "auto",
                                   frame_times=times, masks=masks)
    assert isinstance(cams, OrbitCameras) and cams.masked and (cams.width, cams.height) == (10, 8)
    assert cams.names == [f"{i:05d}.png" for i in range(5)]
    assert Path(path) == cams.root and Path(path).parent.parent == comfy_dirs[1]
    c = calls[0]
    assert (c["estimator"], c["window"], c["closed"], c["flat"], c["fill"]) == ("colmap", 20, True, True, True)
    assert c["masks_dir"] == cams.masks_dir and c["times"] == times.times
    m = (np.asarray(__import__("cv2").imread(str(cams.masks_dir / "00000.png"), 0)))
    assert m.shape == (8, 10) and m[3, 4] == 255 and m[0, 0] == 0
    s = json.loads(stats)
    assert s["registered"] == 5 and s["check"] == {"score": 0.9} and s["estimator"] == "colmap"
    assert checks and checks[0][1] == "auto"

    # same inputs and options: cached, no new run
    node.poses(imgs, "quality", "colmap", 3, False, True, True, True, "auto", frame_times=times, masks=masks)
    assert len(calls) == 1
    # without masks: no mask dir, no pose check
    cams2, _, s2 = node.poses(imgs, "custom", "akaze", 5, False, False, False, True, "always")
    assert len(calls) == 2 and calls[1]["masks_dir"] is None and calls[1]["window"] == 5
    assert calls[1]["closed"] is False and not cams2.masked and len(checks) == 1
    # image-sequence frames: their source indices are not passed as times
    node.poses(imgs, "custom", "akaze", 6, False, False, False, True, "off",
               frame_times=OrbitTimes([0.0, 2.0, 3.0, 7.0, 9.0], from_images=True))
    assert len(calls) == 3 and calls[2]["times"] is None


def test_poses_too_few_registered(monkeypatch, comfy_dirs):
    import gausify.sfm

    def run_sfm(images_dir, names, *a, sparse_dir=None, **kw):
        m = _fake_model(names)
        m.stats["registered"] = 2
        return m

    monkeypatch.setattr(gausify.sfm, "run_sfm", run_sfm)
    with pytest.raises(RuntimeError, match="only 2 cameras"):
        GausifyPoses().poses(torch.rand(4, 8, 8, 3), "balanced", "akaze", 20, True, False, False, False, "off")


def test_train_wiring(monkeypatch, comfy_dirs, tmp_path):
    out_dir, _ = comfy_dirs
    seen = {}

    def train_config(opts, masked):
        seen["opts"], seen["masked"] = opts, masked
        return {"total-train-iters": opts["total_steps"]}

    def to_cli_args(cfg):
        seen["cfg"] = cfg
        return [f"--{k}={v}" for k, v in cfg.items()]

    def train(dataset_dir, args, out_ply, init_ply=None, progress=None, cancel=None):
        seen.update(dataset=Path(dataset_dir), args=args, out=Path(out_ply))
        Path(out_ply).write_text("ply")
        progress(0.5, "half")
        return {"psnr": 31.5, "splats": 1234}

    def build_alpha_dataset(cams, out_dir, despill=False, despill_strength=50.0, log=None, progress=None, cancel=None):
        seen["alpha"] = (despill, despill_strength)
        Path(out_dir).mkdir(parents=True)
        return Path(out_dir)

    _fake_module(monkeypatch, "gausify.train_config", train_config=train_config, to_cli_args=to_cli_args)
    _fake_module(monkeypatch, "gausify.brush", train=train)
    _fake_module(monkeypatch, "gausify.alpha", build_alpha_dataset=build_alpha_dataset)

    root = tmp_path / "ds"
    (root / "sparse" / "0").mkdir(parents=True)
    (root / "sparse" / "0" / "images.bin").write_bytes(b"x")
    cams = OrbitCameras(root=root, names=["00000.png"], width=10, height=8, masked=False)
    node = GausifyTrainSplat()
    r = node.train(cams, "preview", 99000, 3, 9.0, 3840, True, True, False, True, 100, 7, "splats/test")
    rel, n, psnr = r["result"]
    assert rel == "splats/test_00001.ply" and (out_dir / rel).is_file() and seen["out"] == out_dir / rel
    assert (n, psnr) == (1234, 31.5)
    assert seen["dataset"] == root and seen["masked"] is False and "alpha" not in seen
    assert seen["opts"]["total_steps"] == 5000 and seen["opts"]["export_every"] == 5000
    assert seen["opts"]["max_splats"] == 1_000_000 and seen["opts"]["eval_split_every"] == 16
    assert "depth_distortion_weight" not in seen["opts"]
    assert seen["cfg"]["seed"] == 7 and seen["cfg"]["export-every"] == 5000

    cams.masked = True
    r = node.train(cams, "custom", 2000, 0, 0.5, 480, False, True, False, False, 30, 1, "splats/test")
    assert r["result"][0] == "splats/test_00002.ply"
    assert seen["alpha"] == (False, 30.0) and seen["dataset"] != root and seen["masked"] is True
    assert seen["opts"]["depth_distortion_weight"] == REDUCE_HAZE_WEIGHT
    assert seen["opts"]["eval_split_every"] == 0 and seen["opts"]["max_splats"] == 500_000


def test_example_workflow_matches_nodes():
    wf = json.loads((ROOT / "workflows" / "gausify_orbit.json").read_text())
    by_type = {n["type"]: n for n in wf["nodes"]}
    links = {link[0]: link for link in wf["links"]}
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        n = by_type[name]
        it = cls.INPUT_TYPES()
        widgets = [k for sect in ("required", "optional") for k, v in it.get(sect, {}).items()
                   if isinstance(v[0], list) or v[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")]
        extra = sum(1 for k in widgets if k == "seed")  # control_after_generate
        assert len(n["widgets_values"]) == len(widgets) + extra, name
        assert [o["type"] for o in n["outputs"]] == list(cls.RETURN_TYPES)
        for inp in n["inputs"]:
            if inp["link"] is not None:
                assert links[inp["link"]][5] == inp["type"]
    assert by_type["Preview3D"]["inputs"][0]["link"] in links
