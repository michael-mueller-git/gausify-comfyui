"""Port of web/src/pipeline/train-config.test.ts + CLI formatting."""

import re

from gausify.presets import PRESETS, REDUCE_HAZE_WEIGHT
from gausify.train_config import to_cli_args, train_config

OPTS = {
    "total_steps": 15000,
    "sh_degree": 1,
    "max_splats": 2_000_000,
    "max_resolution": 1920,
    "export_every": 2500,
    "eval_split_every": 0,
    "ssim_weight": 0,
    "opac_binary_weight": 0.01,
    "depth_distortion_weight": REDUCE_HAZE_WEIGHT,
}


def test_random_background_and_distortion_only_with_masks():
    plain = train_config(OPTS, False)
    assert plain["total-train-iters"] == 15000
    assert plain["opac-binary-weight"] == 0.01
    for k in plain:
        assert not re.search(r"background|distortion|geom-reg|eval-every", k), k
    masked = train_config(OPTS, True)
    assert masked["background-color"] == [0.5, 0.5, 0.5]
    assert masked["background-noise-strength"] == 0.5
    assert masked["depth-distortion-weight"] == REDUCE_HAZE_WEIGHT
    assert masked["geom-reg-start"] == 0.25
    assert masked["geom-reg-every"] == 2
    assert masked["geom-reg-downscale"] == 1
    assert masked["geom-reg-crop"] is True


def test_no_distortion_keys_when_weight_zero_or_missing():
    for w in (0, None):
        o = {**OPTS, "depth_distortion_weight": w}
        if w is None:
            del o["depth_distortion_weight"]
        for k in train_config(o, True):
            assert not re.search(r"distortion|geom-reg", k), k


def test_presets_reduce_haze_off():
    for name, p in PRESETS.items():
        assert "depth_distortion_weight" not in p["train"], name


def test_removed_scale_loss_weight_ignored():
    cfg = train_config({**OPTS, "scale_loss_weight": 1e-6}, True)
    assert not any("scale-loss" in k for k in cfg)


def test_eval_every_with_eval_split():
    cfg = train_config({**PRESETS["balanced"]["train"]}, False)
    assert cfg["eval-split-every"] == 16 and cfg["eval-every"] == 2500
    assert train_config(OPTS, False)["eval-split-every"] is None
    assert train_config({**OPTS, "sh_degree": 5, "ssim_weight": -1}, False)["sh-degree"] == 3


def test_cli_args():
    args = to_cli_args(train_config(OPTS, True))
    assert args[:4] == ["--total-train-iters", "15000", "--sh-degree", "1"]
    assert "--eval-split-every" not in args  # None omitted
    i = args.index("--background-color")
    assert args[i + 1 : i + 4] == ["0.5", "0.5", "0.5"]
    i = args.index("--geom-reg-crop")
    assert args[i + 1] == "true"
    assert args[args.index("--depth-distortion-weight") + 1] == "10"
    assert args[args.index("--opac-binary-weight") + 1] == "0.01"
    assert args[args.index("--ssim-weight") + 1] == "0"
    assert to_cli_args({"lr-mean": 1e-6, "x": None}) == ["--lr-mean", "1e-06"]
