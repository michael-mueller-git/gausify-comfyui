"""Brush training config for the node's train options.

Port of trainConfig() in gausify-orbit web/src/pipeline/train-config.ts, plus
`to_cli_args` for brush-cli (clap): `--<key> <value>`; `background-color` is
`Vec<f32>` with `value_delimiter = ','` and `num_args = 3` in
brush-train/src/config.rs, emitted as three separate values; `geom-reg-crop`
is a bool with `ArgAction::Set` (brush/patches/geom.patch), emitted as
`--geom-reg-crop true`. None values are omitted.

Input keys (snake_case, as the `train` section of gausify/presets.py, plus
the optional TrainOptions of src/lib/api.ts): total_steps, sh_degree,
max_splats, max_resolution, export_every, eval_split_every, ssim_weight,
contrib_prune_min_views, contrib_prune_ratio, contrib_prune_start,
opac_solid_weight, opac_binary_weight, volume_reg_weight,
depth_distortion_weight, lr_mean, growth_stop_iter.
"""

from __future__ import annotations

#: Background for masked training: a uniform color drawn every step (0.5 +- 0.5).
MASKED_BACKGROUND = {"color": [0.5, 0.5, 0.5], "noise": 0.5}

#: When depth distortion ("Reduce haze") runs during masked training.
DISTORTION_SCHEDULE = {"start": 0.25, "every": 2, "downscale": 1}


def train_config(o: dict, masked: bool) -> dict:
    """The fields Gausify Orbit controls (kebab-case Brush keys); the rest keep
    Brush's defaults. `masked`: the images carry the mask alpha."""
    g = o.get
    cfg: dict = {
        "total-train-iters": o["total_steps"],
        "sh-degree": min(o["sh_degree"], 3),
        "max-splats": o["max_splats"],
        "max-resolution": o["max_resolution"],
        "ssim-weight": max(o["ssim_weight"], 0),
        "eval-split-every": g("eval_split_every") or None,
    }
    # Only with evaluation on: Brush computes iter % eval-every even without
    # eval views, and 0 there crashes training.
    if g("eval_split_every"):
        cfg["eval-every"] = o["export_every"]
    if g("contrib_prune_min_views"):
        cfg["contrib-prune-min-views"] = o["contrib_prune_min_views"]
    if g("contrib_prune_ratio") is not None:
        cfg["contrib-prune-ratio"] = o["contrib_prune_ratio"]
    if g("contrib_prune_start") is not None:
        cfg["contrib-prune-start"] = o["contrib_prune_start"]
    if g("opac_solid_weight"):
        cfg["opac-solid-weight"] = o["opac_solid_weight"]
    if g("opac_binary_weight"):
        cfg["opac-binary-weight"] = o["opac_binary_weight"]
    if g("volume_reg_weight"):
        cfg["volume-reg-weight"] = o["volume_reg_weight"]
    if g("lr_mean") is not None:
        cfg["lr-mean"] = o["lr_mean"]
    if g("growth_stop_iter") is not None:
        cfg["growth-stop-iter"] = o["growth_stop_iter"]
    if masked:
        cfg["background-color"] = list(MASKED_BACKGROUND["color"])
        cfg["background-noise-strength"] = MASKED_BACKGROUND["noise"]
        # Without masks it stays at Brush's 0 (fog sheets collapse training).
        w = g("depth_distortion_weight") or 0
        if w > 0:
            cfg["depth-distortion-weight"] = w
            cfg["geom-reg-start"] = DISTORTION_SCHEDULE["start"]
            cfg["geom-reg-every"] = DISTORTION_SCHEDULE["every"]
            cfg["geom-reg-downscale"] = DISTORTION_SCHEDULE["downscale"]
            cfg["geom-reg-crop"] = True
    return cfg


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # integral floats as integers: valid for both u32 and f32 clap fields
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v)


def to_cli_args(cfg: dict) -> list[str]:
    """brush-cli arguments for a train_config() dict, in its key order."""
    out: list[str] = []
    for k, v in cfg.items():
        if v is None:
            continue
        out.append(f"--{k}")
        if isinstance(v, (list, tuple)):
            out.extend(_fmt(x) for x in v)
        else:
            out.append(_fmt(v))
    return out
