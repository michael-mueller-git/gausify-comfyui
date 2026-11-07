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


# ------------------------------------------------------------ advanced settings
#
# Brush options the app leaves at Brush's defaults, exposed by the "Train
# Advanced" node. Defaults, ranges and meaning from brush-train/src/config.rs
# at the pinned Brush commit (with gausify-orbit's patches: contribution
# pruning, DeG regularisers, geometry schedule). Each entry:
# name -> (kind, Brush default, min, max, step, tooltip). Learning rates are
# exposed as multipliers of Brush's defaults ("lr_*_mult", key in LR_MULT).

#: lr multiplier -> the Brush keys it scales (with their defaults).
LR_MULT = {
    "lr_mean_mult": {"lr-mean": 2e-5, "lr-mean-end": 2e-7},
    "lr_coeffs_mult": {"lr-coeffs-dc": 2e-3},
    "lr_opac_mult": {"lr-opac": 0.012},
    "lr_scale_mult": {"lr-scale": 5e-3},
    "lr_rotation_mult": {"lr-rotation": 2e-3},
}

ADVANCED = {
    # growth / refinement
    "refine_every": ("INT", 200, 1, 5000, 10,
                     "Steps between refinements (splats grown, pruned, relocated). Roughly the number of "
                     "images needed to cover the subject."),
    "growth_grad_threshold": ("FLOAT", 0.0025, 0.0001, 0.05, 0.0001,
                              "Splat growth threshold: lower grows more splats (finer detail, more memory)."),
    "growth_select_fraction": ("FLOAT", 0.25, 0.01, 1.0, 0.01,
                               "Fraction of the splats due to grow that actually grow."),
    "growth_stop_iter": ("INT", 15000, 0, 100000, 500,
                         "Step at which splat growth stops; the rest of training only refines."),
    "split_at_screen_size": ("FLOAT", 0.5, 0.0, 1.0, 0.01,
                             "Split splats larger than this fraction of the image (0 = off)."),
    "opac_decay": ("FLOAT", 0.004, 0.0, 0.1, 0.001, "Opacity decay factor."),
    "min_scale_factor": ("FLOAT", 0.1, 0.0, 2.0, 0.01,
                         "Mip-Splatting 3D filter strength (anti-aliasing when zooming out; 0 = off)."),
    "match_alpha_weight": ("FLOAT", 0.1, 0.0, 10.0, 0.01,
                           "L1 weight on the alpha of masked (cut-out) frames: higher = sharper silhouette."),
    "mean_noise_weight": ("FLOAT", 50.0, 0.0, 500.0, 1.0,
                          "Noise added to the positions of low-opacity splats."),
    # contribution pruning
    "contrib_prune_min_views": ("INT", 0, 0, 100, 1,
                                "Prune splats that hurt the loss (ghosts, doubled edges) once they were seen in "
                                "at least this many views (0 = off)."),
    "contrib_prune_ratio": ("FLOAT", 0.2, 0.0, 1.0, 0.01,
                            "How clearly harmful a splat must be to be pruned (0 = any net loss)."),
    "contrib_prune_start": ("FLOAT", 0.2, 0.0, 1.0, 0.01,
                            "Fraction of training before contribution pruning starts."),
    # regularisers
    "opac_solid_weight": ("FLOAT", 0.0, 0.0, 1.0, 0.001,
                          "Push splats towards opaque (less fog; can harm hair and other thin, soft detail)."),
    "opac_binary_weight": ("FLOAT", 0.0, 0.0, 1.0, 0.001, "Push opacities to 0 or 1."),
    "volume_reg_weight": ("FLOAT", 0.0, 0.0, 1.0, 0.001,
                          "Keep splats compact (fewer blobs smeared across unseen angles)."),
    "flatten_weight": ("FLOAT", 0.0, 0.0, 1.0, 0.001, "Flatten splats into disks (surfaces)."),
    "normal_consistency_weight": ("FLOAT", 0.0, 0.0, 1.0, 0.001,
                                  "2DGS normal consistency (one extra render per step). Meant for masked training."),
    "geom_reg_end": ("FLOAT", 1.0, 0.0, 1.0, 0.01,
                     "Fraction of training after which depth distortion / normal consistency stop."),
    # learning rates (multipliers of Brush's defaults)
    "lr_mean_mult": ("FLOAT", 1.0, 0.01, 10.0, 0.01, "Position learning rate (start and end) x this."),
    "lr_coeffs_mult": ("FLOAT", 1.0, 0.01, 10.0, 0.01, "Colour learning rate x this."),
    "lr_opac_mult": ("FLOAT", 1.0, 0.01, 10.0, 0.01, "Opacity learning rate x this."),
    "lr_scale_mult": ("FLOAT", 1.0, 0.01, 10.0, 0.01, "Scale learning rate x this."),
    "lr_rotation_mult": ("FLOAT", 1.0, 0.01, 10.0, 0.01, "Rotation learning rate x this."),
}


#: Train Advanced "loss": the photometric loss. l1_ssim = Brush's (and the
#: app's): L1 + SSIM (with the Train Splat ssim toggle). l1_ssim_lpips adds
#: LPIPS (VGG, weights built into brush-cli) x lpips_weight: sharper hair,
#: fabric and skin texture; slower steps and more GPU memory.
LOSSES = ["l1_ssim", "l1_ssim_lpips"]
DEFAULT_LPIPS_WEIGHT = 0.1

#: Train Advanced "render_mode": Brush's rasteriser. mip = Mip-Splatting's 2D
#: anti-aliasing filter (with opacity compensation); recorded in the .ply.
RENDER_MODES = ["default", "mip"]

#: Brush keys of the masked-training background: only passed when masked.
MASKED_ONLY = ("background-color", "background-noise-strength")


def merge_advanced(cfg: dict, advanced: dict | None, masked: bool) -> dict:
    """`cfg` updated with the Train Advanced settings; the masked-training
    background only when `masked` (unmasked training keeps Brush's)."""
    for k, v in (advanced or {}).items():
        if masked or k not in MASKED_ONLY:
            cfg[k] = v
    return cfg


def advanced_config(values: dict) -> dict:
    """Brush keys (kebab-case) for the advanced settings that differ from
    Brush's defaults; unknown names raise. An empty dict = Brush's defaults."""
    out: dict = {}
    values = dict(values)
    loss = values.pop("loss", LOSSES[0])
    lpips_weight = float(values.pop("lpips_weight", DEFAULT_LPIPS_WEIGHT))
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r} (expected one of {LOSSES})")
    if loss == "l1_ssim_lpips" and lpips_weight > 0:
        out["lpips-loss-weight"] = lpips_weight
    mode = values.pop("render_mode", RENDER_MODES[0])
    if mode not in RENDER_MODES:
        raise ValueError(f"unknown render_mode {mode!r} (expected one of {RENDER_MODES})")
    if mode != "default":
        out["render-mode"] = mode
    grey = float(values.pop("masked_bg_grey", MASKED_BACKGROUND["color"][0]))
    noise = float(values.pop("masked_bg_noise", MASKED_BACKGROUND["noise"]))
    if (grey, noise) != (MASKED_BACKGROUND["color"][0], MASKED_BACKGROUND["noise"]):
        out["background-color"] = [grey, grey, grey]
        out["background-noise-strength"] = noise
    for name, v in values.items():
        if name not in ADVANCED:
            raise KeyError(f"unknown advanced setting {name!r}")
        kind, default = ADVANCED[name][:2]
        v = int(v) if kind == "INT" else float(v)
        if abs(v - default) <= 1e-12 * max(1.0, abs(default)):
            continue
        if name in LR_MULT:
            for key, base in LR_MULT[name].items():
                out[key] = base * v
        else:
            out[name.replace("_", "-")] = v
    return out
