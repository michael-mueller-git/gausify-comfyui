"""Presets, 1:1 with src/lib/presets.ts of gausify-orbit. Default preset: balanced."""

REDUCE_HAZE_WEIGHT = 10.0
DEFAULT_PRESET = "balanced"

PRESETS = {
    "preview": {
        "frames": {"target_frames": 90, "max_side": 1280, "oversample": 3},
        "poses": {"match_window": 8, "closed_loop": True},
        "train": {
            "total_steps": 5000,
            "sh_degree": 1,
            "max_splats": 1_000_000,
            "max_resolution": 1280,
            "export_every": 1000,
            "eval_split_every": 16,
            "ssim_weight": 0.0,
        },
    },
    "balanced": {
        "frames": {"target_frames": 160, "max_side": 1600, "oversample": 3},
        "poses": {"match_window": 20, "closed_loop": True},
        "train": {
            "total_steps": 15000,
            "sh_degree": 1,
            "max_splats": 2_000_000,
            "max_resolution": 1600,
            "export_every": 2500,
            "eval_split_every": 16,
            "ssim_weight": 0.0,
        },
    },
    "quality": {
        "frames": {"target_frames": 220, "max_side": 1920, "oversample": 4},
        "poses": {"match_window": 20, "closed_loop": True},
        "train": {
            "total_steps": 30000,
            "sh_degree": 2,
            "max_splats": 3_000_000,
            "max_resolution": 1920,
            "export_every": 5000,
            "eval_split_every": 16,
            "ssim_weight": 0.2,
        },
    },
}

PRESET_NAMES = ["preview", "balanced", "quality", "custom"]

# web/src/sfm/index.ts DEFAULT_SFM
SFM_MAX_SIDE = 1600
SFM_MAX_FEATURES = 4000
