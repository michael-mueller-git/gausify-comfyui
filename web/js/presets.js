// Gausify Orbit: when a node's "preset" changes, its widgets take the preset's
// values (same table as gausify/presets.py, see nodes/common.py
// presets_for_js); editing one of those widgets by hand switches the preset to
// "custom". The Python side applies the preset anyway when it runs.
import { app } from "../../scripts/app.js";

// BEGIN PRESETS (generated from gausify/presets.py; tests/test_nodes.py checks it)
const PRESETS = {
  "preview": {
    "frames": {
      "frames": 90,
      "max_side": 1280,
      "candidates": 3
    },
    "poses": {
      "match_window": 8,
      "closed_loop": true
    },
    "train": {
      "steps": 5000,
      "sh_degree": 1,
      "max_splats": 1.0,
      "max_resolution": 1280,
      "ssim": false,
      "psnr": true
    }
  },
  "balanced": {
    "frames": {
      "frames": 160,
      "max_side": 1600,
      "candidates": 3
    },
    "poses": {
      "match_window": 20,
      "closed_loop": true
    },
    "train": {
      "steps": 15000,
      "sh_degree": 1,
      "max_splats": 2.0,
      "max_resolution": 1600,
      "ssim": false,
      "psnr": true
    }
  },
  "quality": {
    "frames": {
      "frames": 220,
      "max_side": 1920,
      "candidates": 4
    },
    "poses": {
      "match_window": 20,
      "closed_loop": true
    },
    "train": {
      "steps": 30000,
      "sh_degree": 2,
      "max_splats": 3.0,
      "max_resolution": 1920,
      "ssim": true,
      "psnr": true
    }
  }
};
// END PRESETS

const STAGE_BY_NODE = {
  GausifyOrbitFrameSelect: "frames",
  GausifyOrbitPoses: "poses",
  GausifyOrbitTrainSplat: "train",
};

const same = (a, b) => (typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-9 : a === b);

function applyPreset(node, stage, name) {
  const values = PRESETS[name]?.[stage];
  if (!values) return;
  for (const w of node.widgets ?? []) {
    if (w.name in values) w.value = values[w.name];
  }
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "gausify.orbit.presets",
  nodeCreated(node) {
    const stage = STAGE_BY_NODE[node.comfyClass];
    if (!stage) return;
    const preset = node.widgets?.find((w) => w.name === "preset");
    if (!preset) return;
    const keys = new Set(Object.keys(PRESETS.balanced[stage]));
    let applying = false;

    const onPreset = preset.callback;
    preset.callback = function (value, ...rest) {
      const r = onPreset?.apply(this, [value, ...rest]);
      applying = true;
      try {
        applyPreset(node, stage, preset.value);
      } finally {
        applying = false;
      }
      return r;
    };

    for (const w of node.widgets) {
      if (!keys.has(w.name)) continue;
      const cb = w.callback;
      w.callback = function (value, ...rest) {
        const r = cb?.apply(this, [value, ...rest]);
        const p = PRESETS[preset.value]?.[stage];
        if (!applying && p && !same(p[w.name], w.value)) {
          preset.value = "custom";
          node.setDirtyCanvas?.(true, true);
        }
        return r;
      };
    }
  },
});
