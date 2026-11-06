# Gausify Orbit for ComfyUI

Turn an orbit video (a camera circling a subject, e.g. an AI-generated
turntable) into a 3D Gaussian splat, inside ComfyUI. The nodes reproduce the
pipeline and output of [gausify-orbit](https://github.com/obraia/gausify-orbit):
sharp-frame selection, camera poses (AKAZE + incremental SfM, or COLMAP's SIFT
+ global mapper), optional masked (cut-out) training, and training with
[Brush](https://github.com/ArthurBrussee/brush).

**NVIDIA / CUDA only, Linux x86-64 only.**

## Nodes (category "Gausify Orbit")

| Node | Inputs | Outputs |
|---|---|---|
| **Gausify Orbit Frame Select** | `video` (core VIDEO) or `images` + `fps` (0 = image sequence); preset, frames, max_side, candidates, trim_start, trim_end (-1 = end) | IMAGE, ORBIT_TIMES |
| **Gausify Orbit Mask (Solid Colour)** | images; auto_color, color, tolerance, softness, contiguous | MASK, stats (JSON) |
| **Gausify Orbit Mask (RMBG-1.4)** | images; accept_noncommercial_license, device | MASK, stats (JSON) |
| **Gausify Orbit Mask (SAM 2.1)** | images; prompt_frame, box, points, variant (`fp32`/`fp16`), device | MASK, stats (JSON) |
| **Gausify Orbit Poses** | images, frame_times, masks (optional); preset, estimator (`akaze`/`colmap`), match_window, closed_loop, flat_orbit, fill_missing, use_mask_for_poses, pose_check (`off`/`auto`/`always`, needs masks) | GAUSIFY_CAMERAS, colmap_path, stats (JSON) |
| **Gausify Orbit Train Splat** | cameras; preset, steps, sh_degree, max_splats (millions), max_resolution, ssim, reduce_haze (masked only), psnr, despill + strength (masked only), seed, filename_prefix | ply_path (relative to the output folder), num_splats, psnr |

`preset` (`preview`, `balanced`, `quality`, `custom`) is the app's quality
preset: anything but `custom` overrides the node's preset-controlled widgets
(the UI updates them when you change the preset; editing one switches to
`custom`). Defaults are `balanced`.

Connect `ply_path` to the core **Preview3D** node to view the splat. An
example workflow is in [`workflows/gausify_orbit.json`](workflows/gausify_orbit.json).

Masks: plug any node that produces one MASK per selected frame (white =
subject) into the Poses node. With masks, training is done on RGBA cut-outs
and the pose check becomes available.

## Masks

Three nodes cut the subject out of the Frame Select images (ports of the
masking methods of gausify-orbit); connect their `masks` to the Poses node.
`stats` is JSON: `count`, `meanCoverage`, `noSubject`, `backend`, `keyColor`.

| Method | How | License | Download |
|---|---|---|---|
| Solid Colour | Chroma key: distance to the background colour (median of the frame borders, or `color`), tolerance, softness, flood fill from the borders (`contiguous` keeps holes in the subject) | none (no model) | none |
| RMBG-1.4 | BRIA AI background removal, one frame at a time | **non-commercial** (BRIA RMBG-1.4); set `accept_noncommercial_license` | 176 MB |
| SAM 2.1 | Meta's SAM 2.1 tiny video tracking: mark the subject on `prompt_frame` with `box` (`x0,y0,x1,y1`) and/or `points` (`x,y,label;...`, label 1 = subject, 0 = background), all normalized 0-1; with neither, the object at the centre is tracked. Propagated forward, then backward; consistent across frames | Apache-2.0 | fp32 165 MB, fp16 84 MB |

Models are downloaded on first use (sha256-checked) into `$GAUSIFY_MODEL_DIR`,
else `ComfyUI/models/gausify/models` (else `~/.cache/gausify/models`). Use a local
file with `GAUSIFY_MODEL_RMBG=/path/model.onnx` (SAM 2.1:
`GAUSIFY_MODEL_SAM2_FP32_VISION_ENCODER`, ...; names in `gausify/models.json`).
`device` is `auto` (CUDA, else CPU), `cuda` or `cpu`.

Despill: with a green or blue screen, enable `despill` on the Train node. It
is meant for Solid Colour masks (as in the app, where only chroma-key masks
are despilled): it needs a saturated green or blue background in the masked-out area.

ONNX Runtime: the nodes need `onnxruntime-gpu`. The plain `onnxruntime` package
conflicts with it (both provide the same module); if another node pack
installed it, `pip uninstall onnxruntime` first, then reinstall `onnxruntime-gpu`.

## Install

Manual install:

```bash
source /path/to/ComfyUI/venv/bin/activate  # ComfyUI's Python env, e.g. /comfy/mnt/venv
cd ComfyUI/custom_nodes
git clone https://github.com/michael-mueller-git/gausify-comfyui
pip install -r gausify-comfyui/requirements.txt
```

Keep a folder name other than `gausify` (that name is the library package
inside). Dependencies: numpy, scipy, opencv-python-headless (>= 4.12), pillow and
onnxruntime-gpu. `brush-cli` is downloaded on first use into
`ComfyUI/models/gausify/bin` (override with `GAUSIFY_BIN_DIR`).

COLMAP (the `colmap` pose estimator) is not installed by `requirements.txt`.
`pycolmap-cuda12` 4.2.1 needs CUDA 12 libraries (`cuda-toolkit<13`), which
conflict with the CUDA 13 `cuda-toolkit` that recent torch builds pin, so
installing it into ComfyUI's venv would break that environment. Instead, the first
use of the `colmap` estimator creates an isolated venv and installs `pycolmap-cuda12`,
`numpy` and the CUDA 12 runtime wheels there (a few hundred MB, needs network and
pip access; your pip configuration, e.g. `PIP_INDEX_URL`, applies). COLMAP then runs
in a subprocess from that venv; ComfyUI never imports it. The venv is at
`$GAUSIFY_COLMAP_VENV` if set, else `ComfyUI/models/gausify/colmap-venv`
(else `~/.cache/gausify/colmap-venv`), and is rebuilt automatically if the pinned
packages change. `GAUSIFY_COLMAP_PIP_ARGS` adds arguments to its `pip install`
(e.g. `--index-url https://...`). The AKAZE estimator needs none of this.

Working files (frames, masks, COLMAP models) go to ComfyUI's temp folder,
keyed by content, so re-running with the same inputs reuses the poses.

## Development

```bash
pip install -r requirements.txt torch pytest ruff
ruff check .
pytest -m "not slow"
```

## License

MIT (see `LICENSE`). Ports gausify-orbit (MIT, © Bryan Diniz); uses Brush
(Apache-2.0) and COLMAP (BSD-3-Clause). See [THIRD_PARTY.md](THIRD_PARTY.md).
