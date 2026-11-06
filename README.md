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
| RMBG-1.4 | BRIA AI background removal, one frame at a time | **non-commercial** (BRIA RMBG-1.4); set `accept_noncommercial_license` (or `GAUSIFY_ACCEPT_RMBG_LICENSE=1` for the whole instance) | 176 MB |
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

```bash
source /path/to/ComfyUI/venv/bin/activate  # ComfyUI's Python env, e.g. /comfy/mnt/venv
cd ComfyUI/custom_nodes
git clone https://github.com/michael-mueller-git/gausify-comfyui
gausify-comfyui/scripts/install.sh            # add --models and/or --colmap, see below
```

`scripts/install.sh` installs `requirements.txt` and repairs the two setups
that break the nodes: it keeps exactly one OpenCV build, between 4.12 and 5
(an installed contrib build stays contrib), and replaces plain `onnxruntime`
with `onnxruntime-gpu`. It then downloads `brush-cli` into
`models/gausify/bin`, and checks that OpenCV has AKAZE and that ONNX Runtime
can use CUDA. Options:

- `--models`: also downloads the mask models (SAM 2.1 fp32 and RMBG-1.4, about 350 MB). With them in
  place, the check also tests that CUDA actually initialises.
- `--colmap`: also builds the COLMAP venv (see below).

Without an active venv it uses `venv/` or `.venv/` next to ComfyUI (or the base directory), or
`/comfy/mnt/venv`; `PYTHON=/path/to/python` selects the interpreter explicitly.
Running the script again is safe. Restart ComfyUI afterwards.

Manual install instead: `pip install -r gausify-comfyui/requirements.txt`.

If ComfyUI runs with `--base-directory` (e.g. Docker images using `/basedir`),
it loads custom nodes from `<base-directory>/custom_nodes`, not
`ComfyUI/custom_nodes`; clone there instead.

Keep a folder name other than `gausify` (that name is the library package
inside). Dependencies: numpy, scipy, opencv-python-headless (>= 4.12, < 5: OpenCV 5 drops `cv2.AKAZE_create`), pillow and
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

### COLMAP environment: set up ahead of time

To build the COLMAP venv before the first run (e.g. to see pip's output, or on
a machine where ComfyUI has no network), run the setup script with ComfyUI's
venv active. It runs the node's own setup, so the venv is exactly what the node
expects, and then checks that COLMAP sees the GPU:

```bash
source /path/to/ComfyUI/venv/bin/activate
ComfyUI/custom_nodes/gausify-comfyui/scripts/setup-colmap-env.sh
```

It prints `cuda build: True | devices: 1` (or more) on success and exits with 3
if COLMAP sees no CUDA device. The venv goes to `$GAUSIFY_COLMAP_VENV`, else
`ComfyUI/models/gausify/colmap-venv`; pass a directory as the first argument to
choose another (then export `GAUSIFY_COLMAP_VENV` for ComfyUI too). `PYTHON=...`
selects the interpreter instead of the active venv. Use ComfyUI's Python: the venv
is stamped with its version and rebuilt if it differs. Don't create it with a plain
`python -m venv` + `pip install`; without the stamp the node replaces it.

### Upgrading from a version that installed pycolmap into ComfyUI's venv

Older versions put `pycolmap-cuda12` in ComfyUI's venv, which downgraded
torch's `cuda-toolkit` metapackage (and pinned opencv 4.12 / numpy < 2.3).
Restore torch's `cuda-toolkit` version (see `pip check`; torch 2.15 wants 13.2.2),
remove pycolmap, and reinstall:

```bash
pip install cuda-toolkit==13.2.2          # the version torch requires
pip uninstall -y pycolmap-cuda12
pip install -r requirements.txt
pip check
```

If other nodes needed newer opencv/numpy than 4.12/2.2, reinstall the versions you had.

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
