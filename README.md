# Gausify Orbit for ComfyUI

Turn an orbit video (a camera circling a subject, e.g. an AI-generated
turntable) into a 3D Gaussian splat, inside ComfyUI. The nodes reproduce the
pipeline and output of [gausify-orbit](https://github.com/obraia/gausify-orbit):
sharp-frame selection, camera poses (AKAZE + incremental SfM, or COLMAP's SIFT
+ global mapper), optional masked (cut-out) training, and training with
[Brush](https://github.com/ArthurBrussee/brush).

**NVIDIA / CUDA only** (Linux and Windows).

## Nodes (category "Gausify Orbit")

| Node | Inputs | Outputs |
|---|---|---|
| **Gausify Orbit Frame Select** | `video` (core VIDEO) or `images` + `fps` (0 = image sequence); preset, frames, max_side, candidates, trim_start, trim_end (-1 = end) | IMAGE, ORBIT_TIMES |
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

## Install

Via ComfyUI-Manager / the Comfy Registry (`gausify-comfyui`), or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/michael-mueller-git/gausify-comfyui
pip install -r gausify-comfyui/requirements.txt
```

Keep a folder name other than `gausify` (that name is the library package
inside). Dependencies: numpy, scipy, opencv-python-headless 4.12, pillow and
COLMAP 4.2.1 (`pycolmap-cuda12` + CUDA runtime wheels on Linux; `pycolmap`
on Windows, plus the official COLMAP CUDA build downloaded on first use for
SIFT/matching). `brush-cli` is downloaded on first use into
`ComfyUI/models/gausify/bin` (override with `GAUSIFY_BIN_DIR`).

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
