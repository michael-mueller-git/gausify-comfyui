# Architecture and module contracts

ComfyUI custom node pack that reproduces the output of
[gausify-orbit](https://github.com/obraia/gausify-orbit) (NVIDIA/CUDA only).
Reference source (read-only copy): `$REF` = `/home/nix/Downloads/gausify/.dev/ref`.

## Dev environment (NixOS host, prebuilt wheels only, no GPU here)

```bash
source .dev/env.sh        # uv venv in .venv on a Nix python + LD_LIBRARY_PATH for manylinux wheels
uv pip install <pkg>      # wheels only; never build Python packages from source
pytest -q tests && ruff check .
```
Installed: numpy, scipy, opencv-python-headless (>= 4.12), pillow, pytest, ruff, torch 2.15 (CPU),
onnxruntime (CPU wheel, NOT onnxruntime-gpu) and onnx for the masking tests.
Rust (for brush-cli): `nix shell nixpkgs#rustc nixpkgs#cargo nixpkgs#cargo-zigbuild` (rustc 1.95).
pycolmap is NOT installed in the dev venv: the slow COLMAP tests build the worker venv
(`GAUSIFY_COLMAP_VENV=$PWD/.dev/colmap-venv pytest -m slow tests/test_sfm_colmap.py` to keep it between runs).
No gsplat: there are no prebuilt wheels for current torch; all rendering goes through brush-cli.

Shared types: `gausify/types.py` (`Pose`, `Point3D`, `SfmModel`, `OrbitTimes`, `OrbitCameras`).
Presets: `gausify/presets.py`. Do not change their public fields without updating every user.

Rules for every module:
- Pure Python + numpy/scipy/opencv; torch (CUDA when available, CPU fallback) where the
  reference uses the GPU. No ComfyUI imports outside `nodes/`.
- Port the TypeScript faithfully: same constants, thresholds, ordering, tie-breaking.
  Cite the source file in a header comment.
- `progress(frac: float, msg: str)` callbacks and `cancel() -> bool` are optional args.
- Tests in `tests/`, runnable on CPU with `pytest` (no GPU, no binaries).

## Modules

| Module | Contents | Reference |
|---|---|---|
| `gausify/pairs.py` | `orbit_pairs(n, window, closed) -> list[tuple[int,int]]`, `orbit_times(names, times) -> list[float]` | `web/src/sfm/index.ts` |
| `gausify/colmap_io.py` | `write_model(model: SfmModel, sparse_dir)`, `read_model(sparse_dir) -> SfmModel`, `rename_images(sparse_dir, fn)` (binary COLMAP, PINHOLE camera id 1) | `web/src/sfm/colmap.ts`, `web/src/host/job.ts` (renameImages) |
| `gausify/orbit.py` | `flatten_orbit(model) -> float|None`, `fill_missing(model, matches, times) -> dict` | `web/src/sfm/orbit.ts` |
| `gausify/frames.py` | `select_frames(frames: np.ndarray (N,H,W,3) float, fps: float|None, target_frames, max_side, oversample, trim_start, trim_end) -> (np.ndarray, OrbitTimes)`; `sharpness(gray)` | `web/src/pipeline/frames.ts` |
| `gausify/alpha.py` | `erode(mask, r)`, `decontaminate(rgb_u8, mask_u8) -> rgb_u8`, `despill(...)`, `build_alpha_dataset(cams: OrbitCameras, out_dir, despill, despill_strength) -> Path` (RGBA PNGs straight alpha + sparse/0 with .png names) | `web/src/pipeline/{erode,decontam,chroma,masks}.ts`, `web/src/host/job.ts` |
| `gausify/train_config.py` | `train_config(opts: dict, masked: bool) -> dict` (kebab-case Brush keys, identical to `trainConfig()`), `to_cli_args(cfg: dict) -> list[str]` | `web/src/pipeline/train-config.ts` |
| `gausify/downloader.py` | `ensure_binary(name: str) -> Path` using `gausify/binaries.json` (sha256-verified, lock file, cache in `ComfyUI/models/gausify/bin` or `$GAUSIFY_BIN_DIR`) | — |
| `gausify/masking/chroma.py` | `border_key(rgb)`, `key_distance(rgb, key)`, `chroma_mask(rgb, key, tolerance, softness, contiguous) -> u8 mask`, `DEFAULT_CHROMA` (despill stays in `alpha.py`) | `web/src/pipeline/chroma.ts` |
| `gausify/masking/onnx.py` | `session(path, device="auto", label) -> (InferenceSession, backend label)` (CUDA, CPU fallback) | `web/src/pipeline/segment.ts`, `sam2.ts` (`openSessions`) |
| `gausify/masking/rmbg.py` | `rmbg_masks(frames_u8, progress, cancel, session=None, device) -> (masks u8 (N,H,W), stats)`, `preprocess`, `postprocess` | `web/src/pipeline/segment.ts`, `masks.ts` |
| `gausify/masking/sam2_math.py` | constants, `parse_params`, `round_bf16`, `pointer_pos_enc`, `fix_vision_shape`, `to_seq`/`to_nchw`, `logits_to_mask`, `prompt_frame`, `prompt_pixels` (`assets/sam2-params.bin` vendored) | `web/src/pipeline/sam2-math.ts` |
| `gausify/masking/sam2.py` | `Sam2Tracker` (memory bank on ONNX Runtime), `sam2_masks(frames_u8, prompt, prompt_frame, variant, progress, cancel, sessions=None) -> (masks, stats)`; fp32/fp16 graphs from `models.json` | `web/src/pipeline/sam2.ts`, `masks.ts` (`sam2Masks`) |
| `gausify/brush.py` | `train(dataset_dir, args: list[str], out_ply, init_ply=None, progress, cancel) -> dict(psnr: float|None, splats: int)` runs `brush-cli`; `render(ply, cameras: list[dict(R,t,f,cx,cy,width,height)]) -> list[dict(rgba: (H,W,4) float32, depth: (H,W) float32)]` runs `brush-cli render` (our patch `brush/patches/gausify-cli.patch`, same renderer as `renderColor`/`renderDepth` in render.patch/geom.patch) | — |
| `gausify/cuda_libs.py` | `preload()` — ctypes-loads `libcudart.so.12`/`libcurand.so.10` from the `nvidia-*-cu12` wheels (RTLD_GLOBAL) before `import pycolmap`; used only inside the COLMAP worker venv | — |
| `gausify/colmap_env.py` | `ensure_colmap_env(progress) -> Path` (python of the COLMAP worker venv): created on first use with `python -m venv` (fallback `uv venv`), pip-installs `pycolmap-cuda12==4.2.1`, numpy and the CUDA 12 wheels; location `$GAUSIFY_COLMAP_VENV`, else `ComfyUI/models/gausify/colmap-venv`, else `~/.cache/gausify/colmap-venv`; `$GAUSIFY_COLMAP_PIP_ARGS`; lock, spec-hash stamp (rebuild on change), atomic rename | — |
| `gausify/colmap_worker.py` | standalone script run as `<venv>/bin/python -I colmap_worker.py job.json`: SIFT extraction, matching of the given pairs, view-graph calibration, global (else incremental) mapping with pycolmap; stdout = JSON lines (`progress`, `log`, `result`, `error`), pycolmap noise on stderr; writes an `.npz` (camera, poses, keypoints, raw matches, points/tracks) | `web/src/sfm/colmap-sfm.ts`, colmap-wasm |
| `gausify/models.json`, `downloader.ensure_model(name) -> Path` | ONNX model specs (url, sha256, bytes; the only copy) and their download: size + sha256 checked, atomic rename, lock; cache `$GAUSIFY_MODEL_DIR`, else `ComfyUI/models/gausify/models`, else `~/.cache/gausify/models`; `$GAUSIFY_MODEL_<NAME>` = local file | `web/src/pipeline/segment.ts` (downloadModel) |
| `gausify/sfm_colmap.py` | `colmap_poses(images_dir, names, masks_dir|None, pairs, progress, cancel) -> SfmModel`; writes job.json, spawns the worker (progress relayed, `cancel()` terminates it, non-zero exit raises `ColmapError` with the stderr tail), builds the `SfmModel` from the `.npz` (colours via cv2). The main process never imports pycolmap | `web/src/sfm/colmap-sfm.ts` |
| `gausify/sfm_akaze/` | `akaze_poses(images_dir, names, masks_dir|None, pairs, max_side, max_features, progress, cancel) -> (SfmModel, matches)` | `web/src/sfm/{features.worker,matcher,geometry,linalg,reconstruct,ba}.ts` |
| `gausify/sfm.py` | `run_sfm(images_dir, names, masks_dir, estimator, window, closed_loop, flat_orbit, fill_missing, times) -> SfmModel` (pairs -> estimator -> flatten/fill -> stats) | `web/src/sfm/index.ts` (`runSfm`) |
| `gausify/posecheck/` | `check_poses(cams: OrbitCameras, mode: "auto"|"always", progress, cancel) -> dict|None` | `web/src/posecheck/*`, docs/web.md "Pose check" |
| `nodes/*.py` | ComfyUI nodes (Frame Select, Poses, Train Splat, Mask: Solid Colour / RMBG-1.4 / SAM 2.1 in `nodes/masks.py`, no preset) | `src/components/Sidebar.tsx` for options |

## Binaries

- `brush-cli` built by `.github/workflows/build-brush.yml` from Brush `38185dedaf8143988a31a64c1f27c2323b1febdf`
  with `brush/patches/{brush-js,geom,deg,render}.patch` (order of `$REF/scripts/build-brush-web.mjs`) then `gausify-cli.patch`.
- COLMAP: `pycolmap-cuda12==4.2.1` (pip) in its own venv (`gausify/colmap_env.py`), because it needs `cuda-toolkit<13` while ComfyUI's torch pins CUDA 13. Linux x86-64 only.
- Models (RMBG-1.4, SAM 2.1 tiny fp32/fp16): downloaded by `ensure_model` into the models cache above (see `gausify/models.json`).
