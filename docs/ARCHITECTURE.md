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
Installed: numpy, scipy, opencv-python-headless 4.12, pillow, pytest, ruff, torch 2.15 (CPU),
pycolmap-cuda12 4.2.1 (+ nvidia-cuda-runtime-cu12, nvidia-curand-cu12).
Rust (for brush-cli): `nix shell nixpkgs#rustc nixpkgs#cargo nixpkgs#cargo-zigbuild` (rustc 1.95).
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
| `gausify/brush.py` | `train(dataset_dir, args: list[str], out_ply, init_ply=None, progress, cancel) -> dict(psnr: float|None, splats: int)` runs `brush-cli`; `render(ply, cameras: list[dict(R,t,f,cx,cy,width,height)]) -> list[dict(rgba: (H,W,4) float32, depth: (H,W) float32)]` runs `brush-cli render` (our patch `brush/patches/gausify-cli.patch`, same renderer as `renderColor`/`renderDepth` in render.patch/geom.patch) | — |
| `gausify/cuda_libs.py` | `preload()` — ctypes-loads `libcudart.so.12`/`libcurand.so.10` from the `nvidia-*-cu12` wheels (RTLD_GLOBAL) before `import pycolmap` | — |
| `gausify/sfm_colmap.py` | `colmap_poses(images_dir, names, masks_dir|None, pairs, progress, cancel) -> SfmModel` (pycolmap-cuda12) | `web/src/sfm/colmap-sfm.ts` |
| `gausify/sfm_akaze/` | `akaze_poses(images_dir, names, masks_dir|None, pairs, max_side, max_features, progress, cancel) -> (SfmModel, matches)` | `web/src/sfm/{features.worker,matcher,geometry,linalg,reconstruct,ba}.ts` |
| `gausify/sfm.py` | `run_sfm(images_dir, names, masks_dir, estimator, window, closed_loop, flat_orbit, fill_missing, times) -> SfmModel` (pairs -> estimator -> flatten/fill -> stats) | `web/src/sfm/index.ts` (`runSfm`) |
| `gausify/posecheck/` | `check_poses(cams: OrbitCameras, mode: "auto"|"always", progress, cancel) -> dict|None` | `web/src/posecheck/*`, docs/web.md "Pose check" |
| `nodes/*.py` | ComfyUI nodes (Frame Select, Poses, Train Splat) | `src/components/Sidebar.tsx` for options |

## Binaries

- `brush-cli` built by `.github/workflows/build-brush.yml` from Brush `38185dedaf8143988a31a64c1f27c2323b1febdf`
  with `brush/patches/{brush-js,geom,deg,render}.patch` (order of `$REF/scripts/build-brush-web.mjs`) then `gausify-cli.patch`.
- COLMAP: `pycolmap-cuda12==4.2.1` (pip). Linux x86-64 only.
