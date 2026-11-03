# Third-party software

## gausify-orbit — MIT License, Copyright (c) 2026 Bryan Diniz

<https://github.com/obraia/gausify-orbit>. The Python modules in `gausify/`
are ports of its TypeScript (frame selection, AKAZE SfM, orbit corrections,
masks/alpha, training configuration, pose check, presets) and the Brush
patches in `brush/patches/` come from it. The MIT license text is in
`LICENSE`; its copyright notice above applies to those portions.

The `colmap` estimator (`gausify/sfm_colmap.py`) follows
[colmap-wasm](https://github.com/obraia/colmap-wasm) (MIT, Bryan Diniz):
same COLMAP steps and options, run natively.

## Brush — Apache License 2.0

<https://github.com/ArthurBrussee/brush>, commit
`38185dedaf8143988a31a64c1f27c2323b1febdf`. `brush-cli` is built from it with
the patches in `brush/patches/` and downloaded on first use. Licensed under the
Apache License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>);
the patches are modifications of that source and are distributed under the
same license.

## COLMAP — BSD 3-Clause License

<https://github.com/colmap/colmap>, Copyright (c) 2023, ETH Zurich and UNC
Chapel Hill. Used through `pycolmap-cuda12` 4.2.1 (pip). COLMAP bundles further third-party code under its own licenses
(see its repository).

## NVIDIA CUDA runtime libraries

`nvidia-cuda-runtime-cu12` and `nvidia-curand-cu12` (pip) are distributed
under the NVIDIA CUDA Toolkit EULA.
