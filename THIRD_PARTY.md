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

## RMBG-1.4 — BRIA AI, non-commercial

<https://huggingface.co/briaai/RMBG-1.4> (revision `2ceba5a5efaec153162aedea169f76caf9b46cf8`).
Released by BRIA AI under the RMBG-1.4 license: **free for non-commercial use
only**; commercial use needs a license from BRIA. The model is downloaded on
first use and never redistributed here; the RMBG node asks you to accept the
license (`accept_noncommercial_license`).

## SAM 2.1 — Meta, Apache License 2.0

<https://github.com/facebookresearch/sam2> (Segment Anything Model 2.1, tiny).
Run from ONNX exports downloaded on first use:
[jax-image-tools/sam21-tiny-video-onnx](https://huggingface.co/jax-image-tools/sam21-tiny-video-onnx)
(fp32) and [obraia/sam2.1-tiny-video-onnx-fp16](https://huggingface.co/obraia/sam2.1-tiny-video-onnx-fp16)
(fp16). `gausify/masking/assets/sam2-params.bin` holds three small parameter
tensors (no_memory_embedding, memory temporal positional encoding, object
pointer temporal projection) extracted from `facebook/sam2.1-hiera-tiny`, as in
gausify-orbit; Apache-2.0.

## ONNX Runtime — MIT License

<https://github.com/microsoft/onnxruntime>, Copyright (c) Microsoft Corporation
(`onnxruntime-gpu`, pip).
