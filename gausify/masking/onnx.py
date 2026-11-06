"""ONNX Runtime sessions: CUDA when available (device "auto"/"cuda"), else CPU.

Replaces the `webgpu` -> `wasm` fallback of `loadOrt`/`Segmenter.create` in
gausify-orbit `web/src/pipeline/segment.ts` and `sam2.ts` (`openSessions`).
"""

from __future__ import annotations

import logging

log = logging.getLogger("gausify.masking")

DEVICES = ("auto", "cuda", "cpu")


def session(path, device: str = "auto", label: str = ""):
    """(InferenceSession, backend label like "RMBG-1.4 · CUDA"). `path`: a model
    file path or the model bytes."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("onnxruntime-gpu is required for the mask nodes (pip install onnxruntime-gpu)") from e
    if hasattr(path, "__fspath__"):
        path = str(path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    if device == "cuda" and not has_cuda:
        raise RuntimeError(
            "device=cuda but ONNX Runtime has no CUDAExecutionProvider "
            "(install onnxruntime-gpu and uninstall plain onnxruntime)"
        )
    if device == "auto" and not has_cuda:
        log.warning("%s runs on CPU: ONNX Runtime %s has no CUDAExecutionProvider (providers: %s); "
                    "install onnxruntime-gpu and uninstall plain onnxruntime",
                    label or "ONNX model", ort.__version__, ", ".join(ort.get_available_providers()))
    if device != "cpu" and has_cuda:
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None:
            try:
                preload()
            except Exception as e:  # noqa: BLE001 - optional helper
                log.debug("onnxruntime.preload_dlls failed: %s", e)
        try:
            s = ort.InferenceSession(path, sess_options=opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            if s.get_providers()[0] == "CUDAExecutionProvider":
                return s, _label(label, "CUDA")
            if device == "cuda":
                raise RuntimeError("CUDAExecutionProvider could not be initialised")
            log.warning("%s runs on CPU: the CUDAExecutionProvider of ONNX Runtime %s could not be initialised "
                        "(usually missing CUDA/cuDNN libraries of the CUDA version this onnxruntime-gpu was built for)",
                        label or "ONNX model", ort.__version__)
        except Exception as e:  # noqa: BLE001
            if device == "cuda":
                raise
            log.warning("CUDA unavailable for %s, using CPU: %s", label or "ONNX model", e)
    s = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return s, _label(label, "CPU")


def _label(label: str, backend: str) -> str:
    return f"{label} · {backend}" if label else backend
