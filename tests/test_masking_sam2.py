"""SAM 2.1 tracker plumbing with fake graphs: processing order, memory bank, pointers."""

import numpy as np
import pytest

from gausify.masking import sam2
from gausify.masking.sam2_math import HIDDEN, LOW, MEM, TOKENS, load_params


class _IO:
    def __init__(self, name):
        self.name = name


class Fake:
    def __init__(self, kind, log):
        self.kind, self.log = kind, log

    def get_inputs(self):
        return [_IO("x")]

    def run(self, names, feeds):
        k = self.kind
        if k == "vision":
            self.log.append(("vision",))
            return [np.zeros((1, 32, 4, 4), np.float32), np.zeros((1, 64, 2, 2), np.float32),
                    np.ones((1, HIDDEN, 64, 64), np.float32), np.zeros((1, HIDDEN, 64, 64), np.float32)]
        if k == "decoder":
            self.log.append(("decoder", feeds["input_points"].copy(), feeds["input_labels"].copy()))
            lg = np.full((1, 1, 1, LOW, LOW), -10, np.float32)
            lg[..., :, : LOW // 2] = 10
            return [lg, np.ones((1, HIDDEN), np.float32), np.ones((1, 1), np.float32), np.zeros((1, 1, 8, 8), np.float32)]
        if k == "memory":
            self.log.append(("memory", bool(feeds["is_mask_from_pts"][0])))
            return [np.ones((TOKENS * MEM,), np.float32), np.zeros((TOKENS, 1, MEM), np.float32)]
        self.log.append(("attention", feeds["spatial_memory"].shape, feeds["pointer_memory"].shape))
        return [np.zeros((1, 1, TOKENS, HIDDEN), np.float32)]


def _sessions(log):
    return [Fake(k, log) for k in ("vision", "decoder", "attention", "memory")]


def test_sam2_masks_order_and_memory():
    log = []
    frames = np.zeros((4, 16, 32, 3), np.uint8)
    masks, stats = sam2.sam2_masks(frames, None, 1, "fp32", sessions=(_sessions(log), "fake"), params=load_params())
    assert masks.shape == (4, 16, 32)
    assert (masks[:, :, :16] == 255).all() and (masks[:, :, 16:] == 0).all()
    assert stats == {"count": 4, "noSubject": 0, "meanCoverage": 0.5, "backend": "fake", "keyColor": None}
    # No prompt: a positive point at the centre of the frame, in 1024^2 coordinates.
    dec = [e for e in log if e[0] == "decoder"]
    assert dec[0][1].reshape(-1).tolist() == [512.0, 512.0] and dec[0][2].reshape(-1).tolist() == [1]
    assert dec[1][2].reshape(-1).tolist() == [-1]
    mem = [e[1] for e in log if e[0] == "memory"]
    assert mem == [True, False, False, False]
    att = [e for e in log if e[0] == "attention"]
    # Frames 2, 3, then 0 (backward): memory blocks 1, 2 and 3; pointers (4 tokens each) 1, 2 and 3.
    assert [a[1][0] // TOKENS for a in att] == [1, 2, 3]
    assert [a[2][0] // 4 for a in att] == [1, 2, 3]


def test_sam2_no_subject_and_box():
    class Neg(Fake):
        def run(self, names, feeds):
            out = super().run(names, feeds)
            if self.kind == "decoder":
                out[2] = -np.ones((1, 1), np.float32)
            return out

    log = []
    s = _sessions(log)
    s[1] = Neg("decoder", log)
    prompt = {"box": [0.25, 0.25, 0.75, 0.75], "points": [{"x": 0.5, "y": 0.5, "label": 0}]}
    _, stats = sam2.sam2_masks(np.zeros((2, 8, 8, 3), np.uint8), prompt, 0, sessions=s, params=load_params())
    assert stats["noSubject"] == 2
    pts, lab = [e for e in log if e[0] == "decoder"][0][1:]
    assert lab.reshape(-1).tolist() == [2, 3, 0]
    assert pts.reshape(-1).tolist() == [256, 256, 768, 768, 512, 512]


def test_sam2_errors():
    with pytest.raises(ValueError):
        sam2.sam2_masks(np.zeros((2, 8, 8, 3), np.uint8), None, 5, sessions=[])
    with pytest.raises(ValueError):
        sam2.model_names("fp8")
    with pytest.raises(InterruptedError):
        sam2.sam2_masks(np.zeros((2, 8, 8, 3), np.uint8), None, 0, cancel=lambda: True,
                        sessions=_sessions([]), params=load_params())


def test_model_specs():
    for v, total in (("fp32", 109497134 + 17800545 + 32364264 + 5586836), ("fp16", 55234395 + 9091532 + 16440396 + 2837008)):
        assert sam2.total_bytes(v) == total


@pytest.mark.slow
def test_sam2_real_models_if_present():
    """Only runs when the fp32 graphs are already in the local model cache; never downloads."""
    from gausify import downloader

    pytest.importorskip("onnxruntime")
    if not all(downloader.model_cached(n) for n in sam2.model_names("fp32")):
        pytest.skip("SAM 2.1 models not in the local cache")
    frames = np.zeros((3, 64, 64, 3), np.uint8)
    frames[:, 16:48, 16:48] = 255
    masks, stats = sam2.sam2_masks(frames, None, 0, "fp32", device="cpu")
    assert masks.shape == (3, 64, 64) and stats["count"] == 3
