"""Cases of web/src/pipeline/sam2-math.test.ts (reference values from Transformers 4.57)."""

import numpy as np
import pytest

from gausify.masking import sam2_math as m


def _close(a, b, tol):
    assert np.allclose(np.asarray(a)[: len(b)], b, atol=tol), (a[: len(b)], b)


def test_params_and_pointer_encoding():
    p = m.load_params()
    assert p.no_memory.shape == (256,) and p.tpos.shape == (7, 64) and p.proj_w.shape == (64, 256)
    _close(m.pointer_pos_enc(0, p), [-0.57184, 0.46389, 1.8605, -1.30934], 1e-4)
    _close(m.pointer_pos_enc(1 / 15, p), [-0.6333, 0.43865, 1.86967, -1.27273], 1e-4)
    _close(m.pointer_pos_enc(1, p), [-1.40792, 0.01282, 1.85801, -0.65518], 1e-4)
    with pytest.raises(ValueError):
        m.parse_params(b"\0" * 8)


def test_bf16_round_nearest_even():
    a = m.round_bf16(np.array([1, 1 + 2**-8, 1 + 3 * 2**-8, -2.5, 1 + 2**-9], np.float32))
    assert a.tolist() == [1, 1, 1 + 2**-6, -2.5, 1]


def test_nchw_seq():
    nchw = np.arange(4096 * 2, dtype=np.float32).reshape(1, 2, 64, 64)
    seq = m.to_seq(nchw, 2)
    assert seq.shape == (4096, 1, 2)
    assert seq.reshape(-1)[:4].tolist() == [0, 4096, 1, 4097]
    assert np.array_equal(m.to_nchw(seq, 2), nchw)


def test_logits_to_mask():
    lg = np.full((256, 256), -20, np.float32)
    lg[:, 128:] = 20
    mk = m.logits_to_mask(lg, 64, 32)
    assert mk.shape == (32, 64) and mk.dtype == np.uint8
    assert mk[0, 0] == 0 and mk[0, 63] == 255
    assert mk[16, 31] == 0 and mk[16, 32] == 255


def test_prompt_pixels_and_frame():
    pr = {"box": [0.25, 0.5, 0.75, 1], "points": [{"x": 0.5, "y": 0.5, "label": 0}]}
    assert m.prompt_pixels(pr, 200, 100) == {"box": [50, 50, 150, 100], "points": [{"x": 100, "y": 50, "label": 0}]}
    assert m.prompt_pixels(None, 200, 100) == {"box": None, "points": [{"x": 100, "y": 50, "label": 1}]}
    assert m.prompt_pixels({"box": None, "points": []}, 200, 100)["points"][0]["label"] == 1
    assert m.prompt_frame(5, 3) == 3 and m.prompt_frame(5, 9) == 4 and m.prompt_frame(5, -2) == 0


def test_fix_vision_shape():
    with pytest.raises(ValueError):
        m.fix_vision_shape(bytes(10200))
    buf = bytearray(10200)
    buf[10137:10148] = bytes([0x0A, 0x0A, 0x08, 0x08, 0x07, 0x12, 0x04, 0x0A, 0x02, 0x08, 0x05])
    out = m.fix_vision_shape(bytes(buf))
    assert out[10147] == 0x04 and len(out) == len(buf)
    assert m.fix_vision_shape(out) == out  # idempotent
