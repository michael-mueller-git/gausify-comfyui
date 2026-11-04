import numpy as np
import pytest

from gausify.masking import rmbg


def test_input_size_preprocess_postprocess():
    assert rmbg.input_size(64, 48) == (1024, 1024)
    t = rmbg.preprocess(np.array([[[255, 0, 128]]], np.uint8))
    assert t.shape == (1, 3, 1, 1) and t.dtype == np.float32
    assert abs(t[0, 0, 0, 0] - 0.5) < 1e-3 and abs(t[0, 1, 0, 0] + 0.5) < 1e-3 and abs(t[0, 2, 0, 0]) < 0.01
    out = np.full(8, 0.1, np.float32)
    out[:4] = 0.9
    m = rmbg.postprocess(out)
    assert (m[0], m[7]) == (255, 0)
    assert (rmbg.postprocess(np.full(8, 0.02, np.float32)) == 5).all()  # near-constant: no stretch


def test_model_spec_single_source():
    s = rmbg.model()
    assert s["sha256"].startswith("8cafcf77") and s["bytes"] == 176153355
    assert s["url"].startswith("https://huggingface.co/briaai/RMBG-1.4/resolve/2ceba5a5")


class _Fake:
    """Stands in for an InferenceSession: the mask is the red channel."""

    class _IO:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self._IO("input")]

    def get_outputs(self):
        return [self._IO("aux"), self._IO("out")]

    def run(self, names, feeds):
        assert names == ["out"]
        x = feeds["input"]
        assert x.shape == (1, 3, 1024, 1024) and x.dtype == np.float32
        return [x[:, :1] + 0.5]


def test_rmbg_masks_fake_session():
    frames = np.zeros((2, 16, 32, 3), np.uint8)
    frames[:, :, :16, 0] = 255
    calls = []
    masks, stats = rmbg.rmbg_masks(frames, lambda f, msg: calls.append(f), session=(_Fake(), "fake"))
    assert masks.shape == (2, 16, 32) and masks.dtype == np.uint8
    assert (masks[:, :, :14] == 255).all() and (masks[:, :, 18:] == 0).all()
    assert stats["count"] == 2 and stats["backend"] == "fake" and stats["noSubject"] == 0
    assert abs(stats["meanCoverage"] - 0.5) < 1e-6
    assert calls[-1] == 1.0


def test_rmbg_cancel():
    with pytest.raises(InterruptedError):
        rmbg.rmbg_masks(np.zeros((2, 8, 8, 3), np.uint8), cancel=lambda: True, session=_Fake())


def test_rmbg_masks_real_onnx_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    from gausify.masking.onnx import session

    node = helper.make_node("ReduceMean", ["input"], ["mask"], axes=[1], keepdims=1)
    g = helper.make_graph(
        [node], "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1024, 1024])],
        [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 1024, 1024])],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, path)
    sess, label = session(path, "cpu", "RMBG-1.4")
    assert label == "RMBG-1.4 · CPU"
    frames = np.zeros((1, 16, 32, 3), np.uint8)
    frames[:, :, :16] = 255
    masks, stats = rmbg.rmbg_masks(frames, session=(sess, label))
    assert (masks[0, :, :14] == 255).all() and (masks[0, :, 18:] == 0).all()
    assert stats["backend"] == label


@pytest.mark.slow
def test_rmbg_real_model_if_present():
    """Only runs when GAUSIFY_MODEL_DIR (or GAUSIFY_MODEL_RMBG) already holds the model; never downloads."""
    from gausify import downloader

    pytest.importorskip("onnxruntime")
    if not downloader.model_cached("rmbg"):
        pytest.skip("RMBG-1.4 model not in the local cache")
    frames = np.full((1, 96, 128, 3), 255, np.uint8)
    masks, stats = rmbg.rmbg_masks(frames, device="cpu")
    assert masks.shape == (1, 96, 128) and stats["backend"].startswith("RMBG-1.4")
