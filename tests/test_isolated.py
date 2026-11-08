"""gausify.isolated: the worker protocol, run with the test interpreter standing in for the core venv."""

import sys

import numpy as np
import pytest

from gausify import core_env, isolated
from gausify.masking.chroma import chroma_masks


@pytest.fixture(autouse=True)
def _same_python(monkeypatch):
    monkeypatch.setattr(core_env, "ensure_core_env", lambda progress=None, packages=None: sys.executable)


def test_isolatable():
    assert isolated.isolatable(chroma_masks)
    assert not isolated.isolatable(lambda: 0)
    from gausify import brush

    assert not isolated.isolatable(brush.train)


def test_call_with_callbacks_and_result():
    frames = np.zeros((3, 16, 16, 3), np.uint8)
    frames[:, 4:12, 4:12] = (200, 30, 30)
    events = []
    masks, keys, cov = isolated.call(
        chroma_masks, frames, True, (0, 255, 0), 12, 6, True,
        progress=lambda f, m: events.append((f, m)), cancel=lambda: False,
    )
    ref = chroma_masks(frames, True, (0, 255, 0), 12, 6, True)
    assert np.array_equal(masks, ref[0]) and keys == ref[1] and cov == pytest.approx(ref[2])
    assert [m for _, m in events] == ["solid colour 1/3", "solid colour 2/3", "solid colour 3/3"]


def test_exception_type_is_kept():
    with pytest.raises(InterruptedError) as e:
        isolated.call(chroma_masks, np.zeros((1, 8, 8, 3), np.uint8), True, None, 1, 1, True, cancel=lambda: True)
    assert any("Gausify environment" in n for n in getattr(e.value, "__notes__", []))


def test_should_cancel_stops_worker():
    with pytest.raises(InterruptedError):
        isolated.call(chroma_masks, np.zeros((1, 8, 8, 3), np.uint8), True, None, 1, 1, True,
                      should_cancel=lambda: True)
