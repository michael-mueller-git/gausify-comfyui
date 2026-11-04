"""Cases of web/src/pipeline/chroma.test.ts (keying part) plus edge cases."""

import numpy as np

from gausify.masking.chroma import DEFAULT_CHROMA, border_key, chroma_mask, key_distance


def _image(w, h, fill):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = fill
    return img


def _opts(**kw):
    return {**DEFAULT_CHROMA, **kw}


def test_chroma_contiguous_keeps_hole():
    w = h = 40
    img = _image(w, h, (20, 200, 30))
    img[10:30, 10:30] = (200, 40, 40)
    img[20, 20] = (20, 200, 30)
    key = border_key(img)
    assert key == (20, 200, 30)
    o = _opts()
    m = chroma_mask(img, key, o["tolerance"], o["softness"], o["contiguous"])
    assert m[0, 0] == 0
    assert m[15, 15] == 255
    assert m[20, 20] == 255  # inner hole preserved
    m = chroma_mask(img, key, o["tolerance"], o["softness"], False)
    assert m[20, 20] == 0


def test_key_distance():
    assert key_distance(np.array([30, 250, 45]), (20, 200, 30)) < 15
    assert key_distance(np.array([30, 80, 40]), (30, 200, 40)) > 100


def test_soft_edge_rounding():
    key = (0, 200, 0)
    # tol = 12*4.42 = 53.04, soft = 6*4.42 = 26.52; gray ramp with brightness scale 1
    img = _image(8, 8, key)
    img[4, 4] = (0, 200 - 66, 0)  # darker: s = 1, d = 66 -> (66-53.04)/26.52*255
    m = chroma_mask(img, key, 12, 6, False)
    assert m[4, 4] == int(np.floor(255 * (66 - 12 * 4.42) / (6 * 4.42) + 0.5))
    assert m[0, 0] == 0


def test_brighter_accepted_up_to_1_6():
    key = (0, 100, 0)
    assert key_distance(np.array([0, 160, 0]), key) < 1e-3
    assert key_distance(np.array([0, 200, 0]), key) > 39  # clamped at 1.6x


def test_border_key_upper_median():
    img = _image(4, 4, (10, 20, 30))
    img[0, 0] = (255, 255, 255)
    assert border_key(img) == (10, 20, 30)
