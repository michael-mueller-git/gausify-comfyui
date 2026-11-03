"""Image primitives for the pose check: distance transforms, binary morphology,
gradients and sampling on (h, w) arrays. Binary masks are uint8 with 0/1.

Port of web/src/posecheck/image.ts (gausify-orbit). The distance transform and
the binary morphology use scipy.ndimage where it computes exactly the same thing
(exact Euclidean distance transform; erosion with pixels outside the image = 0;
dilation by a symmetric disk; hole filling with 4-connectivity). `dt1` keeps the
Felzenszwalb & Huttenlocher 1-D pass for reference and tests.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

INF = 1e20


def dt1(f: np.ndarray) -> np.ndarray:
    """1-D squared distance transform of `f` (Felzenszwalb & Huttenlocher)."""
    n = len(f)
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -INF
    z[1] = INF
    for q in range(1, n):
        s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def distance_to(target: np.ndarray) -> np.ndarray:
    """Euclidean distance from each pixel to the nearest pixel where `target`
    is set (0 on those pixels). With no target pixel at all, every distance is
    huge (1e10), like the TS."""
    t = np.asarray(target) != 0
    if not t.any():
        return np.full(t.shape, np.sqrt(INF), dtype=np.float32)
    return ndimage.distance_transform_edt(~t).astype(np.float32)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the mask's boundary in pixels: > 0 outside, < 0
    inside (+-1 on the pixels next to it)."""
    m = np.asarray(mask) != 0
    outside = distance_to(m)
    inside = distance_to(~m)
    return (outside - inside).astype(np.float32)


def disk(r: int) -> list[tuple[int, int]]:
    """Offsets (dx, dy) of a disk of radius r (x^2 + y^2 <= r^2 + 0.5); r = 1
    is the 4-neighbour cross."""
    return [(x, y) for y in range(-r, r + 1) for x in range(-r, r + 1) if x * x + y * y <= r * r + 0.5]


def _footprint(shape: list[tuple[int, int]]) -> np.ndarray:
    r = max(max(abs(dx), abs(dy)) for dx, dy in shape)
    fp = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    for dx, dy in shape:
        fp[dy + r, dx + r] = True
    return fp


def erode_by(m: np.ndarray, shape: list[tuple[int, int]]) -> np.ndarray:
    """Binary erosion by `shape` (offsets); pixels outside the image count as 0."""
    out = ndimage.binary_erosion(np.asarray(m) != 0, structure=_footprint(shape), border_value=0)
    return out.astype(np.uint8)


def dilate_by(m: np.ndarray, shape: list[tuple[int, int]]) -> np.ndarray:
    """Binary dilation by `shape` (symmetric offsets)."""
    out = ndimage.binary_dilation(np.asarray(m) != 0, structure=_footprint(shape))
    return out.astype(np.uint8)


def erode_cross(m: np.ndarray, n: int = 1) -> np.ndarray:
    """Erosion repeated `n` times by the 4-neighbour cross."""
    out = (np.asarray(m) != 0).astype(np.uint8)
    cross = disk(1)
    for _ in range(n):
        out = erode_by(out, cross)
    return out


def open_disk(m: np.ndarray, r: int) -> np.ndarray:
    """Morphological opening by a disk of radius r."""
    s = disk(r)
    return dilate_by(erode_by(m, s), s)


def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradient divided by 8 (per pixel), borders clamped."""
    p = np.pad(np.asarray(img, dtype=np.float32), 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, k, ll = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    gx = (c + 2 * f + ll - a - 2 * d - g) / 8
    gy = (g + 2 * k + ll - a - 2 * b - c) / 8
    return gx.astype(np.float32), gy.astype(np.float32)


def bilinear(img: np.ndarray, x, y):
    """Bilinear sample at continuous array coordinates (x, y) = pixel indices,
    clamped to the border. Vectorized over x, y."""
    h, w = img.shape
    x = np.clip(np.asarray(x, dtype=np.float64), 0, w - 1)
    y = np.clip(np.asarray(y, dtype=np.float64), 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    i00 = img[y0, x0].astype(np.float64)
    i10 = img[y0, x1].astype(np.float64)
    i01 = img[y1, x0].astype(np.float64)
    i11 = img[y1, x1].astype(np.float64)
    return (i00 * (1 - fx) + i10 * fx) * (1 - fy) + (i01 * (1 - fx) + i11 * fx) * fy


def blur(img: np.ndarray) -> np.ndarray:
    """3x3 binomial blur ([1 2 1]/4 each way), borders clamped."""
    img = np.asarray(img, dtype=np.float32)
    p = np.pad(img, ((0, 0), (1, 1)), mode="edge")
    tmp = ((p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4).astype(np.float32)
    p = np.pad(tmp, ((1, 1), (0, 0)), mode="edge")
    return ((p[:-2] + 2 * p[1:-1] + p[2:]) / 4).astype(np.float32)


def halve(img: np.ndarray) -> np.ndarray:
    """Half size (2x2 mean; an odd last row/column is dropped)."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    w2, h2 = max(1, w >> 1), max(1, h >> 1)
    xs = 2 * np.arange(w2)
    ys = 2 * np.arange(h2)
    xb = np.where(xs + 1 < w, xs + 1, xs)
    yb = np.minimum(ys + 1, h - 1)
    out = (img[np.ix_(ys, xs)] + img[np.ix_(ys, xb)] + img[np.ix_(yb, xs)] + img[np.ix_(yb, xb)]) / 4
    return out.astype(np.float32)


def fill_holes(m: np.ndarray) -> np.ndarray:
    """The mask with its holes filled: background pixels not connected (4-way)
    to the image border become subject."""
    return ndimage.binary_fill_holes(np.asarray(m) != 0, structure=ndimage.generate_binary_structure(2, 1)).astype(np.uint8)
