"""Multi-view geometry: fundamental matrix (RANSAC), essential matrix decomposition,
triangulation and PnP. Port of web/src/sfm/geometry.ts.

"Normalized" coordinates are ((u - cx) / f, (v - cy) / f). Poses are (R, t) world->camera.
The RANSAC loops keep the TS semantics exactly (same mulberry32 PRNG and seeds, same
sampling with rejection, same adaptive iteration limits and strict-improvement rule);
the hypotheses are only evaluated in vectorized chunks, then walked sequentially.
"""

from __future__ import annotations

import math

import numpy as np

from .linalg import exp_so3, nearest_rotation, null_vector, svd3, sym_eig

_LOG_P = math.log(0.001)
_CHUNK = 128


def _adaptive(w: float, k: int, cap: int) -> float:
    """min(cap, ceil(log(0.001) / log(max(1 - w^k, 1e-12)))) with JS semantics: when
    1 - w^k rounds to 1 the division gives -Infinity and the loop stops."""
    d = math.log(max(1 - w**k, 1e-12))
    if d == 0:
        return -math.inf
    return min(cap, math.ceil(_LOG_P / d))


class Mulberry32:
    """Deterministic PRNG of geometry.ts (`rng(seed)`), generated in vectorized blocks.

    The state advances by a constant per draw, so the k-th output is a pure function of
    the seed: blocks are bit-identical to calling the JS closure k times."""

    def __init__(self, seed: int):
        self.a = int(seed) & 0xFFFFFFFF
        self.buf: list[float] = []
        self.pos = 0

    def _block(self, m: int) -> list[float]:
        k = np.arange(1, m + 1, dtype=np.uint64)
        a = ((np.uint64(self.a) + k * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        self.a = int(a[-1])
        t = a
        t = (t ^ (t >> np.uint32(15))) * (t | np.uint32(1))
        t = t ^ (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61)))
        t = t ^ (t >> np.uint32(14))
        return (t.astype(np.float64) / 4294967296.0).tolist()

    def __call__(self) -> float:
        if self.pos >= len(self.buf):
            self.buf = self._block(4096)
            self.pos = 0
        v = self.buf[self.pos]
        self.pos += 1
        return v


def rng(seed: int) -> Mulberry32:
    return Mulberry32(seed)


def sample(n: int, k: int, rand: Mulberry32) -> list[int]:
    out: list[int] = []
    while len(out) < k:
        i = int(rand() * n)
        if i not in out:
            out.append(i)
    return out


def hartley(p: np.ndarray) -> np.ndarray:
    """Hartley normalization: centroid at the origin, mean distance √2."""
    m = p.mean(0)
    d = np.hypot(p[:, 0] - m[0], p[:, 1] - m[1]).sum()
    s = math.sqrt(2) / max(d / len(p), 1e-12)
    return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1.0]])


def eight_point(p1: np.ndarray, p2: np.ndarray, idx: np.ndarray, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """F (pixels) via the normalized 8-point algorithm with rank 2 enforced.
    idx: (B, m) sample indices -> (B,3,3)."""
    idx = np.atleast_2d(idx)
    x1 = T1[0, 0] * p1[idx, 0] + T1[0, 2]
    y1 = T1[1, 1] * p1[idx, 1] + T1[1, 2]
    x2 = T2[0, 0] * p2[idx, 0] + T2[0, 2]
    y2 = T2[1, 1] * p2[idx, 1] + T2[1, 2]
    A = np.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones_like(x1)], -1)
    F = null_vector(A).reshape(-1, 3, 3)
    # rank 2: F <- F - (F v3) v3ᵀ, v3 the smallest eigenvector of FᵀF
    v3 = sym_eig(np.swapaxes(F, 1, 2) @ F)[1][:, :, 0]
    Fv = np.einsum("bij,bj->bi", F, v3)
    F = F - Fv[:, :, None] * v3[:, None, :]
    return T2.T[None] @ F @ T1[None]


def sampson(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Sampson distance² of each correspondence under each F: F (B,3,3) -> (B, n)."""
    F = F.reshape(-1, 9)
    x1, y1, x2, y2 = p1[None, :, 0], p1[None, :, 1], p2[None, :, 0], p2[None, :, 1]
    f = [F[:, i : i + 1] for i in range(9)]
    a0 = f[0] * x1 + f[1] * y1 + f[2]
    a1 = f[3] * x1 + f[4] * y1 + f[5]
    a2 = f[6] * x1 + f[7] * y1 + f[8]
    b0 = f[0] * x2 + f[3] * y2 + f[6]
    b1 = f[1] * x2 + f[4] * y2 + f[7]
    e = x2 * a0 + y2 * a1 + a2
    with np.errstate(invalid="ignore", over="ignore"):
        return (e * e) / (a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1 + 1e-30)


def ransac_fundamental(p1: np.ndarray, p2: np.ndarray, threshold_px: float = 1.5, seed: int = 1) -> dict | None:
    """Robust fundamental matrix (RANSAC + refit on inliers). p1, p2: (n,2) pixels.
    Returns {"F": (3,3), "inliers": bool (n,), "count": int} or None."""
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
    n = len(p1)
    if n < 12:
        return None
    T1, T2 = hartley(p1), hartley(p2)
    thr2 = threshold_px * threshold_px
    rand = rng(seed)
    best = None
    best_count = 0
    max_iter = 2000
    it = 0
    while it < max_iter:
        m = min(_CHUNK, max_iter - it)
        idx = np.array([sample(n, 8, rand) for _ in range(m)])
        Fs = eight_point(p1, p2, idx, T1, T2)
        counts = (sampson(Fs, p1, p2) < thr2).sum(1)
        for j in range(m):
            if it >= max_iter:
                break
            c = int(counts[j])
            if c > best_count:
                best_count = c
                best = Fs[j]
                w = c / n
                max_iter = _adaptive(w, 8, 2000)
            it += 1
    if best is None:
        return None
    F = best
    for _ in range(3):
        idx = np.nonzero(sampson(F, p1, p2)[0] < thr2)[0]
        if len(idx) < 8:
            break
        Fr = eight_point(p1, p2, idx[None], T1, T2)[0]
        if int((sampson(Fr, p1, p2)[0] < thr2).sum()) >= len(idx):
            F = Fr
        else:
            break
    inl = sampson(F, p1, p2)[0] < thr2
    return {"F": F, "inliers": inl, "count": int(inl.sum())}


def triangulate_groups(R: np.ndarray, t: np.ndarray, xs: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DLT triangulation of several points at once. Observation o (pose R[o], t[o],
    normalized xs[o]) belongs to the group starting at `starts` (contiguous groups).
    Returns X (G,3) and a validity mask (|w| >= 1e-12)."""
    x, y = xs[:, 0:1], xs[:, 1:2]
    r1 = np.concatenate([x * R[:, 2] - R[:, 0], x * t[:, 2:3] - t[:, 0:1]], 1)
    r2 = np.concatenate([y * R[:, 2] - R[:, 1], y * t[:, 2:3] - t[:, 1:2]], 1)
    M = r1[:, :, None] * r1[:, None, :] + r2[:, :, None] * r2[:, None, :]
    AtA = np.add.reduceat(M, starts, axis=0)
    v = sym_eig(AtA)[1][:, :, 0]
    ok = np.abs(v[:, 3]) >= 1e-12
    w = np.where(ok, v[:, 3], 1.0)
    return v[:, :3] / w[:, None], ok


def triangulate(poses: list[tuple[np.ndarray, np.ndarray]], xs) -> np.ndarray | None:
    """3D point from >= 2 normalized observations (DLT)."""
    R = np.stack([p[0] for p in poses])
    t = np.stack([np.asarray(p[1], dtype=np.float64) for p in poses])
    X, ok = triangulate_groups(R, t, np.asarray(xs, dtype=np.float64).reshape(-1, 2), np.array([0]))
    return X[0] if ok[0] else None


def triangulate_two(R2: np.ndarray, t2: np.ndarray, n1: np.ndarray, n2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulates each correspondence between the identity camera and (R2, t2)."""
    m = len(n1)
    R = np.empty((2 * m, 3, 3))
    t = np.zeros((2 * m, 3))
    R[0::2] = np.eye(3)
    R[1::2] = R2
    t[1::2] = t2
    xs = np.empty((2 * m, 2))
    xs[0::2] = n1
    xs[1::2] = n2
    return triangulate_groups(R, t, xs, np.arange(0, 2 * m, 2))


def to_cam(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ R.T + t


def center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


def relative_pose(F, f, cx, cy, n1, n2, idx) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    """Relative pose of camera 2 from F and the focal length: the 4 solutions of E,
    picking the one with the most points in front of both cameras."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    E = K.T @ F @ K
    U, _, V = svd3(E)
    for M in (U, V):
        if np.linalg.det(M) < 0:
            M[:, 2] = -M[:, 2]
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    Vt = V.T
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    u3 = U[:, 2].copy()
    a, b = n1[idx], n2[idx]
    best = ((np.eye(3), np.zeros(3)), -1)
    for R in (R1, R2):
        for s in (1, -1):
            t = s * u3
            X, ok = triangulate_two(R, t, a, b)
            good = int((ok & (X[:, 2] > 0) & (to_cam(R, t, X)[:, 2] > 0)).sum())
            if good > best[1]:
                best = ((R, t), good)
    return best


def tri_angle(c1: np.ndarray, c2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the rays from two cameras to X (broadcasting, last axis 3)."""
    a = c1 - X
    b = c2 - X
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def refine_pose(pose, X: np.ndarray, x: np.ndarray, huber: float, iters: int = 10):
    """Gauss-Newton (Huber) pose refinement from normalized 2D-3D correspondences."""
    R, t = pose[0].copy(), np.asarray(pose[1], dtype=np.float64).copy()
    for _ in range(iters):
        RX = X @ R.T
        c = RX + t
        v = c[:, 2] > 1e-9
        RXv, c, xv = RX[v], c[v], x[v]
        iz = 1 / c[:, 2]
        rx = c[:, 0] * iz - xv[:, 0]
        ry = c[:, 1] * iz - xv[:, 1]
        e = np.hypot(rx, ry)
        with np.errstate(divide="ignore"):
            w = np.where(e <= huber, 1.0, huber / e)
        m = len(c)
        J = np.zeros((m, 2, 6))
        P0 = np.stack([iz, np.zeros(m), -c[:, 0] * iz * iz], 1)
        P1 = np.stack([np.zeros(m), iz, -c[:, 1] * iz * iz], 1)
        # −[RX]x
        Sx = np.zeros((m, 3, 3))
        Sx[:, 0, 1], Sx[:, 0, 2] = RXv[:, 2], -RXv[:, 1]
        Sx[:, 1, 0], Sx[:, 1, 2] = -RXv[:, 2], RXv[:, 0]
        Sx[:, 2, 0], Sx[:, 2, 1] = RXv[:, 1], -RXv[:, 0]
        J[:, 0, :3] = np.einsum("nk,nkj->nj", P0, Sx)
        J[:, 1, :3] = np.einsum("nk,nkj->nj", P1, Sx)
        J[:, 0, 3:] = P0
        J[:, 1, 3:] = P1
        res = np.stack([rx, ry], 1)
        g = np.einsum("n,nra,nr->a", w, J, res)
        H = np.einsum("n,nra,nrb->ab", w, J, J)
        H[np.diag_indices(6)] *= 1 + 1e-6
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break
        R = exp_so3(d[:3]) @ R
        t = t + d[3:]
        if math.sqrt(float(d @ d)) < 1e-10:
            break
    return R, t


def dlt_pose(X: np.ndarray, x: np.ndarray, idx: np.ndarray):
    """Poses via 6-point DLT (normalized), 3D points recentered. idx (B,6).
    Returns R (B,3,3), t (B,3), ok (B,)."""
    idx = np.atleast_2d(idx)
    Xs = X[idx]  # (B,m,3)
    m = idx.shape[1]
    c = Xs.mean(1)
    s = m / np.maximum(np.linalg.norm(Xs - c[:, None], axis=2).sum(1), 1e-12)
    Xn = np.concatenate([(Xs - c[:, None]) * s[:, None, None], np.ones(idx.shape + (1,))], 2)
    u = x[idx, 0][..., None]
    v = x[idx, 1][..., None]
    B = len(idx)
    A = np.zeros((B, m, 2, 12))
    A[:, :, 0, 0:4] = Xn
    A[:, :, 0, 8:12] = -u * Xn
    A[:, :, 1, 4:8] = Xn
    A[:, :, 1, 8:12] = -v * Xn
    p = null_vector(A.reshape(B, 2 * m, 12)).reshape(B, 3, 4)
    M = p[:, :, :3]
    p4 = p[:, :, 3]
    neg = np.linalg.det(M) < 0
    M = np.where(neg[:, None, None], -M, M)
    p4 = np.where(neg[:, None], -p4, p4)
    S = np.linalg.svd(M, compute_uv=False)
    scale = S.mean(1)
    ok = scale > 1e-12
    R = nearest_rotation(M)
    sc = np.where(ok, scale, 1.0)
    t = p4 / sc[:, None] / s[:, None] - np.einsum("bij,bj->bi", R, c)
    return R, t, ok


def _inlier_mask(R, t, X, x, thr) -> np.ndarray:
    """Inlier masks (B, n) of poses R (B,3,3), t (B,3)."""
    c = np.einsum("bij,nj->bni", R, X) + t[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.hypot(c[..., 0] / c[..., 2] - x[None, :, 0], c[..., 1] / c[..., 2] - x[None, :, 1])
    return (c[..., 2] > 0) & (e < thr)


def inliers_of(pose, X, x, thr) -> np.ndarray:
    return np.nonzero(_inlier_mask(pose[0][None], np.asarray(pose[1])[None], X, x, thr)[0])[0]


def solve_pnp(X: np.ndarray, x: np.ndarray, thr: float, init=None, seed: int = 1):
    """Registers a camera from (normalized) 2D-3D correspondences: refines the initial
    pose (video: the neighbour's) and runs a DLT RANSAC; keeps whichever has more
    inliers. Returns ((R, t), inlier indices) or None."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
    n = len(X)
    if n < 6:
        return None
    best = None
    if init is not None:
        p = refine_pose(init, X, x, thr * 2, 15)
        best = (p, inliers_of(p, X, x, thr))
    rand = rng(seed)
    max_iter = 500
    it = 0
    while it < max_iter:
        m = min(_CHUNK, max_iter - it)
        idx = np.array([sample(n, 6, rand) for _ in range(m)])
        Rs, ts, ok = dlt_pose(X, x, idx)
        masks = _inlier_mask(Rs, ts, X, x, thr)
        counts = masks.sum(1)
        for j in range(m):
            if it >= max_iter:
                break
            it += 1
            if not ok[j]:
                continue
            if best is None or counts[j] > len(best[1]):
                best = ((Rs[j], ts[j]), np.nonzero(masks[j])[0])
                w = int(counts[j]) / n
                max_iter = _adaptive(w, 6, 500)
    if best is None:
        return None
    inl = best[1]
    pose = refine_pose(best[0], X[inl], x[inl], thr, 10)
    return pose, inliers_of(pose, X, x, thr)
