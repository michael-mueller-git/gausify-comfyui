"""Small dense linear algebra for the SfM (numpy, batched where useful).

Port of web/src/sfm/linalg.ts. Matrices are numpy arrays; 3x3 rotations are (3,3)
float64. The Jacobi eigen-solver of the TS is replaced by LAPACK (`numpy.linalg.eigh`),
same ordering (ascending eigenvalues, eigenvectors as columns).
"""

from __future__ import annotations

import numpy as np


def skew(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def exp_so3(w) -> np.ndarray:
    """exp([w]x): rotation from an axis-angle vector (Rodrigues, same as expSO3)."""
    w = np.asarray(w, dtype=np.float64)
    th = float(np.sqrt(w @ w))
    K = skew(w)
    K2 = K @ K
    a = 1.0 if th < 1e-10 else np.sin(th) / th
    b = 0.5 if th < 1e-10 else (1 - np.cos(th)) / (th * th)
    return np.eye(3) + a * K + b * K2


def exp_so3_batch(w: np.ndarray) -> np.ndarray:
    """Batched exp_so3: (n,3) -> (n,3,3)."""
    w = np.asarray(w, dtype=np.float64).reshape(-1, 3)
    th = np.sqrt((w * w).sum(1))
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
    K2 = K @ K
    small = th < 1e-10
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0, np.sin(ths) / ths)
    b = np.where(small, 0.5, (1 - np.cos(ths)) / (ths * ths))
    return np.eye(3)[None] + a[:, None, None] * K + b[:, None, None] * K2


def quat_from_mat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) of a rotation, as COLMAP stores it (w >= 0)."""
    r = np.asarray(R, dtype=np.float64).ravel()
    tr = r[0] + r[4] + r[8]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        w, x, y, z = 0.25 * s, (r[7] - r[5]) / s, (r[2] - r[6]) / s, (r[3] - r[1]) / s
    elif r[0] > r[4] and r[0] > r[8]:
        s = np.sqrt(1 + r[0] - r[4] - r[8]) * 2
        w, x, y, z = (r[7] - r[5]) / s, 0.25 * s, (r[1] + r[3]) / s, (r[2] + r[6]) / s
    elif r[4] > r[8]:
        s = np.sqrt(1 + r[4] - r[0] - r[8]) * 2
        w, x, y, z = (r[2] - r[6]) / s, (r[1] + r[3]) / s, 0.25 * s, (r[5] + r[7]) / s
    else:
        s = np.sqrt(1 + r[8] - r[0] - r[4]) * 2
        w, x, y, z = (r[3] - r[1]) / s, (r[2] + r[6]) / s, (r[5] + r[7]) / s, 0.25 * s
    n = np.sqrt(w * w + x * x + y * y + z * z)
    q = (w / n, x / n, y / n, z / n)
    return tuple(-v for v in q) if w < 0 else q  # type: ignore[return-value]


def sym_eig(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) and eigenvectors (columns) of symmetric matrices (batched)."""
    A = np.asarray(A, dtype=np.float64)
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    return np.linalg.eigh(A)


def null_vector(A: np.ndarray) -> np.ndarray:
    """Eigenvector of the smallest eigenvalue of AᵀA. A: (..., rows, cols)."""
    A = np.asarray(A, dtype=np.float64)
    AtA = np.swapaxes(A, -1, -2) @ A
    return sym_eig(AtA)[1][..., :, 0]


def svd3(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3x3 SVD as svd3() in linalg.ts: M = U diag(S) Vᵀ, S descending, U built from
    M v₁, M v₂ and their cross product (det U = +1)."""
    M = np.asarray(M, dtype=np.float64)
    values, Ve = sym_eig(M.T @ M)
    V = Ve[:, ::-1].copy()
    S = np.sqrt(np.maximum(values[::-1], 0))
    cols = []
    for k in range(2):
        u = M @ V[:, k]
        n = float(np.sqrt(u @ u)) or 1.0
        cols.append(u / n)
    cols.append(np.cross(cols[0], cols[1]))
    U = np.stack(cols, axis=1)
    return U, S, V


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    """Nearest rotation to M (polar decomposition via SVD), batched over leading dims."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    if np.any(neg):
        U = U.copy()
        U[..., :, 2] = np.where(neg[..., None], -U[..., :, 2], U[..., :, 2])
        R = U @ Vt
    return R


def inv3(a: np.ndarray) -> np.ndarray | None:
    """Inverse of a 3x3 (adjugate), None when |det| <= 1e-300 (as inv3 in the TS)."""
    a = np.asarray(a, dtype=np.float64)
    d = float(np.linalg.det(a))
    if not abs(d) > 1e-300:
        return None
    return np.linalg.inv(a)


def center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera center -Rᵀ t (batched)."""
    return -np.einsum("...ji,...j->...i", R, t)
