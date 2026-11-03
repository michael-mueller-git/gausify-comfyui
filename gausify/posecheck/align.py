"""Pose correction of one frame against a render of a model that never saw it
(the other fold): a 6-DOF Gauss-Newton step that moves the camera so that
 - contour: the render's silhouette edge lands on the frame's mask edge (edge
   points lifted to 3D with the render's depth, residual = the mask's signed
   distance where they project), and
 - interior: the render's texture lands on the frame's (direct photometric
   alignment of points inside both silhouettes, coarse to fine).
Both blocks carry the same total weight.

Port of web/src/posecheck/align.ts (gausify-orbit), vectorized with numpy (the
per-point sums are the same; only the floating-point summation order differs).
`exp_so3`, `cholesky_solve` and `quat_from_mat` port web/src/sfm/linalg.ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .image import bilinear, blur, erode_cross, halve, signed_distance, sobel

# Gray composited over this (the background never enters the interior term).
BACKGROUND = 17 / 255


@dataclass(frozen=True)
class AlignOptions:
    levels: int = 3  # pyramid levels of the interior term
    iterations: int = 4  # Gauss-Newton iterations per level
    contour_huber: float = 2.0  # px at 640 px: Huber width
    contour_cut: float = 15.0  # px at 640 px: cutoff
    interior_huber: float = 1.0
    interior_cut: float = 8.0
    interior_weight: float = 1.0  # interior block weight relative to the contour's
    grid: int = 3  # one interior point per grid^2 pixels
    min_gradient: float = 0.01  # gray gradient per px
    contour_bias: float = 0.0  # model's halo (px, > 0 outside), see edge_offset
    damping: float = 0.05  # pull toward the render's pose, fraction of mean curvature


DEFAULT_ALIGN = AlignOptions()


@dataclass
class Level:
    w: int
    h: int
    gray: np.ndarray
    gx: np.ndarray
    gy: np.ndarray


@dataclass
class FrameTarget:
    w: int
    h: int
    mask: np.ndarray  # (h, w) uint8 0/1 (alpha > 0.5)
    sdf: np.ndarray  # signed distance, inside shifted by +1
    sdf_gx: np.ndarray
    sdf_gy: np.ndarray
    levels: list[Level] = field(default_factory=list)  # gray pyramid, finest first


@dataclass
class StepResult:
    dx: np.ndarray  # camera-frame update (omega, v): R <- exp(omega) R, t <- exp(omega) t + v
    contour_points: int
    interior_points: int
    contour_offset: float  # edge_offset before the step (px; > 0: the model sticks out)
    cost: float  # aligned loss at the render's own pose


# --- linalg (web/src/sfm/linalg.ts) -----------------------------------------


def exp_so3(w) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)[:3]
    th = float(np.sqrt(w @ w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    a = 1.0 if th < 1e-10 else np.sin(th) / th
    b = 0.5 if th < 1e-10 else (1 - np.cos(th)) / (th * th)
    return np.eye(3) + a * K + b * (K @ K)


def quat_from_mat(R) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) of a rotation, as COLMAP stores it."""
    R = np.asarray(R, dtype=np.float64).reshape(9)
    tr = R[0] + R[4] + R[8]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        w, x, y, z = 0.25 * s, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s
    elif R[0] > R[4] and R[0] > R[8]:
        s = np.sqrt(1 + R[0] - R[4] - R[8]) * 2
        w, x, y, z = (R[7] - R[5]) / s, 0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s
    elif R[4] > R[8]:
        s = np.sqrt(1 + R[4] - R[0] - R[8]) * 2
        w, x, y, z = (R[2] - R[6]) / s, (R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s
    else:
        s = np.sqrt(1 + R[8] - R[0] - R[4]) * 2
        w, x, y, z = (R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    q = (w / n, x / n, y / n, z / n)
    return tuple(float(-c) for c in q) if w < 0 else tuple(float(c) for c in q)


def rotation_from_quat(q) -> np.ndarray:
    """Rotation of a COLMAP quaternion (w, x, y, z) (web/src/mesh/colmap.ts)."""
    w, x, y, z = (float(c) for c in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Solve A x = b for a symmetric positive definite A; None if it is not."""
    n = len(b)
    L = np.array(A, dtype=np.float64)
    for j in range(n):
        d = L[j, j] - np.dot(L[j, :j], L[j, :j])
        if not d > 0:
            return None
        d = np.sqrt(d)
        L[j, j] = d
        for i in range(j + 1, n):
            L[i, j] = (L[i, j] - np.dot(L[i, :j], L[j, :j])) / d
    y = np.array(b, dtype=np.float64)
    for i in range(n):
        y[i] = (y[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    for i in range(n - 1, -1, -1):
        y[i] = (y[i] - np.dot(L[i + 1 :, i], y[i + 1 :])) / L[i, i]
    return y


# --- frame preparation -------------------------------------------------------


def pyramid(gray: np.ndarray, n: int) -> list[Level]:
    out: list[Level] = []
    img = blur(gray)
    for lv in range(n):
        gx, gy = sobel(img)
        out.append(Level(img.shape[1], img.shape[0], img, gx, gy))
        if lv + 1 < n:
            img = blur(halve(img))
    return out


def prepare_target(rgba: np.ndarray, levels: int = DEFAULT_ALIGN.levels) -> FrameTarget:
    """`rgba`: (h, w, 4) the frame with the mask in alpha (straight, 0-255)."""
    rgba = np.asarray(rgba, dtype=np.float64)
    h, w = rgba.shape[:2]
    a = rgba[..., 3] / 255
    mask = (a > 0.5).astype(np.uint8)
    c = (rgba[..., 0] + rgba[..., 1] + rgba[..., 2]) / (3 * 255)
    gray = (c * a + BACKGROUND * (1 - a)).astype(np.float32)
    # The signed distance is -1 on the mask's inner boundary: shifting the
    # inside by 1 makes an aligned edge pixel cost 0.
    sdf = signed_distance(mask)
    sdf = np.where(sdf < 0, sdf + 1, sdf).astype(np.float32)
    gx, gy = sobel(sdf)
    return FrameTarget(w, h, mask, sdf, gx, gy, pyramid(gray, levels))


# --- robust loss -------------------------------------------------------------


def huber(r, k: float, cut: float):
    """IRLS weight of the robust loss: Huber of width k, nothing beyond `cut`."""
    a = np.abs(np.asarray(r, dtype=np.float64))
    with np.errstate(divide="ignore"):
        return np.where(a >= cut, 0.0, np.where(a <= k, 1.0, k / np.maximum(a, 1e-300)))


def huber_cost(r, k: float, cut: float):
    """The loss itself (constant beyond `cut`)."""
    a = np.minimum(np.abs(np.asarray(r, dtype=np.float64)), cut)
    return np.where(a <= k, a * a / 2, k * (a - k / 2))


def _median(v: np.ndarray) -> float:
    n = len(v)
    if not n:
        return 0.0
    s = np.sort(v)
    return float(s[n >> 1]) if n % 2 else float((s[(n >> 1) - 1] + s[n >> 1]) / 2)


def edge_offset(alpha: np.ndarray, target: FrameTarget, cut: float | None = None) -> float:
    """Interquartile mean of the signed distance (px, > 0 outside the mask) of
    the silhouette's edge pixels (alpha > 0.5) to the mask's edge, among those
    within `cut` px. Over many frames, its median is the model's halo."""
    if cut is None:
        cut = DEFAULT_ALIGN.contour_cut * (max(target.w, target.h) / 640)
    rm = (np.asarray(alpha) > 0.5).astype(np.uint8)
    er = erode_cross(rm, 1)
    sel = (rm != 0) & (er == 0) & (np.abs(target.sdf) < cut)
    v = np.sort(target.sdf[sel].astype(np.float64))  # raster order, then sorted
    if not len(v):
        return 0.0
    a = len(v) // 4
    b = max(a + 1, int(np.ceil(3 * len(v) / 4)))
    return float(v[a:b].sum() / (b - a))


# --- Gauss-Newton ------------------------------------------------------------


def _proj_jacobian(p: np.ndarray, K: dict) -> tuple[np.ndarray, np.ndarray]:
    """d(projection)/d(omega, v) of camera points p (N,3) for a left update
    p' = exp(omega) p + v: rows u and v, (N,6) each."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    iz = 1 / z
    iz2 = iz * iz
    a = K["fx"] * iz
    c = -K["fx"] * x * iz2
    b = K["fy"] * iz
    d = -K["fy"] * y * iz2
    zero = np.zeros_like(x)
    ju = np.stack([c * y, a * z - c * x, -a * y, a, zero, c], axis=1)
    jv = np.stack([-b * z + d * y, -d * x, b * x, zero, b, d], axis=1)
    return ju, jv


def _in_bounds(u, v, w, h):
    return ~((u < 0) | (v < 0) | (u > w) | (v > h))


def _project(q: np.ndarray, K: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zpos = q[:, 2] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K["fx"] * q[:, 0] / q[:, 2] + K["cx"]
        v = K["fy"] * q[:, 1] / q[:, 2] + K["cy"]
    return u, v, zpos


def align_cost(render: dict, target: FrameTarget, K: dict, o: AlignOptions = DEFAULT_ALIGN) -> float:
    """StepResult.cost of the render's pose, without a step."""
    return align_step(render, target, K, replace(o, iterations=0)).cost


def align_step(render: dict, target: FrameTarget, K: dict, o: AlignOptions = DEFAULT_ALIGN) -> StepResult:
    """One alignment of the frame to the render (several Gauss-Newton
    iterations on a fixed render). `render`: {"rgba": (h,w,4) premultiplied
    colour + alpha, "depth": (h,w)}; `K`: {"fx","fy","cx","cy"} at that size."""
    w, h = target.w, target.h
    rgba = np.asarray(render["rgba"], dtype=np.float32)
    depth = np.asarray(render["depth"], dtype=np.float32)
    if rgba.shape[0] != h or rgba.shape[1] != w:
        raise ValueError("render and frame sizes differ")
    alpha = rgba[..., 3]
    gray = ((rgba[..., 0] + rgba[..., 1] + rgba[..., 2]) / 3 + BACKGROUND * (1 - np.minimum(alpha, 1))).astype(np.float32)
    rm = (alpha > 0.5).astype(np.uint8)
    scale = max(w, h) / 640

    def back(xs, ys, d):
        return np.stack(
            [(xs + 0.5 - K["cx"]) / K["fx"] * d, (ys + 0.5 - K["cy"]) / K["fy"] * d, d.astype(np.float64)], axis=1
        ).astype(np.float64)

    # Contour points: the render's edge pixels, depth read 2 px inside.
    er = erode_cross(rm, 1)
    ax, ay = sobel(alpha)
    ys, xs = np.nonzero((rm != 0) & (er == 0))
    axe = ax[ys, xs].astype(np.float64)
    aye = ay[ys, xs].astype(np.float64)
    gn = np.hypot(axe, aye) + 1e-6
    ix = np.clip(np.floor(xs + 2 * axe / gn + 0.5), 0, w - 1).astype(np.int64)  # JS Math.round
    iy = np.clip(np.floor(ys + 2 * aye / gn + 0.5), 0, h - 1).astype(np.int64)
    dc = depth[iy, ix]
    keep = dc > 0
    Pc = back(xs[keep].astype(np.float64), ys[keep].astype(np.float64), dc[keep])
    kc = len(Pc)
    contour_offset = edge_offset(alpha, target, o.contour_cut * scale)

    # Interior points: well inside the render's silhouette and inside the mask.
    inner = erode_cross(rm, 3)
    rgx, rgy = sobel(blur(gray))
    sub = (
        (inner != 0) & (target.mask != 0) & (depth > 0) & (np.hypot(rgx.astype(np.float64), rgy.astype(np.float64)) >= o.min_gradient)
    )[:: o.grid, :: o.grid]
    py, px = np.nonzero(sub)
    py, px = py * o.grid, px * o.grid
    nf = len(px) if o.interior_weight > 0 else 0
    px, py = px[:nf].astype(np.float64), py[:nf].astype(np.float64)
    Pf = back(px, py, depth[py.astype(np.int64), px.astype(np.int64)]) if nf else np.zeros((0, 3))
    L = min(o.levels, len(target.levels))
    ref = pyramid(gray, L)
    ref_at = [bilinear(lv.gray, (px + 0.5) / 2**li - 0.5, (py + 0.5) / 2**li - 0.5) for li, lv in enumerate(ref)]

    x6 = np.zeros(6)
    prior: np.ndarray | None = None
    proj = np.zeros((nf, 2))  # persists across iterations (TS reuses stale entries)
    eps = o.min_gradient / 2
    k_hub, k_cut = o.contour_huber * scale, o.contour_cut * scale

    # The loss at the render's pose, finest level.
    cost = 0.0
    if kc:
        u, v, _ = _project(Pc, K)
        ok = _in_bounds(u, v, w, h)
        if ok.any():
            r = bilinear(target.sdf, u[ok] - 0.5, v[ok] - 0.5) - o.contour_bias
            cost += float(huber_cost(r, k_hub, k_cut).sum()) / int(ok.sum())
    if nf:
        lv = target.levels[0]
        u, v, _ = _project(Pf, K)
        ok = _in_bounds(u, v, w, h)
        if ok.any():
            X, Y = u[ok] - 0.5, v[ok] - 0.5
            gx, gy = bilinear(lv.gx, X, Y), bilinear(lv.gy, X, Y)
            g = np.sqrt(gx * gx + gy * gy + eps * eps)
            resid = bilinear(lv.gray, X, Y) - ref_at[0][ok]
            bias = _median(resid)
            f = float(huber_cost((resid - bias) / g, o.interior_huber * scale, o.interior_cut * scale).sum())
            cost += o.interior_weight * f / int(ok.sum())

    for li in range(L - 1, -1, -1):
        lv = target.levels[li]
        s = 2**li
        f_hub, f_cut = o.interior_huber * scale * s, o.interior_cut * scale * s
        for _ in range(o.iterations):
            R = exp_so3(x6)
            Hc, gc, wc = np.zeros((6, 6)), np.zeros(6), 0.0
            Hf, gf, wf = np.zeros((6, 6)), np.zeros(6), 0.0
            if kc:
                q = Pc @ R.T + x6[3:]
                u, v, zpos = _project(q, K)
                ok = zpos & _in_bounds(u, v, w, h)
                q, u, v = q[ok], u[ok], v[ok]
                r = bilinear(target.sdf, u - 0.5, v - 0.5) - o.contour_bias
                wk = huber(r, k_hub, k_cut)
                nz = wk > 0
                if nz.any():
                    q, u, v, r, wk = q[nz], u[nz], v[nz], r[nz], wk[nz]
                    sx = bilinear(target.sdf_gx, u - 0.5, v - 0.5)
                    sy = bilinear(target.sdf_gy, u - 0.5, v - 0.5)
                    ju, jv = _proj_jacobian(q, K)
                    J = sx[:, None] * ju + sy[:, None] * jv
                    Hc = (J * wk[:, None]).T @ J
                    gc = (wk * r) @ J
                    wc = float(wk.sum())
            if nf:
                # First the residuals, to remove the median brightness offset.
                q = Pf @ R.T + x6[3:]
                u, v, zpos = _project(q, K)
                ok = zpos & _in_bounds(u, v, w, h)
                proj[ok, 0] = u[ok]
                proj[ok, 1] = v[ok]
                resid = bilinear(lv.gray, u[ok] / s - 0.5, v[ok] / s - 0.5) - ref_at[li][ok]
                bias = _median(resid)
                U, V = proj[:, 0], proj[:, 1]
                sel = zpos & _in_bounds(U, V, w, h)
                if sel.any():
                    X, Y = U[sel] / s - 0.5, V[sel] / s - 0.5
                    gx, gy = bilinear(lv.gx, X, Y), bilinear(lv.gy, X, Y)
                    gnf = np.sqrt(gx * gx + gy * gy + eps * eps)
                    # Normal flow in full-size px.
                    r = (bilinear(lv.gray, X, Y) - ref_at[li][sel] - bias) / gnf * s
                    wk = huber(r, f_hub, f_cut)
                    nz = wk > 0
                    if nz.any():
                        ju, jv = _proj_jacobian(q[sel][nz], K)
                        J = (gx[nz, None] * ju + gy[nz, None] * jv) / gnf[nz, None]
                        Hf = (J * wk[nz, None]).T @ J
                        gf = (wk[nz] * r[nz]) @ J
                        wf = float(wk[nz].sum())
            if not wc and not wf:
                break
            sc = 1 / wc if wc else 0.0
            sf = o.interior_weight / wf if wf else 0.0
            H = sc * Hc + sf * Hf
            if prior is None:
                rr = (H[0, 0] + H[1, 1] + H[2, 2]) / 3
                tt = (H[3, 3] + H[4, 4] + H[5, 5]) / 3
                prior = o.damping * np.array([rr, rr, rr, tt, tt, tt])
            g = sc * gc + sf * gf + prior * x6
            H[np.diag_indices(6)] += prior
            H[np.diag_indices(6)] += 1e-9 + 1e-6 * np.diag(H)
            dx = cholesky_solve(H, g)
            if dx is None:
                break
            x6 = x6 - dx
            if np.max(np.abs(dx)) < 1e-7:
                break
    return StepResult(x6, kc, nf, contour_offset, cost)


def choose_pose(costs, margin: float) -> int:
    """Index of the pose to keep (the first is the start): the cheapest, if it
    beats the start by the relative `margin`."""
    best = 0
    for i in range(1, len(costs)):
        if costs[i] < costs[best]:
            best = i
    return best if costs[best] < costs[0] * (1 - margin) else 0


def rotation_angle(A, B) -> float:
    """Angle between two rotations (radians)."""
    tr = float(np.sum(np.asarray(A) * np.asarray(B)))  # trace(A B^T)
    return float(np.arccos(min(1.0, max(-1.0, (tr - 1) / 2))))


def apply_step(R, t, dx) -> tuple[np.ndarray, np.ndarray]:
    """World -> camera pose after the update `dx` (see StepResult)."""
    dR = exp_so3(dx[:3])
    t2 = dR @ np.asarray(t, dtype=np.float64)
    return dR @ np.asarray(R, dtype=np.float64), t2 + np.asarray(dx[3:6], dtype=np.float64)
