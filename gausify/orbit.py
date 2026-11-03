"""Post-SfM orbit corrections: flat orbit (flatten_orbit) and inferred poses for
frames the reconstruction did not register (fill_missing), plus the 2D-3D
correspondences used to refine those inferred poses.

Port of gausify-orbit web/src/sfm/orbit.ts (same constants, thresholds and
order of operations). The points-only bundle adjustment of `flattenOrbit`
(`bundleAdjust` of web/src/sfm/ba.ts with `varCams: []`, `refineFocal: false`,
`huber: 2`, `iters: 15`) is reimplemented here (`_bundle_adjust_points`), with
the same Levenberg-Marquardt schedule, so this module does not depend on
gausify/sfm_akaze.

Matches structure (contract shared with gausify/sfm_akaze and gausify/sfm.py),
the Python form of `PairMatches` in web/src/sfm/matcher.ts::

    matches: list[dict] = [
        {"a": int, "b": int, "idx": np.ndarray (M, 2) int32},
        ...
    ]

`a`, `b` are image indices (orbit order, as in `orbit_pairs`); `idx[:, 0]`
are keypoint indices in image `a` (= `ia`), `idx[:, 1]` in image `b`
(= `ib`), indices into `model.xy[a]` / `model.xy[b]`. For convenience, dicts
with separate "ia"/"ib" arrays are accepted too.

Keypoints: `xy[i]` is an (N, 2) array of pixel coordinates of image i
(`SfmModel.xy`); `Point3D.obs` holds (image index, keypoint index) into it.
"""

from __future__ import annotations

import math

import numpy as np

from .types import Pose, SfmModel

# ---------------------------------------------------------------- small math


def _v(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(3)


def _unit(a) -> np.ndarray:
    a = _v(a)
    return a * (1 / max(math.hypot(a[0], a[1], a[2]), 1e-12))


def _norm(a) -> float:
    return math.hypot(float(a[0]), float(a[1]), float(a[2]))


def _skew(w) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(w) -> np.ndarray:
    """expSO3 of web/src/sfm/linalg.ts."""
    w = _v(w)
    th = _norm(w)
    K = _skew(w)
    K2 = K @ K
    a = 1.0 if th < 1e-10 else math.sin(th) / th
    b = 0.5 if th < 1e-10 else (1 - math.cos(th)) / (th * th)
    return np.eye(3) + a * K + b * K2


def quat_from_mat(R) -> list[float]:
    """quatFromMat of web/src/sfm/linalg.ts: (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64).reshape(9)
    tr = R[0] + R[4] + R[8]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        w, x, y, z = 0.25 * s, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s
    elif R[0] > R[4] and R[0] > R[8]:
        s = math.sqrt(1 + R[0] - R[4] - R[8]) * 2
        w, x, y, z = (R[7] - R[5]) / s, 0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s
    elif R[4] > R[8]:
        s = math.sqrt(1 + R[4] - R[0] - R[8]) * 2
        w, x, y, z = (R[2] - R[6]) / s, (R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s
    else:
        s = math.sqrt(1 + R[8] - R[0] - R[4]) * 2
        w, x, y, z = (R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    q = [float(w) / n, float(x) / n, float(y) / n, float(z) / n]
    return [-v for v in q] if q[0] < 0 else q


def _mat_from_quat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _center_of(p: Pose) -> np.ndarray:
    return -(np.asarray(p.R, dtype=np.float64).T @ _v(p.t))


def _pose_at(R: np.ndarray, c: np.ndarray) -> Pose:
    return Pose(R=np.array(R, dtype=np.float64), t=-(R @ c))


def _median(v) -> float:
    """Upper median, like `median` in orbit.ts (s[len >> 1]); 0 if empty."""
    v = list(v)
    if not v:
        return 0.0
    s = sorted(v)
    return float(s[len(s) >> 1])


def _mean(centers) -> np.ndarray:
    acc = np.zeros(3)
    for c in centers:
        acc = acc + c
    return acc * (1 / len(centers))


def _target(model: SfmModel, centers) -> np.ndarray:
    """Where the cameras look: median of the points (like scene.rs: read)."""
    if not model.points:
        return _mean(centers)
    X = np.array([_v(p.X) for p in model.points])
    return np.array([_median(X[:, k]) for k in range(3)])


def _orbit_normal(centers) -> np.ndarray | None:
    """Orbit plane normal (Newell), or None if the cameras don't form an arc."""
    k = len(centers)
    mean = _mean(centers)
    n = np.zeros(3)
    for i in range(1, k):
        n = n + np.cross(centers[i - 1] - mean, centers[i] - mean)
    spread = sum(float(np.dot(c - mean, c - mean)) for c in centers) / k
    return None if _norm(n) < 1e-3 * spread * k else _unit(n)


def _rotation_between(a, b) -> np.ndarray:
    """Smallest rotation taking unit vector `a` to `b` (Rodrigues)."""
    v = np.cross(a, b)
    K = _skew(v)
    s = 1 / (1 + float(np.dot(a, b)))
    return np.eye(3) + K + (K @ K) * s


def _about(n, th: float) -> np.ndarray:
    return exp_so3(_v(n) * th)


def _rotation_angle(a, b) -> float:
    M = a @ b.T
    return math.acos(min(1.0, max(-1.0, (M[0, 0] + M[1, 1] + M[2, 2] - 1) / 2)))


def _keypoints(xy, i) -> np.ndarray:
    return np.asarray(xy[i], dtype=np.float64).reshape(-1, 2)


# --------------------------------------------------- points-only bundle adjust


def _huber_cost(e2: np.ndarray, d: float) -> np.ndarray:
    e = np.sqrt(e2)
    return np.where(e <= d, e2, 2 * d * e - d * d)


def _bundle_adjust_points(
    f: float,
    cx: float,
    cy: float,
    R: list,
    t: list,
    points: np.ndarray,
    obs_img: np.ndarray,
    obs_pt: np.ndarray,
    obs_uv: np.ndarray,
    huber: float,
    iters: int,
) -> dict:
    """`bundleAdjust` of web/src/sfm/ba.ts with no varying camera and fixed
    focal: LM on the points only (Schur complement degenerates to a 3x3 block
    per point), Huber IRLS in pixels, lambda 1e-4, /3 on success, x5 on
    failure, 8 tries per iteration, stop at relative decrease < 1e-6.
    `points` (P, 3) is modified in place."""
    n_obs = len(obs_img)
    Rs = np.stack([np.asarray(R[i], dtype=np.float64) for i in obs_img])  # (O,3,3)
    ts = np.stack([_v(t[i]) for i in obs_img])
    uv = np.asarray(obs_uv, dtype=np.float64).reshape(-1, 2)
    pt_list, inv = np.unique(obs_pt, return_inverse=True)  # order is irrelevant: blocks are independent
    np_ = len(pt_list)

    def eval_cost(X: np.ndarray):
        c = np.einsum("oij,oj->oi", Rs, X[obs_pt]) + ts
        bad = c[:, 2] <= 1e-9
        cz = np.where(bad, 1.0, c[:, 2])
        ex = f * c[:, 0] / cz + cx - uv[:, 0]
        ey = f * c[:, 1] / cz + cy - uv[:, 1]
        e2 = ex * ex + ey * ey
        cost = float(np.sum(np.where(bad, huber * 1e3, _huber_cost(e2, huber))))
        sq = float(np.sum(np.where(bad, 1e6, e2)))
        return cost, math.sqrt(sq / max(n_obs, 1))

    X = np.array(points, dtype=np.float64).reshape(-1, 3)
    cur = eval_cost(X)
    rms_before = cur[1]
    lam = 1e-4
    it = 0
    while it < iters:
        Xo = X[obs_pt]
        r = np.einsum("oij,oj->oi", Rs, Xo)
        c = r + ts
        ok = c[:, 2] > 1e-9
        iz = np.where(ok, 1 / np.where(ok, c[:, 2], 1.0), 0.0)
        ex = f * c[:, 0] * iz + cx - uv[:, 0]
        ey = f * c[:, 1] * iz + cy - uv[:, 1]
        e = np.hypot(ex, ey)
        w = np.where(ok, np.where(e <= huber, 1.0, huber / np.maximum(e, 1e-300)), 0.0)
        P00 = f * iz
        P02 = -f * c[:, 0] * iz * iz
        P12 = -f * c[:, 1] * iz * iz
        Jp = np.empty((n_obs, 2, 3))
        Jp[:, 0, :] = P00[:, None] * Rs[:, 0, :] + P02[:, None] * Rs[:, 2, :]
        Jp[:, 1, :] = P00[:, None] * Rs[:, 1, :] + P12[:, None] * Rs[:, 2, :]
        res = np.stack([ex, ey], axis=1)
        res[~ok] = 0
        V = np.zeros((np_, 3, 3))
        gp = np.zeros((np_, 3))
        np.add.at(V, inv, w[:, None, None] * np.einsum("oai,oaj->oij", Jp, Jp))
        np.add.at(gp, inv, w[:, None] * np.einsum("oai,oa->oi", Jp, res))

        accepted = False
        tries = 0
        while tries < 8 and not accepted:
            tries += 1
            Vd = V.copy()
            for a in range(3):
                Vd[:, a, a] = V[:, a, a] * (1 + lam) + 1e-9
            det = np.linalg.det(Vd)
            good = np.abs(det) > 1e-300
            Xn = X.copy()
            if good.any():
                dp = np.linalg.solve(Vd[good], -gp[good][:, :, None])[:, :, 0]
                Xn[pt_list[good]] += dp
            nxt = eval_cost(Xn)
            if nxt[0] < cur[0]:
                rel = (cur[0] - nxt[0]) / cur[0]
                X, cur = Xn, nxt
                lam = max(lam / 3, 1e-9)
                accepted = True
                if rel < 1e-6:
                    it = iters
            else:
                lam *= 5
        if not accepted:
            break
        it += 1
    points[...] = X.reshape(points.shape)
    return {"f": f, "rmsBefore": rms_before, "rmsAfter": cur[1], "iterations": it}


# -------------------------------------------------------------- flat orbit


def flatten_orbit(model: SfmModel, xy: list | None = None) -> float | None:
    """Forces a flat orbit: projects each camera center onto the orbit plane
    and rotates the camera so the subject stays at the same pixel; then
    re-fits only the 3D points (poses and focal fixed). `xy[i]`: keypoints of
    image i (default `model.xy`). Returns the largest height deviation
    removed, as a fraction of the radius, or None if the cameras don't form an
    arc. Updates `model.poses`, the points, their errors and
    `model.stats["meanError"]` (setting `stats["flat"]` is left to the caller,
    as in index.ts)."""
    xy = model.xy if xy is None else xy
    reg = [i for i, p in enumerate(model.poses) if p is not None]
    if len(reg) < 3:
        return None
    centers = [_center_of(model.poses[i]) for i in reg]
    normal = _orbit_normal(centers)
    if normal is None:
        return None
    o = _target(model, centers)
    mean = _mean(centers)
    max_h = 0.0
    for k, i in enumerate(reg):
        c = centers[k]
        h = float(np.dot(c - mean, normal))
        max_h = max(max_h, abs(h))
        c2 = c - normal * h
        R = np.asarray(model.poses[i].R, dtype=np.float64) @ _rotation_between(_unit(o - c2), _unit(o - c))
        model.poses[i] = _pose_at(R, c2)

    # Points: BA with only the points varying.
    obs_img, obs_pt, obs_uv = [], [], []
    for j, p in enumerate(model.points):
        for i, k in p.obs:
            obs_img.append(int(i))
            obs_pt.append(j)
            obs_uv.append(_keypoints(xy, i)[int(k)])
    points = np.array([_v(p.X) for p in model.points], dtype=np.float64).reshape(-1, 3)
    if obs_img:
        _bundle_adjust_points(
            model.f, model.cx, model.cy,
            [p.R if p is not None else None for p in model.poses],
            [p.t if p is not None else None for p in model.poses],
            points, np.array(obs_img), np.array(obs_pt), np.array(obs_uv),
            huber=2, iters=15,
        )
    err_sum, n_obs = 0.0, 0
    for j, p in enumerate(model.points):
        p.X = points[j].copy()
        e = 0.0
        for i, k in p.obs:
            pose = model.poses[i]
            c = np.asarray(pose.R) @ p.X + _v(pose.t)
            u = _keypoints(xy, i)[int(k)]
            e += (
                math.hypot(model.f * c[0] / c[2] + model.cx - u[0], model.f * c[1] / c[2] + model.cy - u[1])
                if c[2] > 0
                else 1e3
            )
        p.error = e / max(len(p.obs), 1)
        err_sum += e
        n_obs += len(p.obs)
    model.stats["meanError"] = err_sum / max(n_obs, 1)
    radius = _median([_norm(c - o) for c in centers])
    return max_h / max(radius, 1e-12)


# -------------------------------------------------------------- interpolation


def _slerp(a, b, u: float) -> list[float]:
    d = sum(a[k] * b[k] for k in range(4))
    s = -1 if d < 0 else 1
    d *= s
    wa, wb = 1 - u, u * s
    if d < 0.9995:
        th = math.acos(d)
        wa = math.sin((1 - u) * th) / math.sin(th)
        wb = (math.sin(u * th) / math.sin(th)) * s
    q = [wa * a[k] + wb * b[k] for k in range(4)]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def _hermite(s, y, a: int, b: int, at: float) -> float:
    """Cubic Hermite with non-uniform Catmull-Rom tangents."""

    def slope(i, j):
        return (y[j] - y[i]) / (s[j] - s[i])

    ma = slope(a - 1, b) if a > 0 else slope(a, b)
    mb = slope(a, b + 1) if b < len(s) - 1 else slope(a, b)
    h = s[b] - s[a]
    u = (at - s[a]) / h
    u2 = u * u
    u3 = u2 * u
    return (2 * u3 - 3 * u2 + 1) * y[a] + (u3 - 2 * u2 + u) * h * ma + (-2 * u3 + 3 * u2) * y[b] + (u3 - u2) * h * mb


def _fit_slope(s, y) -> float:
    n = len(s)
    ms = sum(s) / n
    my = sum(y) / n
    num = den = 0.0
    for i in range(n):
        num += (s[i] - ms) * (y[i] - my)
        den += (s[i] - ms) ** 2
    return num / den if den > 0 else 0.0


def _find_gt(s, t) -> int:
    """`s.findIndex((v) => v > t)`."""
    for i, v in enumerate(s):
        if v > t:
            return i
    return -1


def _orbit_at(s, th, rad, hgt, t) -> tuple[float, float, float]:
    """Orbit angle, radius and height at time `t` (Hermite between neighbours;
    extrapolated with the speed of the 4 end frames outside)."""
    b = _find_gt(s, t)
    if b > 0:
        return tuple(_hermite(s, y, b - 1, b, t) for y in (th, rad, hgt))  # type: ignore[return-value]
    end = 0 if b == 0 else len(s) - 1
    ks = [k for k in (0, 1, 2, 3) if k < len(s)] if end == 0 else [k for k in (end - 3, end - 2, end - 1, end) if k >= 0]

    def mean(y):
        return sum(y[k] for k in ks) / len(ks)

    return (
        th[end] + _fit_slope([s[k] for k in ks], [th[k] for k in ks]) * (t - s[end]),
        mean(rad),
        mean(hgt),
    )


# ---------------------------------------------------------------- refine pose

#: Typical noise of a matched 2D point (pixels).
KEYPOINT_SIGMA = 0.5
#: Minimum inliers (<= 4 px) to accept a refined pose.
REFINE_MIN_INLIERS = 15


def _reproj_all(model: SfmModel, R, t, uv: np.ndarray, X: np.ndarray) -> np.ndarray:
    p = X @ np.asarray(R).T + t
    bad = p[:, 2] <= 1e-9
    z = np.where(bad, 1.0, p[:, 2])
    e = np.hypot(model.f * p[:, 0] / z + model.cx - uv[:, 0], model.f * p[:, 1] / z + model.cy - uv[:, 1])
    return np.where(bad, np.inf, e)


def _solve6(A: list[list[float]], b: list[float]) -> list[float] | None:
    """6x6 `A x = b`, elimination with partial pivoting (solve6 in orbit.ts)."""
    for col in range(6):
        piv = col
        for r in range(col + 1, 6):
            if abs(A[r][col]) > abs(A[piv][col]):
                piv = r
        if abs(A[piv][col]) < 1e-12:
            return None
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, 6):
            f = A[r][col] / A[col][col]
            for k in range(col, 6):
                A[r][k] -= f * A[col][k]
            b[r] -= f * b[col]
    x = [0.0] * 6
    for r in range(5, -1, -1):
        acc = b[r]
        for k in range(r + 1, 6):
            acc -= A[r][k] * x[k]
        x[r] = acc / A[r][r]
    return x


def _corr_arrays(corr) -> tuple[np.ndarray, np.ndarray, int]:
    n = int(corr["n"])
    uv = np.asarray(corr["uv"], dtype=np.float64).reshape(-1, 2)[:n]
    X = np.asarray(corr["X"], dtype=np.float64).reshape(-1, 3)[:n]
    return uv, X, n


def refine_pose(model: SfmModel, R0, c0, sigma: float, corr: dict) -> dict | None:
    """Refines the inferred pose (world->camera rotation `R0`, center `c0`)
    against the correspondences `{"uv": (n,2), "X": (n,3), "n": n}`: rounds
    with thresholds 32/16/8/4 px, damped Gauss-Newton with Huber loss
    (delta = thr/2) and the center as a prior of deviation `sigma`.
    Returns {"pose": Pose, "inliers": int} or None. Same as refinePose."""
    uv, Xc, n = _corr_arrays(corr)
    R = np.array(R0, dtype=np.float64)
    c0 = _v(c0)
    t = -(R @ c0)
    prior = (KEYPOINT_SIGMA / max(sigma, 1e-12)) ** 2

    def center_at(R, t):
        return -(R.T @ t)

    for thr in (32, 16, 8, 4):
        sel = np.nonzero(_reproj_all(model, R, t, uv, Xc) < thr)[0]
        if len(sel) < REFINE_MIN_INLIERS:
            return None
        delta = thr / 2
        Xs, uvs = Xc[sel], uv[sel]

        def cost(R, t):
            d = center_at(R, t) - c0
            e = np.minimum(_reproj_all(model, R, t, uvs, Xs), 1e3)
            return prior * float(np.dot(d, d)) + float(np.sum(np.where(e <= delta, e * e, 2 * delta * e - delta * delta)))

        lam = 1e-3
        current = cost(R, t)
        for _ in range(100):
            RX = Xs @ R.T
            p = RX + t
            ok = p[:, 2] > 1e-9
            RX, p, uvo = RX[ok], p[ok], uvs[ok]
            iz = 1 / p[:, 2]
            e0 = model.f * p[:, 0] * iz + model.cx - uvo[:, 0]
            e1 = model.f * p[:, 1] * iz + model.cy - uvo[:, 1]
            nrm = np.hypot(e0, e1)
            w = np.where(nrm <= delta, 1.0, delta / np.maximum(nrm, 1e-300))
            m = len(p)
            jp = np.zeros((m, 2, 3))
            jp[:, 0, 0] = model.f * iz
            jp[:, 0, 2] = -model.f * p[:, 0] * iz * iz
            jp[:, 1, 1] = model.f * iz
            jp[:, 1, 2] = -model.f * p[:, 1] * iz * iz
            sk = np.zeros((m, 3, 3))
            sk[:, 0, 1], sk[:, 0, 2] = RX[:, 2], -RX[:, 1]
            sk[:, 1, 0], sk[:, 1, 2] = -RX[:, 2], RX[:, 0]
            sk[:, 2, 0], sk[:, 2, 1] = RX[:, 1], -RX[:, 0]
            J = np.concatenate([np.einsum("mak,mkb->mab", jp, sk), jp], axis=2)  # (m,2,6)
            ev = np.stack([e0, e1], axis=1)
            g = np.einsum("m,mai,ma->i", w, J, ev)
            H = np.einsum("m,mai,maj->ij", w, J, J)
            Rt = R.T
            d = center_at(R, t) - c0
            Rtx = Rt @ _skew(t)
            Jc = np.concatenate([-Rtx, -Rt], axis=1)  # (3,6)
            g = g + prior * (Jc.T @ d)
            H = H + prior * (Jc.T @ Jc)
            damped = [[float(H[a, b]) + (lam * max(float(H[a, b]), 1e-9) if a == b else 0.0) for b in range(6)] for a in range(6)]
            step = _solve6(damped, [-float(v) for v in g])
            if step is None:
                break
            R2 = exp_so3(step[:3]) @ R
            t2 = t + np.array(step[3:])
            nxt = cost(R2, t2)
            if nxt < current:
                done = current - nxt < 1e-9 * max(current, 1e-12)
                R, t, current = R2, t2, nxt
                lam = max(lam / 10, 1e-9)
                if done:
                    break
            else:
                lam *= 10
    inliers = int(np.sum(_reproj_all(model, R, t, uv, Xc) < 4))
    return {"pose": Pose(R=R, t=t), "inliers": inliers} if inliers >= REFINE_MIN_INLIERS else None


# --------------------------------------------------------------- fill missing


def fill_missing_at(
    model: SfmModel,
    at: list[float],
    flat: bool = False,
    corr: dict[int, dict] | None = None,
) -> dict:
    """`fillMissing(model, at, {flat, correspondences})`: poses for the
    unregistered frames from their neighbours in orbit coordinates (cubic
    Hermite at each frame's timestamp `at[i]`, slerped orientation relative
    to the orbit, extrapolation at the ends), then refined against the 3D
    points (`refine_pose`) when `corr` has an entry for the frame.
    Returns {"inferred": [i, ...], "refined": [(i, inliers), ...]}."""
    reg = [i for i, p in enumerate(model.poses) if p is not None]
    missing = [i for i, p in enumerate(model.poses) if p is None]
    refined: list[tuple[int, int]] = []
    if len(reg) < 3 or not missing:
        return {"inferred": [], "refined": refined}
    centers = [_center_of(model.poses[i]) for i in reg]
    s = [float(at[i]) for i in reg]
    nn = _orbit_normal(centers)
    n = nn if nn is not None else np.zeros(3)
    linear = not bool(np.any(n != 0))
    o = _target(model, centers)
    radius = _median([_norm(c - o) for c in centers])

    # Plane basis: e1 points toward the first camera.
    d0 = centers[0] - o
    e1 = _unit(d0 - n * float(np.dot(d0, n)))
    e2 = np.cross(n, e1)
    th: list[float] = []
    rad: list[float] = []
    hgt: list[float] = []
    raw: list[float] = []
    for c in centers:
        d = c - o
        hh = float(np.dot(d, n))
        p = d - n * hh
        rad.append(_norm(p))
        hgt.append(hh)
        raw.append(math.atan2(float(np.dot(p, e2)), float(np.dot(p, e1))))

    def wrap(a):
        return a - 2 * math.pi * _js_round(a / (2 * math.pi))

    speed = _median([wrap(raw[k + 1] - raw[k]) / max(s[k + 1] - s[k], 1e-9) for k in range(len(raw) - 1)])
    for k, a in enumerate(raw):
        expected = th[k - 1] + speed * (s[k] - s[k - 1]) if k else a
        th.append(a + 2 * math.pi * _js_round((expected - a) / (2 * math.pi)))
    # "On-orbit" orientation: the camera rotated back to theta = 0.
    q0: list[list[float]] = []
    for k, i in enumerate(reg):
        R = np.asarray(model.poses[i].R, dtype=np.float64)
        R0 = R if linear else R @ _about(n, th[k])
        q = quat_from_mat(R0)
        if q0 and sum(q0[-1][j] * q[j] for j in range(4)) < 0:
            q = [-v for v in q]
        q0.append(q)

    def at_center(orb):
        theta, r, h = orb
        return o + e1 * (r * math.cos(theta)) + e2 * (r * math.sin(theta)) + n * h

    def loo(k):
        def drop(y):
            return [v for j, v in enumerate(y) if j != k]

        return _norm(at_center(_orbit_at(drop(s), drop(th), drop(rad), drop(hgt), s[k])) - centers[k])

    last = len(reg) - 1
    step = _median([s[k + 1] - s[k] for k in range(len(s) - 1)])
    if linear or len(reg) < 6:
        err_mid = err_end = 0.05 * radius
    else:
        err_mid = _median([loo(k + 1) for k in range(last - 1)])
        err_end = max(loo(0), loo(last))

    def sigma_at(t):
        b = _find_gt(s, t)
        if b > 0:
            sigma = err_mid * max(1, (s[b] - s[b - 1]) / (2 * step))
        else:
            sigma = err_end * max(1, abs(t - s[0 if b == 0 else last]) / step) ** 2
        return max(sigma, 1e-4 * radius)

    for i in missing:
        t = float(at[i])
        b = _find_gt(s, t)
        if b > 0:
            u = (t - s[b - 1]) / (s[b] - s[b - 1])
            q = _slerp(q0[b - 1], q0[b], u)
            lin = centers[b - 1] + (centers[b] - centers[b - 1]) * u
        else:
            end = 0 if b == 0 else last
            k0, k1 = (0, min(3, last)) if end == 0 else (max(0, last - 3), last)
            q = q0[end]
            lin = centers[end] + (centers[k1] - centers[k0]) * ((t - s[end]) / (s[k1] - s[k0]))
        orbit = _orbit_at(s, th, rad, hgt, t)
        R = _mat_from_quat(q) if linear else _mat_from_quat(q) @ _about(n, orbit[0]).T
        c = lin if linear else at_center(orbit)
        cr = corr.get(i) if corr else None
        r = refine_pose(model, R, c, sigma_at(t), cr) if cr is not None else None
        if r:
            c2 = _center_of(r["pose"])
            # Too far from the trajectory: bad matches, keep the inferred pose.
            if _norm(c2 - c) <= 0.2 * radius and _rotation_angle(r["pose"].R, R) <= 15 * math.pi / 180:
                R, c = r["pose"].R, c2
                if flat and not linear:
                    # Back onto the plane, with the subject at the same pixel.
                    fl = c - n * (float(np.dot(c - o, n)) - orbit[2])
                    R = R @ _rotation_between(_unit(o - fl), _unit(o - c))
                    c = fl
                refined.append((i, r["inliers"]))
        model.poses[i] = _pose_at(R, c)
    return {"inferred": missing, "refined": refined}


def _js_round(x: float) -> float:
    """Math.round: halves go toward +infinity."""
    return math.floor(x + 0.5)


def _match_idx(m: dict) -> tuple[np.ndarray, np.ndarray]:
    if "idx" in m:
        idx = np.asarray(m["idx"]).reshape(-1, 2)
        return idx[:, 0], idx[:, 1]
    return np.asarray(m["ia"]).reshape(-1), np.asarray(m["ib"]).reshape(-1)


def correspondences(model: SfmModel, xy: list | None, matches: list[dict]) -> dict[int, dict]:
    """Correspondences for unregistered frames: their matches with registered
    frames, carried over to the 3D points those frames see. A keypoint
    matched to different points keeps the most voted one (first on ties).
    Returns {frame: {"uv": (n,2), "X": (n,3), "n": n}}. Same as
    `correspondences` in orbit.ts."""
    xy = model.xy if xy is None else xy
    seen: dict[int, dict[int, int]] = {}
    for pi, p in enumerate(model.points):
        for i, k in p.obs:
            seen.setdefault(int(i), {})[int(k)] = pi
    votes: dict[int, dict[int, dict[int, int]]] = {}
    for m in matches:
        a, b = int(m["a"]), int(m["b"])
        a_reg = model.poses[a] is not None and a in seen
        b_reg = model.poses[b] is not None and b in seen
        # Exactly one of the two outside the model.
        if a_reg == b_reg or (model.poses[b] if a_reg else model.poses[a]) is not None:
            continue
        ia, ib = _match_idx(m)
        missing, km, kr, reg = (b, ib, ia, a) if a_reg else (a, ia, ib, b)
        pts = seen[reg]
        v = votes.setdefault(missing, {})
        for j in range(len(km)):
            p = pts.get(int(kr[j]))
            if p is None:
                continue
            c = v.setdefault(int(km[j]), {})
            c[p] = c.get(p, 0) + 1
    out: dict[int, dict] = {}
    for i, v in votes.items():
        kp = _keypoints(xy, i)
        uv = np.zeros((len(v), 2))
        X = np.zeros((len(v), 3))
        for n, (k, c) in enumerate(v.items()):
            best = None
            for pt, cnt in c.items():
                if best is None or cnt > best[1]:
                    best = (pt, cnt)
            uv[n] = kp[k]
            X[n] = _v(model.points[best[0]].X)
        out[i] = {"uv": uv, "X": X, "n": len(v)}
    return out


def fill_missing(
    model: SfmModel,
    matches: list[dict] | None = None,
    times: list[float] | None = None,
    *,
    flat: bool = False,
    refine: bool = True,
    xy: list | None = None,
) -> dict:
    """The `fillMissing` step of runSfm (web/src/sfm/index.ts): `times` is
    `orbitTimes(names, times)` (gausify.pairs.orbit_times; default: frame
    index); correspondences come from `matches` (structure in the module
    docstring) and `xy` (default `model.xy`) unless `refine` is False or no
    matches are given. Returns {"inferred": [...], "refined": [(i, n), ...]};
    the caller sets model.stats["inferred"/"refined"] like index.ts."""
    at = list(range(len(model.poses))) if times is None else list(times)
    corr = correspondences(model, xy, matches) if (refine and matches is not None) else None
    return fill_missing_at(model, at, flat=flat, corr=corr)
