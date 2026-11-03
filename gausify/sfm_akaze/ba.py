"""Bundle adjustment: Levenberg-Marquardt with a Schur complement. Port of web/src/sfm/ba.ts.

Shared SIMPLE_PINHOLE camera (focal f, fixed principal point), world->camera poses with
left update R <- exp(dr) R, t <- t + dt, Huber loss (IRLS) in pixels. Same damping,
λ schedule (start 1e-4, /3 on success, x5 on failure, x10 when Cholesky fails, 8 tries
per iteration) and stopping rule (relative decrease < 1e-6).

Vectorized in torch (float64; CUDA when available): Jacobians per observation, point
blocks inverted in closed form, reduced camera system assembled from all pairs of
observations of the same point and solved densely (Cholesky).
"""

from __future__ import annotations

import math

import numpy as np
import torch

_PAIR_CHUNK = 1 << 20


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _exp_so3(w: torch.Tensor) -> torch.Tensor:
    th = w.norm(dim=1)
    n = w.shape[0]
    K = torch.zeros((n, 3, 3), dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
    small = th < 1e-10
    ths = torch.where(small, torch.ones_like(th), th)
    a = torch.where(small, torch.ones_like(th), torch.sin(ths) / ths)
    b = torch.where(small, torch.full_like(th, 0.5), (1 - torch.cos(ths)) / (ths * ths))
    eye = torch.eye(3, dtype=w.dtype, device=w.device)[None]
    return eye + a[:, None, None] * K + b[:, None, None] * (K @ K)


def _inv3(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched 3x3 inverse by the adjugate; zero (and ok=False) when |det| <= 1e-300."""
    m = a.reshape(-1, 9)
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = m.unbind(1)
    c0 = a4 * a8 - a5 * a7
    c1 = a3 * a8 - a5 * a6
    c2 = a3 * a7 - a4 * a6
    d = a0 * c0 - a1 * c1 + a2 * c2
    ok = d.abs() > 1e-300
    idet = torch.where(ok, 1 / torch.where(ok, d, torch.ones_like(d)), torch.zeros_like(d))
    inv = torch.stack(
        [c0, a2 * a7 - a1 * a8, a1 * a5 - a2 * a4, -c1, a0 * a8 - a2 * a6, a2 * a3 - a0 * a5, c2, a1 * a6 - a0 * a7, a0 * a4 - a1 * a3],
        1,
    ) * idet[:, None]
    return inv.reshape(-1, 3, 3), ok


def bundle_adjust(
    f: float,
    cx: float,
    cy: float,
    R: np.ndarray,
    t: np.ndarray,
    var_cams,
    points: np.ndarray,
    obs_img: np.ndarray,
    obs_pt: np.ndarray,
    obs_uv: np.ndarray,
    refine_focal: bool,
    huber: float,
    iters: int,
    device: torch.device | None = None,
) -> dict:
    """R (N,3,3), t (N,3) and points (P,3) are updated in place (varying cameras and all
    observed points). Returns {"f", "rmsBefore", "rmsAfter", "iterations"}."""
    obs_img = np.asarray(obs_img, dtype=np.int64)
    obs_pt = np.asarray(obs_pt, dtype=np.int64)
    n_obs = len(obs_img)
    if n_obs == 0:
        return {"f": f, "rmsBefore": 0.0, "rmsAfter": 0.0, "iterations": 0}
    dev = device or _device()
    dt = torch.float64
    var_cams = np.asarray(var_cams, dtype=np.int64)
    nc = len(var_cams)
    rf = 1 if refine_focal else 0
    nC = 6 * nc + rf
    fI = 6 * nc
    cam_idx = np.full(len(R), -1, dtype=np.int64)
    cam_idx[var_cams] = np.arange(nc)
    upts, inv = np.unique(obs_pt, return_inverse=True)
    npt = len(upts)

    def T(a, dtype=dt):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)

    Rt = T(R)
    tt = T(t)
    X = T(points[upts])
    oi = T(obs_img, torch.int64)
    op = T(inv, torch.int64)
    uv = T(np.asarray(obs_uv, dtype=np.float64).reshape(-1, 2))
    oc = cam_idx[obs_img]
    var_o_np = np.nonzero(oc >= 0)[0]
    vo = T(var_o_np, torch.int64)
    vc = T(oc[var_o_np], torch.int64)
    vp = op[vo]
    nv = len(var_o_np)

    # pairs of var-camera observations of the same point, in chunks of whole points
    vp_np = inv[var_o_np]
    srt = np.argsort(vp_np, kind="stable")
    _, gstart, gcount = np.unique(vp_np[srt], return_index=True, return_counts=True)
    chunks = []
    if nv:
        cum = np.cumsum(gcount.astype(np.int64) ** 2)
        g0 = 0
        while g0 < len(gcount):
            base = cum[g0 - 1] if g0 else 0
            g1 = int(np.searchsorted(cum, base + _PAIR_CHUNK, side="right"))
            g1 = max(g1, g0 + 1)
            chunks.append((g0, g1))
            g0 = g1
    srt_t = T(srt, torch.int64)
    gstart_t = T(gstart, torch.int64)
    gcount_t = T(gcount, torch.int64)

    def pairs_of(g0: int, g1: int):
        cnt = gcount_t[g0:g1]
        st = gstart_t[g0:g1]
        e_grp = torch.repeat_interleave(torch.arange(g1 - g0, device=dev), cnt)
        e = st[e_grp] + (torch.arange(len(e_grp), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt))
        m_e = cnt[e_grp]
        A = torch.repeat_interleave(e, m_e)
        j = torch.arange(len(A), device=dev) - torch.repeat_interleave(torch.cumsum(m_e, 0) - m_e, m_e)
        B = st[e_grp][torch.repeat_interleave(torch.arange(len(e), device=dev), m_e)] + j
        return srt_t[A], srt_t[B]

    pair_cache = [pairs_of(g0, g1) for g0, g1 in chunks] if nv and int(np.sum(gcount.astype(np.int64) ** 2)) <= 8 * _PAIR_CHUNK else None

    def eval_cost(Rc, tc, fc, Xc):
        c = torch.einsum("oij,oj->oi", Rc[oi], Xc[op]) + tc[oi]
        cz = c[:, 2]
        bad = cz <= 1e-9
        czs = torch.where(bad, torch.ones_like(cz), cz)
        ex = fc * c[:, 0] / czs + cx - uv[:, 0]
        ey = fc * c[:, 1] / czs + cy - uv[:, 1]
        e2 = ex * ex + ey * ey
        e = torch.sqrt(e2)
        hc = torch.where(e <= huber, e2, 2 * huber * e - huber * huber)
        cost = torch.where(bad, torch.full_like(hc, huber * 1e3), hc).sum()
        sq = torch.where(bad, torch.full_like(e2, 1e6), e2).sum()
        return float(cost), math.sqrt(float(sq) / max(n_obs, 1))

    fcur = float(f)
    cur_cost, cur_rms = eval_cost(Rt, tt, fcur, X)
    rms_before = cur_rms
    lam = 1e-4
    it = 0
    ar = torch.arange(nc, device=dev)
    d3 = torch.arange(3, device=dev)
    dC = torch.arange(nC, device=dev)

    while it < iters:
        # --- Jacobians ---
        Ro = Rt[oi]
        r = torch.einsum("oij,oj->oi", Ro, X[op])
        c = r + tt[oi]
        valid = c[:, 2] > 1e-9
        iz = 1 / torch.where(valid, c[:, 2], torch.ones_like(c[:, 2]))
        ex = fcur * c[:, 0] * iz + cx - uv[:, 0]
        ey = fcur * c[:, 1] * iz + cy - uv[:, 1]
        res = torch.stack([ex, ey], 1)
        e = torch.hypot(ex, ey)
        w = torch.where(e <= huber, torch.ones_like(e), huber / torch.where(e > 0, e, torch.ones_like(e)))
        w = torch.where(valid, w, torch.zeros_like(w))
        P00 = fcur * iz
        P02 = -fcur * c[:, 0] * iz * iz
        P12 = -fcur * c[:, 1] * iz * iz
        rx, ry, rz = r.unbind(1)
        z = torch.zeros_like(P00)
        Jc = torch.stack(
            [
                torch.stack([P02 * ry, P00 * rz - P02 * rx, -P00 * ry, P00, z, P02], 1),
                torch.stack([-P00 * rz + P12 * ry, -P12 * rx, P00 * rx, z, P00, P12], 1),
            ],
            1,
        )
        Jf = torch.stack([c[:, 0] * iz, c[:, 1] * iz], 1)
        Jp = torch.stack([P00[:, None] * Ro[:, 0] + P02[:, None] * Ro[:, 2], P00[:, None] * Ro[:, 1] + P12[:, None] * Ro[:, 2]], 1)

        # --- normal equations ---
        wJp = w[:, None, None] * Jp
        V = torch.zeros((npt, 3, 3), dtype=dt, device=dev).index_add_(0, op, torch.einsum("oai,oaj->oij", wJp, Jp))
        gp = torch.zeros((npt, 3), dtype=dt, device=dev).index_add_(0, op, torch.einsum("oai,oa->oi", wJp, res))
        U = torch.zeros((nC, nC), dtype=dt, device=dev)
        gc = torch.zeros(nC, dtype=dt, device=dev)
        Jcv = Jc[vo]
        wJc = w[vo][:, None, None] * Jcv
        Wc = torch.einsum("oai,oaj->oij", wJc, Jp[vo])  # (nv,6,3)
        if nc:
            Ub = torch.zeros((nc, 6, 6), dtype=dt, device=dev).index_add_(0, vc, torch.einsum("oai,oaj->oij", wJc, Jcv))
            U6 = torch.zeros((nc, 6, nc, 6), dtype=dt, device=dev)
            U6[ar, :, ar, :] = Ub
            U[: 6 * nc, : 6 * nc] = U6.reshape(6 * nc, 6 * nc)
            gc[: 6 * nc] = torch.zeros((nc, 6), dtype=dt, device=dev).index_add_(0, vc, torch.einsum("oai,oa->oi", wJc, res[vo])).reshape(-1)
        if rf:
            wJf = w[:, None] * Jf
            if nc:
                col = torch.zeros((nc, 6), dtype=dt, device=dev).index_add_(0, vc, torch.einsum("oai,oa->oi", wJc, Jf[vo])).reshape(-1)
                U[:fI, fI] += col
                U[fI, :fI] += col
            gc[fI] = (wJf * res).sum()
            U[fI, fI] = (wJf * Jf).sum()
            Wf = torch.einsum("oa,oaj->oj", wJf, Jp)
            Wfj = torch.zeros((npt, 3), dtype=dt, device=dev).index_add_(0, op, Wf)

        # --- LM step (try, adjust λ) ---
        accepted = False
        for _ in range(8):
            S = U.clone()
            S[dC, dC] = U[dC, dC] * (1 + lam) + 1e-9
            bvec = -gc
            Vd = V.clone()
            Vd[:, d3, d3] = V[:, d3, d3] * (1 + lam) + 1e-9
            Vinv, _ok = _inv3(Vd)
            if nc:
                Y = torch.einsum("oij,ojk->oik", Wc, Vinv[vp])
                bvec[: 6 * nc] += torch.zeros((nc, 6), dtype=dt, device=dev).index_add_(0, vc, torch.einsum("oij,oj->oi", Y, gp[vp])).reshape(-1)
                Sb = torch.zeros((nc * nc, 6, 6), dtype=dt, device=dev)
                for k, (g0, g1) in enumerate(chunks):
                    A, B = pair_cache[k] if pair_cache is not None else pairs_of(g0, g1)
                    Sb.index_add_(0, vc[A] * nc + vc[B], torch.einsum("pik,pjk->pij", Y[A], Wc[B]))
                S[: 6 * nc, : 6 * nc] -= Sb.view(nc, nc, 6, 6).permute(0, 2, 1, 3).reshape(6 * nc, 6 * nc)
            if rf:
                Yf = torch.einsum("pj,pjk->pk", Wfj, Vinv)
                if nc:
                    col = torch.zeros((nc, 6), dtype=dt, device=dev).index_add_(0, vc, torch.einsum("oij,oj->oi", Y, Wfj[vp])).reshape(-1)
                    S[:fI, fI] -= col
                    S[fI, :fI] -= col
                S[fI, fI] -= (Yf * Wfj).sum()
                bvec[fI] += (Yf * gp).sum()
            if nC > 0:
                L, info = torch.linalg.cholesky_ex(S)
                if int(info) != 0:
                    lam *= 10
                    continue
                dc = torch.cholesky_solve(bvec[:, None], L)[:, 0]
                if not bool(torch.isfinite(dc).all()):
                    lam *= 10
                    continue
            else:
                dc = torch.zeros(0, dtype=dt, device=dev)
            # points: dp = V⁻¹ (−gp − Wᵀ dc)
            rhs = -gp
            if nc:
                dcc = dc[: 6 * nc].view(nc, 6)
                rhs = rhs.index_add(0, vp, -torch.einsum("oij,oi->oj", Wc, dcc[vc]))
            if rf:
                rhs = rhs - Wfj * dc[fI]
            Xn = X + torch.einsum("pij,pj->pi", Vinv, rhs)
            Rn, tn = Rt, tt
            if nc:
                vct = T(var_cams, torch.int64)
                Rn = Rt.clone()
                tn = tt.clone()
                Rn[vct] = _exp_so3(dcc[:, :3]) @ Rt[vct]
                tn[vct] = tt[vct] + dcc[:, 3:]
            fn = fcur + float(dc[fI]) if rf else fcur
            n_cost, n_rms = eval_cost(Rn, tn, fn, Xn)
            if n_cost < cur_cost:
                rel = (cur_cost - n_cost) / cur_cost
                Rt, tt, fcur, X = Rn, tn, fn, Xn
                cur_cost, cur_rms = n_cost, n_rms
                lam = max(lam / 3, 1e-9)
                accepted = True
                if rel < 1e-6:
                    it = iters
                break
            lam *= 5
        if not accepted:
            break
        it += 1

    if nc:
        R[var_cams] = Rt[T(var_cams, torch.int64)].cpu().numpy()
        t[var_cams] = tt[T(var_cams, torch.int64)].cpu().numpy()
    points[upts] = X.cpu().numpy()
    return {"f": fcur, "rmsBefore": rms_before, "rmsAfter": cur_rms, "iterations": it}
