"""Incremental SfM for orbit video: geometric verification, tracks, initial pair, PnP
registration, triangulation and bundle adjustment. Port of web/src/sfm/reconstruct.ts.

Same constants, seeds, orderings and tie-breaking as the TS; the per-track and
per-observation loops are vectorized with numpy (same results up to float rounding).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..types import Point3D, Pose, SfmModel
from .ba import bundle_adjust
from .features import ImageFeatures
from .geometry import ransac_fundamental, relative_pose, solve_pnp, tri_angle, triangulate_groups, triangulate_two
from .linalg import exp_so3, inv3, sym_eig

MAX_ERR = 4.0  # px
MIN_ANGLE = 1.5  # degrees
MIN_PNP = 20

_log = logging.getLogger("gausify.sfm_akaze")


@dataclass
class _Verified:
    a: int
    b: int
    ia: np.ndarray
    ib: np.ndarray


@dataclass
class _InitPair:
    a: int
    b: int
    R: np.ndarray
    t: np.ndarray
    score: float


def _js_round(x):
    return np.floor(np.asarray(x) + 0.5)


def _check(cancel):
    if cancel is not None and cancel():
        raise InterruptedError("cancelled")


def verify(imgs: list[ImageFeatures], matches: list[dict], cancel=None) -> list[_Verified]:
    """Geometric verification: fundamental RANSAC (2 px, seed a*7919+b), >= 20 inliers."""
    verified = []
    for m in matches:
        _check(cancel)
        idx = np.asarray(m["idx"]).reshape(-1, 2)
        n = len(idx)
        if n < 20:
            continue
        a, b = int(m["a"]), int(m["b"])
        p1 = imgs[a].xy[idx[:, 0]].astype(np.float64)
        p2 = imgs[b].xy[idx[:, 1]].astype(np.float64)
        r = ransac_fundamental(p1, p2, 2.0, a * 7919 + b)
        if r is None or r["count"] < 20:
            continue
        inl = r["inliers"]
        verified.append(_Verified(a, b, idx[inl, 0].astype(np.int64), idx[inl, 1].astype(np.int64)))
    return verified


def build_tracks(imgs: list[ImageFeatures], verified: list[_Verified], closed_loop: bool):
    """Union-find tracks, closest pairs first, refusing merges that would put two
    features of one image in a track. Returns (track_ptr, track_node, feat_track, refused, dropped)."""
    N = len(imgs)
    off = np.concatenate([[0], np.cumsum([im.n for im in imgs])]).astype(np.int64)
    nodes = int(off[-1])
    node_img_np = np.repeat(np.arange(N), [im.n for im in imgs])
    node_img = node_img_np.tolist()
    parent = list(range(nodes))

    def gap_of(v):
        g = v.b - v.a
        return min(g, N - g) if closed_loop else g

    by_trust = sorted(verified, key=lambda v: (gap_of(v), -len(v.ia)))
    members: dict[int, set] = {}
    refused = 0
    for v in by_trust:
        oa, ob = int(off[v.a]), int(off[v.b])
        for ka, kb in zip(v.ia.tolist(), v.ib.tolist()):
            x = oa + ka
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            ra = x
            x = ob + kb
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            rb = x
            if ra == rb:
                continue
            sa = members.get(ra)
            sb = members.get(rb)
            if sa is None:
                sa = {node_img[ra]}
            if sb is None:
                sb = {node_img[rb]}
            if not sa.isdisjoint(sb):
                refused += 1
                continue
            parent[ra] = rb
            members.pop(ra, None)
            members[rb] = sa | sb

    root = np.array(parent, dtype=np.int64)
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    # groups in order of first appearance (Map insertion order), members ascending
    _, first, inv = np.unique(root, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    g = rank[inv]
    size = np.bincount(g, minlength=len(first))
    x = np.nonzero(size[g] >= 2)[0]
    gx = g[x]
    key = gx * N + node_img_np[x]
    _, kinv, kcnt = np.unique(key, return_inverse=True, return_counts=True)
    uniq = kcnt[kinv] == 1
    dropped = int((~uniq).sum())
    x, gx = x[uniq], gx[uniq]
    cnt = np.bincount(gx, minlength=len(first))
    keep = cnt[gx] >= 2
    x, gx = x[keep], gx[keep]
    srt = np.lexsort((x, gx))
    x, gx = x[srt], gx[srt]
    ug, starts, counts = np.unique(gx, return_index=True, return_counts=True)
    track_ptr = np.concatenate([starts, [len(x)]]).astype(np.int64)
    tid = np.repeat(np.arange(len(ug)), counts)
    feat_track = np.full(nodes, -1, dtype=np.int64)
    feat_track[x] = tid
    return track_ptr, x.astype(np.int64), feat_track, refused, dropped


def reconstruct(
    imgs: list[ImageFeatures],
    matches: list[dict],
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    closed_loop: bool = False,
    cancel: Callable[[], bool] | None = None,
) -> SfmModel:
    log = log or _log.info
    N = len(imgs)
    W, H = imgs[0].width, imgs[0].height
    cx, cy = W / 2, H / 2
    f0 = 1.2 * max(W, H)
    off = np.concatenate([[0], np.cumsum([im.n for im in imgs])]).astype(np.int64)
    all_xy = np.concatenate([im.xy.astype(np.float64).reshape(-1, 2) for im in imgs]) if off[-1] else np.zeros((0, 2))
    all_rgb = np.concatenate([np.asarray(im.rgb).reshape(-1, 3) for im in imgs]).astype(np.float64) if off[-1] else np.zeros((0, 3))
    node_img = np.repeat(np.arange(N), [im.n for im in imgs])

    # ---- 1. geometric verification ----
    verified = verify(imgs, matches, cancel)
    log(f"Geometric verification: {len(verified)}/{len(matches)} pairs")

    # ---- 2. tracks ----
    track_ptr, track_node, feat_track, refused, dropped = build_tracks(imgs, verified, closed_loop)
    n_tracks = len(track_ptr) - 1
    pair_inliers = {(v.a, v.b): len(v.ia) for v in verified}
    log(f"Tracks: {n_tracks} ({refused} conflicting merges refused, {dropped} ambiguous observations dropped)")

    def expand(tr: np.ndarray):
        s = track_ptr[tr]
        ln = track_ptr[tr + 1] - s
        rep = np.repeat(np.arange(len(tr)), ln)
        pos = np.arange(int(ln.sum())) - np.repeat(np.cumsum(ln) - ln, ln)
        return rep, track_node[s[rep] + pos]

    # ---- 3. initial pair candidates ----
    candidates: list[_InitPair] = []
    bands: dict[int, list[_Verified]] = {}
    for v in verified:
        gap = min(v.b - v.a, N - (v.b - v.a))
        if gap > N / 4:
            continue
        bands.setdefault(math.floor(2 * math.log2(gap)), []).append(v)
    per_band = math.ceil(60 / max(1, len(bands)))
    pool = [v for b in bands.values() for v in sorted(b, key=lambda x: -len(x.ia))[:per_band]]
    for v in pool:
        _check(cancel)
        p1 = all_xy[off[v.a] + v.ia]
        p2 = all_xy[off[v.b] + v.ib]
        c0 = np.array([cx, cy])
        n1 = (p1 - c0) / f0
        n2 = (p2 - c0) / f0
        F = ransac_fundamental(p1, p2, 2.0, 99)
        if F is None:
            continue
        idx = np.nonzero(F["inliers"])[0]
        (Rr, tr_), good = relative_pose(F["F"], f0, cx, cy, n1, n2, idx)
        X, ok = triangulate_two(Rr, tr_, n1[idx], n2[idx])
        sel = ok & (X[:, 2] > 0)
        angles = np.sort(tri_angle(np.zeros(3), -Rr.T @ tr_, X[sel]))
        med = float(angles[len(angles) >> 1]) if len(angles) else 0.0
        if med >= 2 and good >= 25:
            candidates.append(_InitPair(v.a, v.b, Rr, tr_, good * min(med, 6)))
    candidates.sort(key=lambda c: -c.score)
    tries: list[_InitPair] = []
    for c in candidates:
        if any(c.a in (t.a, t.b) or c.b in (t.a, t.b) for t in tries):
            continue
        tries.append(c)
        if len(tries) == 3:
            break
    if not tries:
        raise RuntimeError("No initial pair with enough parallax.")

    def attempt(init: _InitPair) -> SfmModel:
        st = {"f": f0}
        R = np.zeros((N, 3, 3))
        T = np.zeros((N, 3))
        reg = np.zeros(N, dtype=bool)
        track_point = np.full(n_tracks, -1, dtype=np.int64)
        pts = {"X": np.zeros((0, 3)), "track": np.zeros(0, dtype=np.int64)}
        order: list[int] = []

        def centers():
            return -np.einsum("nji,nj->ni", R, T)

        def reproj(nodes: np.ndarray, X: np.ndarray) -> np.ndarray:
            img = node_img[nodes]
            c = np.einsum("nij,nj->ni", R[img], X) + T[img]
            f = st["f"]
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.hypot(f * c[:, 0] / c[:, 2] + cx - all_xy[nodes, 0], f * c[:, 1] / c[:, 2] + cy - all_xy[nodes, 1])
            e[c[:, 2] <= 0] = np.inf
            return e

        def normed(nodes: np.ndarray) -> np.ndarray:
            return (all_xy[nodes] - np.array([cx, cy])) / st["f"]

        def max_angles(rep: np.ndarray, nodes: np.ndarray, X: np.ndarray, starts: np.ndarray) -> np.ndarray:
            """Per group: largest angle between the first observation's camera and the others."""
            C = centers()
            img = node_img[nodes]
            grp = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(rep))))
            c0 = C[img[starts]][grp]
            ang = tri_angle(c0, C[img], X)
            ang[starts] = 0
            return np.maximum.reduceat(ang, starts)

        def triangulate_tracks(tracks, max_err=MAX_ERR) -> int:
            tr = np.asarray(tracks, dtype=np.int64)
            tr = tr[track_point[tr] < 0]
            if not len(tr):
                return 0
            rep, nodes = expand(tr)
            m = reg[node_img[nodes]]
            rep, nodes = rep[m], nodes[m]
            cnt = np.bincount(rep, minlength=len(tr))
            m = cnt[rep] >= 2
            rep, nodes = rep[m], nodes[m]
            if not len(rep):
                return 0
            ut, starts, counts = np.unique(rep, return_index=True, return_counts=True)
            img = node_img[nodes]
            X, ok = triangulate_groups(R[img], T[img], normed(nodes), starts)
            grp = np.repeat(np.arange(len(ut)), counts)
            Xo = X[grp]
            good = np.add.reduceat((reproj(nodes, Xo) < max_err).astype(np.int64), starts)
            cond = ok & (good >= 2) & ~(good < counts * 0.5)
            cond &= ~(max_angles(rep, nodes, Xo, starts) < MIN_ANGLE)
            new = tr[ut[cond]]
            if not len(new):
                return 0
            P = len(pts["track"])
            track_point[new] = P + np.arange(len(new))
            pts["track"] = np.concatenate([pts["track"], new])
            pts["X"] = np.concatenate([pts["X"], X[cond]])
            return len(new)

        def image_tracks(img: int) -> np.ndarray:
            ft = feat_track[off[img] : off[img + 1]]
            return ft[ft >= 0]

        def run_ba(var_cams: list[int], refine_focal: bool, iters: int, max_err: float) -> dict:
            P = len(pts["track"])
            if len(var_cams) in (len(order) - 1, len(order)):
                sel = np.arange(P)
            else:
                ft = np.concatenate([image_tracks(i) for i in var_cams]) if var_cams else np.zeros(0, np.int64)
                tp = track_point[ft]
                tp = tp[tp >= 0]
                _, first = np.unique(tp, return_index=True)
                sel = tp[np.sort(first)]
            rep, nodes = expand(pts["track"][sel])
            p_o = sel[rep]
            m = reg[node_img[nodes]]
            p_o, nodes = p_o[m], nodes[m]
            m = ~(reproj(nodes, pts["X"][p_o]) > max_err)
            p_o, nodes = p_o[m], nodes[m]
            r = bundle_adjust(
                st["f"], cx, cy, R, T, var_cams, pts["X"], node_img[nodes], p_o, all_xy[nodes],
                refine_focal, 2.0, iters,
            )
            st["f"] = r["f"]
            return r

        def filter_points() -> int:
            P = len(pts["track"])
            if not P:
                return 0
            rep, nodes = expand(pts["track"])
            m = reg[node_img[nodes]]
            rep, nodes = rep[m], nodes[m]
            m = reproj(nodes, pts["X"][rep]) < MAX_ERR
            rep, nodes = rep[m], nodes[m]
            cnt = np.bincount(rep, minlength=P)
            maxang = np.zeros(P)
            m = cnt[rep] >= 2
            rep2, nodes2 = rep[m], nodes[m]
            if len(rep2):
                ut, starts = np.unique(rep2, return_index=True)
                maxang[ut] = max_angles(rep2, nodes2, pts["X"][rep2], starts)
            remove = (cnt < 2) | (maxang < MIN_ANGLE)
            track_point[pts["track"][remove]] = -1
            keep = ~remove
            pts["track"] = pts["track"][keep]
            pts["X"] = pts["X"][keep]
            track_point[pts["track"]] = np.arange(len(pts["track"]))
            return int(remove.sum())

        R[init.a] = np.eye(3)
        T[init.a] = 0
        R[init.b] = init.R
        T[init.b] = init.t
        reg[[init.a, init.b]] = True
        order.extend([init.a, init.b])
        triangulate_tracks(image_tracks(init.a))
        run_ba([init.b], False, 20, 1e9)
        log(f"Initial pair {imgs[init.a].name} + {imgs[init.b].name}: {len(pts['track'])} points")

        # ---- 4. incremental registration ----
        failed: set[int] = set()
        last_global = 2
        while True:
            _check(cancel)
            ftp = np.where(feat_track >= 0, track_point[np.maximum(feat_track, 0)], -1)
            counts = np.bincount(node_img[ftp >= 0], minlength=N).astype(np.int64)
            counts[reg] = -1
            if failed:
                counts[list(failed)] = -1
            nxt = int(np.argmax(counts)) if N else -1
            best_count = int(counts[nxt]) if N else 0
            if best_count <= 0 or best_count < MIN_PNP:
                break
            nodes = np.arange(off[nxt], off[nxt + 1])
            nodes = nodes[ftp[nodes] >= 0]
            Xc = pts["X"][ftp[nodes]]
            xc = normed(nodes)
            neighbor = None
            for d in range(1, N):
                for j in ((nxt + d) % N, (nxt - d + N) % N):
                    if reg[j]:
                        neighbor = (R[j].copy(), T[j].copy())
                        break
                if neighbor is not None:
                    break
            r = solve_pnp(Xc, xc, MAX_ERR / st["f"], neighbor, nxt + 1)
            if r is None or len(r[1]) < MIN_PNP:
                failed.add(nxt)
                log(f"PnP failed on {imgs[nxt].name} ({0 if r is None else len(r[1])}/{len(nodes)} inliers)")
                continue
            R[nxt], T[nxt] = r[0]
            reg[nxt] = True
            order.append(nxt)
            triangulate_tracks(image_tracks(nxt))

            local = [i for i in order[-8:] if i != order[0]]
            run_ba(local, False, 6, MAX_ERR * 2)
            if len(order) >= last_global * 1.25 and len(order) >= 8:
                g = run_ba(order[1:], True, 150 if len(order) < 20 else 15, MAX_ERR * 2)
                filter_points()
                last_global = len(order)
                log(f"Global BA with {len(order)} cameras: {g['rmsBefore']:.2f} -> {g['rmsAfter']:.2f} px, f = {st['f']:.1f}")
            if progress is not None:
                progress(len(order) / N, f"{len(order)}/{N} cameras registered")
            if failed and len(order) % 5 == 0:
                failed.clear()

        # ---- 5. close the orbit ----
        def close_orbit():
            ring = [i for i in range(N) if reg[i]]
            L = len(ring)
            if L < max(12, 0.8 * N):
                return
            C = centers()[ring]
            m = C.mean(0)
            D = C - m
            values, V = sym_eig(D.T @ D)
            if values[0] > 0.2 * values[1]:
                return
            axis = V[:, 0].copy()
            e1 = V[:, 2].copy()
            e2 = np.cross(axis, e1)
            P = np.stack([D @ e1, D @ e2], 1)
            rows = np.stack([2 * P[:, 0], 2 * P[:, 1], np.ones(L)], 1)
            A = rows.T @ rows
            rhs = rows.T @ (P[:, 0] ** 2 + P[:, 1] ** 2)
            Ai = inv3(A)
            if Ai is None:
                return
            sol = Ai @ rhs
            o = m + sol[0] * e1 + sol[1] * e2
            theta = np.arctan2(P[:, 1] - sol[1], P[:, 0] - sol[0])
            d = np.roll(theta, -1) - theta
            steps = np.arctan2(np.sin(d), np.cos(d))
            srt = np.sort(steps)
            med = srt[L >> 1]
            dr = float(np.sign(med)) if np.isfinite(med) and med != 0 else 1.0
            steps = steps * dr
            total = float(steps.sum())
            if abs(total - 2 * math.pi) > 0.2:
                return
            seam = int(np.argmax(steps))
            around = np.sort([steps[(seam + k + L) % L] for k in range(-4, 5) if k])
            expected = max(0.0, (around[3] + around[4]) / 2)
            excess = steps[seam] - expected
            if excess < 2 * math.pi / 180 or steps[seam] < 3 * max(expected, med * dr):
                return

            def inl(i, j):
                return pair_inliers.get((min(i, j), max(i, j)), 0)

            consecutive = sorted(inl(ring[k], ring[(k + 1) % L]) for k in range(L))
            if inl(ring[seam], ring[(seam + 1) % L]) < 0.5 * consecutive[L >> 1]:
                return
            scale = (total - expected) / (total - steps[seam])
            delta = np.zeros(L)
            cum = 0.0
            for j in range(L):
                k = (seam + 1 + j) % L
                delta[k] = (scale - 1) * cum
                cum += steps[k]
            ref = delta[ring.index(order[0])]
            for k, i in enumerate(ring):
                Q = exp_so3(axis * dr * (delta[k] - ref))
                c_new = o + Q @ (C[k] - o)
                R[i] = R[i] @ Q.T
                T[i] = -(R[i] @ c_new)
            track_point.fill(-1)
            pts["X"] = np.zeros((0, 3))
            pts["track"] = np.zeros(0, dtype=np.int64)
            triangulate_tracks(np.arange(n_tracks), 24)
            log(
                f"Orbit closed: {math.degrees(steps[seam]):.1f} deg jump between {imgs[ring[seam]].name} and "
                f"{imgs[ring[(seam + 1) % L]].name} spread out (steps x{scale:.3f}), {len(pts['track'])} points"
            )

        if closed_loop:
            close_orbit()

        # ---- 6. wrap-up: recover observations with a decreasing threshold ----
        for thr in (24, 12, 6, MAX_ERR):
            _check(cancel)
            triangulate_tracks(np.arange(n_tracks))
            g = run_ba(order[1:], True, 25, thr)
            filter_points()
            log(f"Final BA (threshold {thr} px): {g['rmsBefore']:.2f} -> {g['rmsAfter']:.2f} px, f = {st['f']:.1f}")

        # ---- output ----
        points: list[Point3D] = []
        err_sum, n_obs = 0.0, 0
        P = len(pts["track"])
        if P:
            rep, nodes = expand(pts["track"])
            m = reg[node_img[nodes]]
            rep, nodes = rep[m], nodes[m]
            e = reproj(nodes, pts["X"][rep])
            m = e < MAX_ERR
            rep, nodes, e = rep[m], nodes[m], e[m]
            cnt = np.bincount(rep, minlength=P)
            m = cnt[rep] >= 2
            rep, nodes, e = rep[m], nodes[m], e[m]
            cnt = np.bincount(rep, minlength=P)
            rgb = np.stack([np.bincount(rep, all_rgb[nodes, c], minlength=P) for c in range(3)], 1)
            esum = np.bincount(rep, e, minlength=P)
            ok = cnt >= 2
            rgb_m = _js_round(rgb[ok] / cnt[ok, None]).astype(int)
            imgs_o = node_img[nodes]
            kps = nodes - off[imgs_o]
            bounds = np.concatenate([[0], np.cumsum(cnt)])
            for q, p in enumerate(np.nonzero(ok)[0]):
                s, t_ = bounds[p], bounds[p + 1]
                obs = list(zip(imgs_o[s:t_].tolist(), kps[s:t_].tolist()))
                points.append(Point3D(X=pts["X"][p].copy(), rgb=tuple(rgb_m[q].tolist()), obs=obs, error=float(esum[p] / cnt[p])))
            err_sum = float(esum[ok].sum())
            n_obs = int(cnt[ok].sum())
        registered = int(reg.sum())
        f = st["f"]
        return SfmModel(
            f=f, cx=cx, cy=cy, width=W, height=H,
            poses=[Pose(R[i].copy(), T[i].copy()) if reg[i] else None for i in range(N)],
            points=points,
            names=[im.name for im in imgs],
            stats={
                "registered": registered, "total": N, "points": len(points), "observations": n_obs,
                "meanError": err_sum / max(n_obs, 1), "focal": f,
            },
        )

    best: SfmModel | None = None
    for init in tries:
        if best is not None:
            log(f"Only {best.stats['registered']}/{N} cameras: trying another initial pair")
        m = attempt(init)
        if best is None or m.stats["registered"] > best.stats["registered"]:
            best = m
        if best.stats["registered"] >= 0.9 * N:
            break
    return best
