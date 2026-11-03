"""Brute-force matching of binary AKAZE descriptors (486 bits, 61 bytes) with torch.

Port of web/src/sfm/matcher.ts (a WebGPU shader there). For each descriptor of A: the
best and second-best Hamming distance in B, best index = first minimum (strict <, as
the shader's scan). Kept when best <= maxDist, best <= ratio * second and the best of
B->A points back (mutual check). Hamming distances come from a matmul of the unpacked
bits (exact integers in fp16/fp32), on CUDA when available.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

NO_DIST = 0xFFFF  # the shader's initial best/second


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hamming_matrix(a: torch.Tensor, b: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
    """Hamming distances of unpacked {0,1} descriptors: |a| + |b| - 2 a·b."""
    return pa[:, None] + pb[None, :] - 2 * (a @ b.T)


def match_pair(a: torch.Tensor, b: torch.Tensor, pa, pb, ratio: float, max_dist: int) -> tuple[np.ndarray, np.ndarray]:
    na, nb = a.shape[0], b.shape[0]
    if na == 0 or nb == 0:
        e = np.zeros(0, dtype=np.int32)
        return e, e
    D = hamming_matrix(a, b, pa, pb).float().round_()
    if nb >= 2:
        vals = torch.topk(D, 2, dim=1, largest=False, sorted=True).values
        best, second = vals[:, 0], vals[:, 1]
    else:
        best = D[:, 0]
        second = torch.full_like(best, NO_DIST)
    bi = torch.argmin(D, dim=1)  # first minimum
    ai = torch.argmin(D, dim=0)  # B -> A
    i = torch.arange(na, device=D.device)
    best, second = best.double(), second.double()  # JS doubles
    keep = (best <= max_dist) & ~(best > ratio * second) & (ai[bi] == i)
    ia = i[keep].to(torch.int32).cpu().numpy()
    ib = bi[keep].to(torch.int32).cpu().numpy()
    return ia, ib


def match(
    descs: list[np.ndarray],
    pairs: list[tuple[int, int]],
    ratio: float = 0.8,
    max_dist: int = 110,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    """descs: per image (n, 61) uint8. Returns, for every pair in order,
    {"a", "b", "idx": (M,2) int32 [ia, ib]} (PairMatches of matcher.ts)."""
    dev = device or _device()
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    bits: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(i: int):
        if i not in bits:
            d = np.asarray(descs[i], dtype=np.uint8).reshape(len(descs[i]), -1)
            u = torch.from_numpy(np.unpackbits(d, axis=1)).to(dev)
            bits[i] = (u, u.sum(1).to(dt))
        u, p = bits[i]
        return u.to(dt), p

    out = []
    step = max(1, len(pairs) // 100)
    for k, (a, b) in enumerate(pairs):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        ua, pa = get(a)
        ub, pb = get(b)
        ia, ib = match_pair(ua, ub, pa, pb, ratio, max_dist)
        out.append({"a": int(a), "b": int(b), "idx": np.stack([ia, ib], 1).astype(np.int32)})
        if progress is not None and ((k + 1) % step == 0 or k + 1 == len(pairs)):
            progress(k + 1, len(pairs))
    return out
