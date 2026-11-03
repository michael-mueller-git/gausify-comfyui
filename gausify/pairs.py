"""Pair selection and orbit timestamps.

Port of `orbitPairs` / `orbitTimes` in gausify-orbit web/src/sfm/index.ts.
"""

from __future__ import annotations

import math
import re

_JS_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def orbit_pairs(n: int, window: int, closed: bool) -> list[tuple[int, int]]:
    """Same pair selection as the app (colmap.rs: write_pairs)."""
    offsets = [k + 1 for k in range(window)]
    for o in (16, 32, 64):
        if o > window:
            offsets.append(o)
    limit = n // 2 if closed else n - 1
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i in range(n):
        for o in offsets:
            if o > limit:
                continue
            j = i + o
            if j >= n and not closed:
                continue
            a, b = min(i, j % n), max(i, j % n)
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b))
    return out


def _js_number(s: str) -> float:
    """`Number(s)` of JavaScript for the decimal strings frame names carry."""
    s = s.strip()
    if s == "":
        return 0.0
    if _JS_NUMBER.match(s):
        return float(s)
    if re.match(r"^0[xX][0-9a-fA-F]+$", s):
        return float(int(s, 16))
    return math.nan


def orbit_times(names: list[str], times: dict[str, float] | None = None) -> list[float]:
    """Timestamp of each image (in name order); if missing or not increasing
    (frames from loose images), the number in the name (00012.jpg -> 12),
    which preserves the gaps left by rejected frames; else the index."""
    t = [times.get(n) if times else None for n in names]
    if all(v is not None and (i == 0 or v > t[i - 1]) for i, v in enumerate(t)):
        return [float(v) for v in t]  # type: ignore[arg-type]
    idx = [_js_number(re.sub(r"\.[^.]+$", "", n, count=1)) for n in names]
    if all(math.isfinite(v) and (i == 0 or v > idx[i - 1]) for i, v in enumerate(idx)):
        return idx
    return [float(i) for i in range(len(names))]
