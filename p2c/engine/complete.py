"""Use read dimensions to complete missing walls.

A dimension runs from wall face to wall face (or centre line).  At each end
(tick) we expect a wall running perpendicular to the dimension line.  If no wall
is drawn there, we search the scan at that spot with relaxed settings: two
parallel ink lines (the wall faces) with white between them, running along the
extension direction.
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from shapely.geometry import Point
from shapely.ops import unary_union

from ..model import Wall
from ..geometry import wall_polygon


def _sampler(ink):
    H, W = ink.shape

    def at(pts):
        x = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
        inside = (pts[..., 0] >= 0) & (pts[..., 0] < W) & (pts[..., 1] >= 0) & (pts[..., 1] < H)
        return ink[y, x] & inside
    return at


def _longest_run(flags, centre, max_gap):
    """Longest run of True (gaps <= max_gap allowed) that contains or touches index `centre`."""
    n = len(flags)
    if n == 0:
        return None
    best = None
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i; gap = 0; last = i
        while j + 1 < n:
            j += 1
            if flags[j]:
                gap = 0; last = j
            else:
                gap += 1
                if gap > max_gap:
                    break
        a, b = i, last
        if a - max_gap <= centre <= b + max_gap and (best is None or b - a > best[1] - best[0]):
            best = (a, b)
        i = last + 1
    return best


def find_wall_at(gat, q, n_dir, out_dir, s, t_min, t_max, max_len, min_len, dark=185):
    """Search a wall running along n_dir whose face or centre passes near q.
    Uses gray levels: both faces dark, the strip between them clearly lighter
    (works for 10 cm partitions whose faces are only ~3 px apart)."""
    A = np.arange(-max_len, max_len + 1, 1.0)
    base = q[None, :] + A[:, None] * n_dir
    prof = {}
    def col(b):
        k = round(b * 2) / 2
        if k not in prof:
            prof[k] = gat(base + k * out_dir)
        return prof[k]
    best = None
    tp_lo, tp_hi = max(2, int(math.floor(t_min))), int(math.ceil(t_max))
    for b0 in np.arange(-tp_hi - 3, 3.5, 1.0):
        f1 = np.minimum.reduce([col(b0 - 0.5), col(b0), col(b0 + 0.5)])
        if f1[len(A) // 2 - 6:len(A) // 2 + 7].min() > dark:
            continue
        for t in range(tp_lo, tp_hi + 1):
            f2 = np.minimum.reduce([col(b0 + t - 0.5), col(b0 + t), col(b0 + t + 0.5)])
            mids = [col(b0 + t * k / 4) for k in (1, 2, 3)] if t >= 4 else [col(b0 + t / 2)]
            mid = np.mean(mids, axis=0)
            ok = (f1 < dark) & (f2 < dark) & (mid > np.maximum(f1, f2) + 30) & (mid > 150)
            # a covered tick position must be part of the run
            run = _longest_run(ok, len(A) // 2, max_gap=4)
            if run is None:
                continue
            L = run[1] - run[0]
            if L < min_len:
                continue
            fill = ok[run[0]:run[1] + 1].mean()
            if fill < 0.6:
                continue
            score = L * fill
            if best is None or score > best[0]:
                best = (score, A[run[0]], A[run[1]], b0 + t / 2, t)
    return None if best is None else best[1:]


def complete_walls(project, gray, params, max_new=200):
    s = project.scale
    if not project.dims or not project.walls:
        return 0
    gb = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    H, W = gb.shape
    def gat(pts):
        return cv2.remap(gb, pts[:, 0].astype(np.float32).reshape(1, -1), pts[:, 1].astype(np.float32).reshape(1, -1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255).ravel()
    t_min = max(2.0, 0.07 / s); t_max = params['wall_t_max'] / s
    max_len = int(params.get('complete_max_len', 8.0) / s)
    min_len = max(10, params.get('complete_min_len', 0.6) / s)
    polys = [wall_polygon(w, s) for w in project.walls]
    U = unary_union(polys)
    TQ = np.array(params['wall_thicknesses'])
    added = 0
    for dm in project.dims:
        p0, p1 = np.array(dm.p0, float), np.array(dm.p1, float)
        L = np.linalg.norm(p1 - p0)
        if L < 3:
            continue
        u = (p1 - p0) / L; n = np.array([-u[1], u[0]])
        for q, out in ((p0, -u), (p1, u)):
            # already explained by a drawn wall?
            if U.distance(Point(q)) <= 0.08 / s:
                continue
            res = find_wall_at(gat, q, n, out, s, t_min, t_max, max_len, min_len)
            if res is None:
                continue
            a0, a1, bc, t = res
            c0 = q + n * a0 + out * bc; c1 = q + n * a1 + out * bc
            tm = float(TQ[np.argmin(abs(TQ - t * s))])
            w = Wall(pts=[tuple(map(float, c0)), tuple(map(float, c1))], t=tm)
            pg = wall_polygon(w, s)
            if pg.intersection(U).area > 0.6 * pg.area:       # mostly duplicates an existing wall
                continue
            project.walls.append(w); polys.append(pg); U = U.union(pg)
            added += 1
            if added >= max_new:
                return added
    return added
