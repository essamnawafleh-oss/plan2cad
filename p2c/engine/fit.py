"""Adjust wall positions so the drawing matches the written dimensions.

Each read dimension says: (face or centre of wall B) - (face or centre of wall A) = value.
Unknowns: a shift d_i of every involved wall along its own normal.
We solve   min  sum_k w_k (residual_k)^2  +  lam * sum_i d_i^2
robustly (dimensions that disagree strongly with the drawing are treated as misreads and dropped),
then move the walls, and drag everything attached to them (wall ends, doors, windows, dimension ends).
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Point

from ..geometry import wall_polygon


def _wall_frame(w):
    a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
    e = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    return a, b, e, np.array([-e[1], e[0]])


def _end_wall(walls, polys, q, u, s, tol, reach=0.0):
    """Wall at a dimension end: roughly perpendicular to the dimension, close to the tick.
    With reach > 0 the tick is also followed along its extension line (outer dimension chains)."""
    best = None
    P = Point(q)
    if reach > 0:
        from shapely.geometry import LineString
        n = np.array([-u[1], u[0]])
        ray = LineString([tuple(q - n * reach), tuple(q + n * reach)])
        for i, (w, pg) in enumerate(zip(walls, polys)):
            if len(w.pts) != 2:
                continue
            a, b, e, m = _wall_frame(w)
            if abs(e @ u) > 0.2 or not ray.intersects(pg):
                continue
            c = (np.asarray(q) - a) @ m; t = w.t / s
            off = min(abs(c), abs(c - t / 2), abs(c + t / 2))
            if off > tol:
                continue
            dist = pg.distance(P)
            if best is None or dist < best[0]:
                best = (dist, i)
        return None if best is None else best[1]
    for i, (w, pg) in enumerate(zip(walls, polys)):
        if len(w.pts) != 2:
            continue
        a, b, e, m = _wall_frame(w)
        if abs(e @ u) > 0.35:                      # wall must cross the dimension direction
            continue
        d = pg.distance(P)
        if d <= tol and (best is None or d < best[0]):
            best = (d, i)
    return None if best is None else best[1]


def fit_to_dimensions(project, max_shift=0.15, max_resid=0.20, lam=0.05, report=None):
    s = project.scale
    walls = project.walls
    if not project.dims or not walls:
        return dict(used=0)
    polys = [wall_polygon(w, s) for w in walls]
    cons = []                                      # (ia, ib, sa, sb, ra, rb, target_px, cur_px, dim_index)
    for k, dm in enumerate(project.dims):
        p0, p1 = np.array(dm.p0, float), np.array(dm.p1, float)
        L = np.linalg.norm(p1 - p0)
        if L < 5:
            continue
        u = (p1 - p0) / L
        ia = _end_wall(walls, polys, p0, u, s, 0.20 / s)
        ib = _end_wall(walls, polys, p1, u, s, 0.20 / s)
        if ia is None:
            ia = _end_wall(walls, polys, p0, u, s, 0.06 / s, reach=12.0 / s)
        if ib is None:
            ib = _end_wall(walls, polys, p1, u, s, 0.06 / s, reach=12.0 / s)
        if ia is None or ib is None or ia == ib:
            continue
        refs = []
        for i, q in ((ia, p0), (ib, p1)):
            a, b, e, m = _wall_frame(walls[i])
            sgn = 1.0 if m @ u >= 0 else -1.0       # orientation of the wall normal relative to u
            c = (q - a) @ m                         # tick offset from the centre line (along m)
            t = walls[i].t / s
            ref = min((0.0, -t / 2, t / 2), key=lambda r: abs(c - r))   # tick on centre or on a face
            refs.append((sgn, ref, (a @ m)))
        (sa, ra, ca), (sb, rb, cb) = refs
        # position along u of the reference line = sgn * (centre_offset + ref) ; centre offset along m = a.m
        cur = (sb * (cb + rb)) - (sa * (ca + ra))
        target = dm.value / s
        if cur * target < 0 or abs(abs(cur) - target) * s > 0.20:
            continue
        cons.append([ia, ib, sa, sb, ra, rb, abs(target), abs(cur), k, np.sign(cur)])
    if not cons:
        return dict(used=0)
    idx = sorted({c[0] for c in cons} | {c[1] for c in cons})
    col = {i: j for j, i in enumerate(idx)}
    active = [True] * len(cons)
    d = np.zeros(len(idx))
    for _ in range(6):
        rows, rhs, wts = [], [], []
        for c, on in zip(cons, active):
            if not on:
                continue
            ia, ib, sa, sb, ra, rb, tgt, cur, k, sg = c
            r = np.zeros(len(idx))
            r[col[ib]] += sb * sg; r[col[ia]] -= sa * sg      # shifting along m moves the line by sgn*d
            rows.append(r); rhs.append(tgt - cur); wts.append(1.0)
        if not rows:
            break
        A = np.array(rows); bvec = np.array(rhs)
        lam_px = lam
        AtA = A.T @ A + lam_px * np.eye(len(idx)); Atb = A.T @ bvec
        d = np.linalg.solve(AtA, Atb)
        lim = max_shift / s
        d = np.clip(d, -lim, lim)
        # drop worst constraint if its residual is too large (misread or mismatched dimension)
        res = []
        for n_, (c, on) in enumerate(zip(cons, active)):
            ia, ib, sa, sb, ra, rb, tgt, cur, k, sg = c
            new = cur + (sb * d[col[ib]] - sa * d[col[ia]]) * sg
            res.append(abs(new - tgt) * s if on else -1)
        worst = int(np.argmax(res))
        if res[worst] > max_resid:
            active[worst] = False
            continue
        break
    # stats
    before, after = [], []
    for c, on in zip(cons, active):
        if not on:
            continue
        ia, ib, sa, sb, ra, rb, tgt, cur, k, sg = c
        new = cur + (sb * d[col[ib]] - sa * d[col[ia]]) * sg
        before.append(abs(cur - tgt) * s); after.append(abs(new - tgt) * s)
    # apply shifts (+ drag attached geometry)
    shifts = {i: d[col[i]] for i in idx if abs(d[col[i]]) > 0.05}
    _apply(project, polys, shifts)
    used = sum(active)
    info = dict(used=used, rejected=len(cons) - used, walls_moved=len(shifts),
                err_before=float(np.mean(before)) if before else 0.0,
                err_after=float(np.mean(after)) if after else 0.0,
                max_before=float(np.max(before)) if before else 0.0,
                max_after=float(np.max(after)) if after else 0.0)
    project.fit_info = info
    return info


def _apply(project, polys, shifts):
    s = project.scale
    walls = project.walls
    old = {i: (_wall_frame(walls[i]), polys[i].buffer(1.5)) for i in shifts}
    moves = []                                   # (zone polygon, normal m, shift)
    for i, dd in shifts.items():
        (a, b, e, m), zone = old[i]
        walls[i].pts = [tuple(map(float, np.array(q) + m * dd)) for q in walls[i].pts]
        moves.append((i, zone, e, m, dd))

    def drag(pt, own_dir=None, skip=None):
        P = Point(pt)
        q = np.array(pt, float)
        for i, zone, e, m, dd in moves:
            if i == skip or not zone.contains(P):
                continue
            if own_dir is not None and abs(own_dir @ m) > 0.2:
                q = q + own_dir * (dd / (own_dir @ m))       # slide along own direction to stay on the moved wall
            else:
                q = q + m * dd
        return tuple(map(float, q))

    for j, w in enumerate(walls):
        if j in shifts or len(w.pts) != 2:
            continue
        a, b, e, m = _wall_frame(w)
        w.pts = [drag(w.pts[0], e, j), drag(w.pts[1], e, j)]
    for o in list(project.doors) + list(project.windows):
        o.p0, o.p1 = drag(o.p0), drag(o.p1)
    for dm in project.dims:
        p0, p1 = np.array(dm.p0), np.array(dm.p1)
        u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        dm.p0, dm.p1 = drag(dm.p0, u), drag(dm.p1, u)


def dimension_report(project):
    """For every read dimension: written value vs. distance in the current drawing (or None)."""
    s = project.scale
    polys = [wall_polygon(w, s) for w in project.walls]
    out = []
    for dm in project.dims:
        p0, p1 = np.array(dm.p0, float), np.array(dm.p1, float)
        L = np.linalg.norm(p1 - p0)
        if L < 5:
            continue
        u = (p1 - p0) / L
        ends = []
        for q in (p0, p1):
            i = _end_wall(project.walls, polys, q, u, s, 0.20 / s)
            if i is None:
                i = _end_wall(project.walls, polys, q, u, s, 0.06 / s, reach=12.0 / s)
            ends.append(i)
        drawn = None
        if None not in ends and ends[0] != ends[1]:
            vals = []
            for i, q in zip(ends, (p0, p1)):
                a, b, e, m = _wall_frame(project.walls[i]); t = project.walls[i].t / s
                c = (q - a) @ m
                ref = min((0.0, -t / 2, t / 2), key=lambda r: abs(c - r))
                vals.append(((a @ m) + ref) * (1 if m @ u >= 0 else -1))
            drawn = abs(vals[1] - vals[0]) * s
        out.append(dict(text=dm.text, value=dm.value, drawn=drawn,
                        diff=None if drawn is None else drawn - dm.value))
    return out
