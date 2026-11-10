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


def _classify(cur_m, target_m):
    """Why a dimension disagrees with the drawing."""
    if cur_m <= 0:
        return 'geometry'
    r = target_m / cur_m
    if abs(math.log10(r)) > 0.8:
        return 'unit'
    a, b = f"{target_m:.2f}".replace('.', ''), f"{cur_m:.2f}".replace('.', '')
    if len(a) == len(b) and (sorted(a) == sorted(b) or sum(x != y for x, y in zip(a, b)) == 1):
        return 'ocr'                       # swapped or single wrong digit
    return 'inconsistent'


def fit_to_dimensions(project, max_shift=0.15, max_resid=0.10, lam=0.05, report=None, huber=0.03,
                      far=0.50):
    """Robust, confidence-weighted least squares.  Dimensions far from the drawing (> `far` m) are not
    used and are reported with a reason; the rest are solved with Huber weights (IRLS) so one wrong
    reading cannot pull the drawing; residual outliers are rejected one at a time.  No wall moves more
    than `max_shift` m."""
    s = project.scale
    walls = project.walls
    if not project.dims or not walls:
        return dict(used=0)
    polys = [wall_polygon(w, s) for w in walls]
    cons = []                                      # (ia, ib, sa, sb, ra, rb, target_px, cur_px, dim_index, sign, weight)
    rejected = []
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
            rejected.append((k, 'no_wall_at_end')); continue
        refs = []
        for i, q in ((ia, p0), (ib, p1)):
            a, b, e, m = _wall_frame(walls[i])
            sgn = 1.0 if m @ u >= 0 else -1.0
            c = (q - a) @ m
            t = walls[i].t / s
            ref = min((0.0, -t / 2, t / 2), key=lambda r: abs(c - r))
            refs.append((sgn, ref, (a @ m)))
        (sa, ra, ca), (sb, rb, cb) = refs
        cur = (sb * (cb + rb)) - (sa * (ca + ra))
        target = dm.value / s
        if cur * target < 0 or abs(abs(cur) - target) * s > far:
            rejected.append((k, _classify(abs(cur) * s, dm.value))); continue
        wgt = float(getattr(dm, 'conf', 1.0) or 1.0)
        cons.append([ia, ib, sa, sb, ra, rb, abs(target), abs(cur), k, np.sign(cur), wgt])
    if not cons:
        _mark(project, rejected, [])
        project.fit_info = dict(used=0, rejected=len(rejected))
        return project.fit_info
    idx = sorted({c[0] for c in cons} | {c[1] for c in cons})
    col = {i: j for j, i in enumerate(idx)}
    active = [True] * len(cons)
    d = np.zeros(len(idx))
    lim = max_shift / s

    def resid(dvec, c):
        ia, ib, sa, sb, ra, rb, tgt, cur, k, sg, wgt = c
        return (cur + (sb * dvec[col[ib]] - sa * dvec[col[ia]]) * sg) - tgt
    for _outer in range(len(cons)):
        rob = np.ones(len(cons))
        for _it in range(8):                        # IRLS with Huber weights
            rows, rhs, wts = [], [], []
            for n_, (c, on) in enumerate(zip(cons, active)):
                if not on:
                    continue
                ia, ib, sa, sb, ra, rb, tgt, cur, k, sg, wgt = c
                r = np.zeros(len(idx))
                r[col[ib]] += sb * sg; r[col[ia]] -= sa * sg
                rows.append(r); rhs.append(tgt - cur); wts.append(wgt * rob[n_])
            if not rows:
                break
            A = np.array(rows); bvec = np.array(rhs); W = np.array(wts)
            AtA = A.T @ (A * W[:, None]) + lam * np.eye(len(idx)); Atb = A.T @ (bvec * W)
            d = np.clip(np.linalg.solve(AtA, Atb), -lim, lim)
            for n_, c in enumerate(cons):
                e = abs(resid(d, c)) * s
                rob[n_] = 1.0 if e <= huber else huber / e
        res = [abs(resid(d, c)) * s if on else -1 for c, on in zip(cons, active)]
        worst = int(np.argmax(res))
        if res[worst] > max_resid:
            active[worst] = False
            continue
        break
    before, after = [], []
    for c, on in zip(cons, active):
        if not on:
            ia, ib, sa, sb, ra, rb, tgt, cur, k, sg, wgt = c
            rejected.append((k, 'residual ' + _classify(cur * s, tgt * s)))
            continue
        ia, ib, sa, sb, ra, rb, tgt, cur, k, sg, wgt = c
        before.append(abs(cur - tgt) * s); after.append(abs(resid(d, c)) * s)
    shifts = {i: d[col[i]] for i in idx if abs(d[col[i]]) > 0.05}
    log = getattr(project, 'corrections', None)
    if log is not None:
        for i, dd in shifts.items():
            log.append(dict(action='fit_shift', wall=int(i), shift_m=round(float(dd * s), 4),
                            reason='match written dimensions'))
    _apply(project, polys, shifts)
    for i in shifts:
        if getattr(walls[i], 'origin', '') in ('', 'auto'):
            walls[i].origin = 'fixed' if walls[i].origin == 'auto' else walls[i].origin
    _mark(project, rejected, [c[8] for c, on in zip(cons, active) if on])
    used = sum(active)
    info = dict(used=used, rejected=len(rejected), walls_moved=len(shifts),
                err_before=float(np.mean(before)) if before else 0.0,
                err_after=float(np.mean(after)) if after else 0.0,
                max_before=float(np.max(before)) if before else 0.0,
                max_after=float(np.max(after)) if after else 0.0,
                reasons={r: sum(1 for _, x in rejected if x == r) for r in {x for _, x in rejected}})
    project.fit_info = info
    return info


def _mark(project, rejected, used):
    for k in used:
        project.dims[k].status = 'used'
    for k, why in rejected:
        if not str(getattr(project.dims[k], 'status', '')).startswith('rejected'):
            project.dims[k].status = 'rejected:' + why


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
