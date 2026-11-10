"""Geometric validation before export.

``validate(project)`` returns a list of issues; each issue is a dict:
    severity  'critical' | 'major' | 'minor' | 'info'
    code      short machine name ('zero_length', 'duplicate_wall', ...)
    kind/index  the object concerned (or kind None for drawing-wide issues)
    at        (x, y) image px, for zooming to it
    msg_ar / msg_en
    fix       name of an automatic fix, or None

``apply_fix(project, issue)`` performs one fix and returns a log entry (callers push an undo snapshot
first, so every fix can be undone).  ``auto_fix(project)`` only applies fixes that cannot destroy
information (invalid / zero-length entities, exact duplicates) and logs them in project.corrections.
Nothing uncertain is changed silently: it is reported for review.
"""
from __future__ import annotations

import math
import numpy as np

SEV_ORDER = {'critical': 0, 'major': 1, 'minor': 2, 'info': 3}


def _issue(sev, code, kind, index, at, ar, en, fix=None, **extra):
    d = dict(severity=sev, code=code, kind=kind, index=index,
             at=[float(at[0]), float(at[1])] if at is not None else None, msg_ar=ar, msg_en=en, fix=fix)
    d.update(extra)
    return d


def _finite(*vals):
    return all(math.isfinite(float(v)) for v in vals)


def _seg(w):
    a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L if L > 1e-9 else np.array([1.0, 0.0])
    return a, b, u, L


def _dist_pt_seg(p, a, b):
    ab = b - a; L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip((p - a) @ ab / L2, 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _dist_pts_segs(P, A, B):
    """Distances from points P (m,2) to segments A->B (n,2): (m, n) array."""
    P = np.asarray(P, float).reshape(-1, 2)
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - Q, axis=2)


def validate(project):
    s = project.scale
    out = []
    if not (s and math.isfinite(s) and s > 0):
        out.append(_issue('critical', 'bad_scale', None, None, None, 'المقياس غير صالح', 'Invalid drawing scale'))
        return out
    walls = project.walls
    good = []
    # --- per wall
    for i, w in enumerate(walls):
        pts = list(w.pts)
        if len(pts) < 2 or not all(_finite(*p) for p in pts) or not _finite(w.t):
            out.append(_issue('critical', 'invalid_coords', 'walls', i, pts[0] if pts and _finite(*pts[0]) else None,
                              'جدار بإحداثيات غير صالحة', 'Wall with invalid coordinates', 'delete'))
            continue
        a, b, u, L = _seg(w)
        if L * s < 0.01:
            out.append(_issue('major', 'zero_length', 'walls', i, a, 'جدار طوله صفر', 'Zero-length wall', 'delete'))
            continue
        if not 0.05 <= w.t <= 0.60:
            out.append(_issue('major', 'bad_thickness', 'walls', i, (a + b) / 2,
                              f'سماكة جدار غير منطقية ({w.t:.2f} م)', f'Implausible wall thickness ({w.t:.2f} m)'))
        good.append(i)
    segs = {i: _seg(walls[i]) for i in good}
    gi = np.array(good, int)
    A = np.array([segs[i][0] for i in good]).reshape(-1, 2); B = np.array([segs[i][1] for i in good]).reshape(-1, 2)
    T = np.array([walls[i].t / s for i in good])
    ENDS = np.vstack([A, B]) if len(good) else np.zeros((0, 2))
    DE = _dist_pts_segs(ENDS, A, B) if len(good) else np.zeros((0, 0))       # (2n, n)
    # --- connectivity of every wall end
    for k, i in enumerate(good):
        a, b, u, L = segs[i]
        t = walls[i].t / s
        free = []
        for end, p in ((0, a), (1, b)):
            row = DE[k + end * len(good)].copy()
            row[k] = np.inf
            if not np.any(row <= T / 2 + t / 2 + 0.03 / s):
                free.append((end, p))
        if L * s < 0.15 and len(free) == 2:
            out.append(_issue('minor', 'tiny_wall', 'walls', i, (a + b) / 2, 'قطعة جدار قصيرة جداً ومعزولة',
                              'Very short isolated wall piece', 'delete'))
        elif len(free) == 2 and L * s < 1.0:
            out.append(_issue('minor', 'isolated_wall', 'walls', i, (a + b) / 2, 'جدار معزول لا يتصل بأي جدار',
                              'Isolated wall (connected to nothing)'))
        for end, p in free:
            dirn = u if end == 1 else -u
            best = None
            for j in good:
                if j == i:
                    continue
                c, d, v, Lj = segs[j]
                # distance along the wall direction to the other wall's centre line
                den = dirn[0] * v[1] - dirn[1] * v[0]
                if abs(den) < 0.3:
                    continue
                M = np.array([dirn, -v]).T
                tt, ss = np.linalg.solve(M, c - p)
                if 0 < tt * s <= 0.35 + walls[j].t / 2 and -0.05 <= ss <= Lj + 0.05 / s:
                    face = tt - walls[j].t / s / 2 - t / 2          # visible gap: cap end to the other face
                    if face > 0.005 / s and (best is None or tt < best[0]):
                        best = (tt, face)
            if best is not None and best[1] * s <= 0.35:
                out.append(_issue('minor', 'open_gap', 'walls', i, p,
                                  f'فراغ {best[1] * s * 100:.0f} سم بين طرف الجدار وجدار آخر',
                                  f'{best[1] * s * 100:.0f} cm gap between a wall end and another wall', 'join',
                                  end=end, gap_m=round(best[1] * s, 3), ext_m=round(best[0] * s, 4)))
    # --- duplicates / overlaps (collinear walls)
    lo = np.minimum(A, B) - T[:, None]; hi = np.maximum(A, B) + T[:, None]
    for ii, i in enumerate(good):
        a, b, u, L = segs[i]
        near = np.nonzero((lo[ii + 1:, 0] <= hi[ii, 0]) & (hi[ii + 1:, 0] >= lo[ii, 0]) &
                          (lo[ii + 1:, 1] <= hi[ii, 1]) & (hi[ii + 1:, 1] >= lo[ii, 1]))[0] + ii + 1
        for jj in near:
            j = good[jj]
            c, d, v, Lj = segs[j]
            if abs(abs(u @ v) - 1) > 1 - math.cos(math.radians(2)):
                continue
            n = np.array([-u[1], u[0]])
            off = abs((c - a) @ n)
            if off > 0.5 * max(walls[i].t, walls[j].t) / s:
                continue
            p0, p1 = sorted([0.0, L]); q = sorted([(c - a) @ u, (d - a) @ u])
            ov = min(p1, q[1]) - max(p0, q[0])
            if ov <= 0.10 / s:
                continue
            short = min(L, Lj)
            if ov >= 0.8 * short:
                out.append(_issue('major', 'duplicate_wall', 'walls', j if Lj <= L else i, (c + d) / 2,
                                  'جدار مكرّر فوق جدار آخر', 'Duplicate wall on top of another', 'merge',
                                  other=i if Lj <= L else j))
            else:
                out.append(_issue('minor', 'overlap_wall', 'walls', j, (c + d) / 2,
                                  f'جداران متراكبان {ov * s:.2f} م', f'Walls overlap by {ov * s:.2f} m', 'merge', other=i))
    # --- wall outline validity
    try:
        from ..geometry import walls_union
        U = walls_union(project, cut_windows=False)
        if U is not None and not U.is_valid:
            out.append(_issue('major', 'invalid_polygon', None, None, None, 'شكل الجدران المجمّع غير صالح (تقاطع ذاتي)',
                              'Combined wall outline is invalid (self-intersection)', 'repair_union'))
    except Exception as e:                                  # pragma: no cover
        out.append(_issue('major', 'union_failed', None, None, None, f'تعذّر دمج الجدران: {e}', f'Wall union failed: {e}'))
    # --- openings on walls
    for kind, objs, ar, en, (lo, hi) in (('doors', project.doors, 'باب', 'Door', (0.5, 2.6)),
                                         ('windows', project.windows, 'شباك', 'Window', (0.3, 6.0))):
        for k, o in enumerate(objs):
            if not (_finite(*o.p0) and _finite(*o.p1)):
                out.append(_issue('critical', 'invalid_coords', kind, k, None, f'{ar} بإحداثيات غير صالحة',
                                  f'{en} with invalid coordinates', 'delete')); continue
            p0, p1 = np.array(o.p0, float), np.array(o.p1, float)
            wdt = float(np.linalg.norm(p1 - p0)) * s
            c = (p0 + p1) / 2
            if wdt < 0.05:
                out.append(_issue('major', 'zero_length', kind, k, c, f'{ar} عرضه صفر', f'Zero-width {en.lower()}', 'delete'))
                continue
            # an opening sits in a wall (window) or between two wall ends (door): test centre and both jambs
            dmin = float((_dist_pts_segs(np.array([c, p0, p1]), A, B).min(0) - T / 2).min()) if len(good) else 1e9
            if dmin * s > 0.30:
                out.append(_issue('major', 'off_wall', kind, k, c, f'{ar} ليس على جدار', f'{en} not on a wall'))
            if not lo <= wdt <= hi:
                out.append(_issue('minor', 'odd_width', kind, k, c, f'عرض {ar} غير معتاد ({wdt:.2f} م)',
                                  f'Unusual {en.lower()} width ({wdt:.2f} m)'))
    # --- confidence
    for kind in ('walls', 'doors', 'windows', 'columns'):
        for k, o in enumerate(getattr(project, kind)):
            if getattr(o, 'origin', '') == 'uncertain' or getattr(o, 'conf', 1.0) < 0.5:
                at = None
                if hasattr(o, 'pts') and o.pts:
                    at = np.mean(np.array(o.pts, float), axis=0)
                elif hasattr(o, 'p0'):
                    at = (np.array(o.p0) + np.array(o.p1)) / 2
                elif hasattr(o, 'x'):
                    at = (o.x, o.y)
                out.append(_issue('info', 'uncertain', kind, k, at, f'عنصر غير مؤكد (ثقة {getattr(o, "conf", 0):.0%}) — راجعه',
                                  f'Uncertain element (confidence {getattr(o, "conf", 0):.0%}) - please review'))
    for k, dm in enumerate(project.dims):
        st = str(getattr(dm, 'status', ''))
        if st.startswith('rejected'):
            out.append(_issue('info', 'dim_rejected', 'dims', k, (np.array(dm.p0) + np.array(dm.p1)) / 2,
                              f'البعد {dm.text} لم يُستعمل: {st[9:]}', f'Dimension {dm.text} not used: {st[9:]}'))
    out.sort(key=lambda d: SEV_ORDER[d['severity']])
    return out


def summary(issues):
    c = {k: 0 for k in SEV_ORDER}
    for d in issues:
        c[d['severity']] += 1
    return c


# ------------------------------------------------------------------ fixes
def apply_fix(project, issue):
    """Apply the fix of one issue. Returns a log entry, or None when not applicable."""
    fx, kind, i = issue.get('fix'), issue.get('kind'), issue.get('index')
    s = project.scale
    if fx == 'delete' and kind and i is not None and i < len(getattr(project, kind)):
        o = getattr(project, kind).pop(i)
        return dict(action='delete', kind=kind, code=issue['code'], what=_brief(o))
    if fx == 'merge' and kind == 'walls':
        j = issue.get('other')
        if j is None or max(i, j) >= len(project.walls):
            return None
        wi, wj = project.walls[i], project.walls[j]
        a, b, u, L = _seg(wj)
        pts = [np.array(p, float) for p in (wi.pts[0], wi.pts[-1], wj.pts[0], wj.pts[-1])]
        ts = [(p - a) @ u for p in pts]
        n = np.array([-u[1], u[0]])
        off = np.mean([(p - a) @ n for p in pts])
        lo, hi = min(ts), max(ts)
        wj.pts = [tuple(map(float, a + u * lo + n * off)), tuple(map(float, a + u * hi + n * off))]
        wj.t = max(wi.t, wj.t)
        project.walls.pop(i)
        return dict(action='merge_walls', code=issue['code'], kept=j if j < i else j - 1)
    if fx == 'join' and kind == 'walls':
        w = project.walls[i]
        a, b, u, L = _seg(w)
        g = issue.get('ext_m', issue.get('gap_m', 0)) / s       # extend to the other wall's centre line
        end = issue.get('end', 1)
        if end == 1:
            w.pts = [tuple(w.pts[0]), tuple(map(float, b + u * g))]
        else:
            w.pts = [tuple(map(float, a - u * g)), tuple(w.pts[-1])]
        w.origin = 'fixed'
        return dict(action='join', code=issue['code'], gap_m=issue.get('gap_m'))
    if fx == 'repair_union':
        return dict(action='repair_union', note='outline repaired at export (buffer 0)')
    return None


def _brief(o):
    d = {}
    for k in ('pts', 'p0', 'p1', 't', 'x', 'y'):
        if hasattr(o, k):
            v = getattr(o, k)
            d[k] = [list(map(float, p)) for p in v] if k == 'pts' else v
    return d


SAFE = {'invalid_coords', 'zero_length', 'duplicate_wall'}


def auto_fix(project, log=None):
    """Apply only information-preserving fixes.  Each round validates once and applies every safe fix of
    that round from the highest index down (so earlier indices stay valid); repeats until none is left."""
    done = []
    for _ in range(20):
        iss = [d for d in validate(project) if d['code'] in SAFE and d.get('fix') and d.get('kind')]
        if not iss:
            break
        touched = set()
        iss.sort(key=lambda d: -(d['index'] if d['index'] is not None else -1))
        for d in iss:
            key = (d['kind'], d['index']); other = (d['kind'], d.get('other'))
            if key in touched or other in touched:
                continue                         # an index this round already changed: next round
            e = apply_fix(project, d)
            if e is not None:
                done.append(e); touched.add(key); touched.add(other)
        if not touched:
            break
    if log is not None:
        for e in done:
            log.append(dict(e, auto=True))
    return done
