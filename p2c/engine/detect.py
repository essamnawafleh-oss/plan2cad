"""Automatic recognition of a scanned floor plan.

Pipeline (all coordinates in working-image px):
    binarize -> skeleton trace (raw lines) -> dominant angles -> wall pairs ->
    wall cleanup (merge / snap / extend to junctions) -> columns -> door openings
    -> windows -> furniture shapes
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..model import Wall, Column, Door, Window, Furn


# =========================================================== binarize / trace
def binarize(gray, thr=185, blur=3):
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
    return (g < thr).astype(np.uint8) * 255


def _thin(bw):
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(bw, thinningType=cv2.ximgproc.THINNING_GUOHALL) > 0
    try:
        from skimage.morphology import skeletonize
        return skeletonize(bw > 0)
    except Exception:
        pass
    # fallback: morphological skeleton
    skel = np.zeros_like(bw); img = bw.copy()
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        er = cv2.erode(img, el)
        skel |= cv2.subtract(img, cv2.dilate(er, el))
        img = er
    return skel > 0


def trace_lines(gray, thr=185, up=2, min_len=3.0, eps=0.9, progress=None):
    """Vectorise the ink into centre-line polylines (orig px).
    Thresholding is done on the up-sampled image so 1-px gaps between close
    parallel lines (thin walls) survive."""
    H0, W0 = gray.shape
    gbig = cv2.resize(gray, (W0*up, H0*up), interpolation=cv2.INTER_CUBIC)
    big = (gbig < thr).astype(np.uint8) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(big, 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 10*up*up
    big = (keep[lab] * 255).astype(np.uint8)
    sk = _thin(big)
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))
    NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def nbrs(p):
        y, x = p
        return [(y+dy, x+dx) for dy, dx in NB if (y+dy, x+dx) in pix]

    nodes = {p for p in pix if len(nbrs(p)) != 2}
    visited = set(); paths = []

    def walk(a, b):
        path = [a, b]; visited.add(frozenset((a, b))); prev, cur = a, b
        while cur not in nodes:
            nx = [q for q in nbrs(cur) if q != prev and frozenset((cur, q)) not in visited]
            if not nx:
                break
            visited.add(frozenset((cur, nx[0]))); prev, cur = cur, nx[0]; path.append(cur)
        return path

    total = max(1, len(nodes))
    for i, a in enumerate(nodes):
        for b in nbrs(a):
            if frozenset((a, b)) not in visited:
                paths.append(walk(a, b))
        if progress and i % 2000 == 0:
            progress(i / total)
    for p in pix:
        for b in nbrs(p):
            if frozenset((p, b)) not in visited:
                nodes.add(p); paths.append(walk(p, b)); nodes.discard(p)
    out = []
    for path in paths:
        if len(path) < 2:
            continue
        arr = np.array([[x, y] for y, x in path], np.float32).reshape(-1, 1, 2)
        ap = cv2.approxPolyDP(arr, eps*up, False).reshape(-1, 2) / up
        if np.sum(np.linalg.norm(np.diff(ap, axis=0), axis=1)) < min_len:
            continue
        out.append(ap.astype(float))
    return out


# =========================================================== dominant angles
def dominant_angles(polys, min_seg=8, max_n=4):
    """Main drawing directions in image space, deg in [0,180)."""
    h = np.zeros(360)
    for p in polys:
        d = np.diff(p, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        a = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180)
        for ai, li in zip(a, L):
            if li >= min_seg:
                h[int(round(ai*2)) % 360] += li
    k = np.ones(9)
    hsm = np.convolve(np.r_[h[-4:], h, h[:4]], k, 'valid')     # +-2 deg smoothing (circular)
    res, hs = [], hsm.copy()
    total = h.sum() + 1e-9
    for _ in range(max_n * 2):
        i = int(np.argmax(hs))
        if hs[i] < 0.06 * total:
            break
        idx = [(i + j) % 360 for j in range(-4, 5)]
        w = h[idx] + 1e-9
        ang = (np.sum((i + np.arange(-4, 5)) * w) / w.sum() / 2) % 180
        for j in range(-10, 11):
            hs[(i + j) % 360] = 0
        res.append(float(ang))
    # keep one per perpendicular family, families sorted by strength
    fam = []
    for a in res:
        if all(min(abs(a - b) % 90, 90 - abs(a - b) % 90) > 3 for b in fam):
            fam.append(a)
    # snap near-axis families exactly to 0
    fam = [0.0 if min(a % 90, 90 - a % 90) < 0.6 else a for a in fam]
    return fam[:max_n] or [0.0]


def _families(angles):
    out = []
    for a in angles:
        out += [a % 180, (a + 90) % 180]
    return sorted(set(round(x, 3) for x in out))


# =========================================================== walls
def _line_darkness(gray, a, b):
    L = np.linalg.norm(b - a)
    n = max(3, int(L))
    xs = np.clip(np.linspace(a[0], b[0], n).round().astype(int), 0, gray.shape[1]-1)
    ys = np.clip(np.linspace(a[1], b[1], n).round().astype(int), 0, gray.shape[0]-1)
    return float(np.median(gray[ys, xs]))


def detect_walls(polys, gray, scale, angles, p):
    """Find walls as pairs of parallel lines.  p = params dict."""
    gmin = cv2.erode(gray, np.ones((3, 3), np.uint8))        # darkest value in 3x3
    fams = _families(angles)
    segs = []
    for pl in polys:
        for i in range(len(pl) - 1):
            segs.append((pl[i], pl[i+1]))
    t_min = p['wall_t_min'] / scale; t_max = p['wall_t_max'] / scale
    min_face = max(6, p['wall_min_len'] / scale * 0.5)
    walls = []
    for A in fams:
        th = math.radians(A); e = np.array([math.cos(th), math.sin(th)]); n = np.array([-e[1], e[0]])
        lines = []
        for a, b in segs:
            d = b - a; L = np.hypot(*d)
            if L < 4:
                continue
            ang = math.degrees(math.atan2(d[1], d[0])) % 180
            if min(abs(ang - A), 180 - abs(ang - A)) > 3.5:
                continue
            u0, u1 = sorted([a @ e, b @ e])
            lines.append([((a + b) / 2) @ n, u0, u1, _line_darkness(gmin, a, b), L])
        lines.sort()
        merged = []
        for v, u0, u1, dk, L in lines:
            for m in merged:
                if abs(m[0] - v) < 1.2 and u0 <= m[2] + 2 and u1 >= m[1] - 2:
                    w0, w1 = m[2] - m[1], u1 - u0
                    m[0] = (m[0]*w0 + v*w1) / (w0 + w1 + 1e-9)
                    m[3] = (m[3]*w0 + dk*w1) / (w0 + w1 + 1e-9)
                    m[1] = min(m[1], u0); m[2] = max(m[2], u1); break
            else:
                merged.append([v, u0, u1, dk])
        merged = [m for m in merged if m[2] - m[1] >= min_face and m[3] <= p['wall_dark']]
        merged.sort()
        for i, (v, u0, u1, dk) in enumerate(merged):
            for j in range(i + 1, len(merged)):
                v2, a0, a1, _ = merged[j]
                dv = v2 - v
                if dv < t_min:
                    continue
                if dv > t_max:
                    break
                o0, o1 = max(u0, a0), min(u1, a1)
                if o1 - o0 < min_face:
                    continue
                # interior of a real wall is (mostly) white
                mid0 = e*o0 + n*(v + dv/2); mid1 = e*o1 + n*(v + dv/2)
                if _line_darkness(gray, mid0, mid1) < 120:
                    continue
                walls.append([A, v + dv/2, o0, o1, dv])
                break
    if p.get('_raw'):
        return walls
    return _clean_walls(walls, scale, p, gray)


def _clean_walls(walls, scale, p, gray=None):
    # merge collinear pieces
    def mergew(ws):
        ws.sort(key=lambda w: (w[0], w[1]))
        out = []
        for w in ws:
            for m in out:
                if m[0] == w[0] and abs(m[1] - w[1]) < 1.8 and abs(m[4] - w[4]) < 2.5 and w[2] <= m[3] + 3 and w[3] >= m[2] - 3:
                    l0, l1 = m[3] - m[2], w[3] - w[2]
                    m[1] = (m[1]*l0 + w[1]*l1) / (l0 + l1); m[4] = max(m[4], w[4])
                    m[2] = min(m[2], w[2]); m[3] = max(m[3], w[3]); break
            else:
                out.append(list(w))
        return out
    for _ in range(3):
        walls = mergew(walls)
    # bridge gaps along a wall where both faces still carry ink (windows, columns, hatching)
    if gray is not None:
        ink = cv2.dilate((cv2.GaussianBlur(gray, (3, 3), 0) < 175).astype(np.uint8), np.ones((3, 3), np.uint8))
        Hh, Ww = ink.shape
        gmax = p.get('wall_bridge', 3.0) / scale
        def face_ink(A, v, t, u0, u1):
            th = math.radians(A); e = np.array([math.cos(th), math.sin(th)]); n = np.array([-e[1], e[0]])
            us = np.arange(u0, u1, 1.0)
            if len(us) == 0:
                return 1.0
            hits = 0
            for uu in us:
                ok = 0
                for o in (-t/2, t/2):
                    x, y = (e*uu + n*(v + o)).round().astype(int)
                    ok += 1 if (0 <= x < Ww and 0 <= y < Hh and ink[y, x]) else 0
                hits += ok == 2
            return hits / len(us)
        changed = True
        while changed:
            changed = False
            walls.sort(key=lambda w: (w[0], w[1], w[2]))
            for i in range(len(walls)):
                a = walls[i]
                if a is None:
                    continue
                for j in range(len(walls)):
                    b = walls[j]
                    if b is None or i == j or a[0] != b[0] or abs(a[1] - b[1]) > 2.0 or abs(a[4] - b[4]) > 3:
                        continue
                    gap = b[2] - a[3]
                    if 0 < gap <= gmax and face_ink(a[0], (a[1] + b[1]) / 2, max(a[4], b[4]), a[3], b[2]) >= 0.7:
                        la, lb = a[3] - a[2], b[3] - b[2]
                        a[1] = (a[1]*la + b[1]*lb) / (la + lb); a[3] = max(a[3], b[3]); a[4] = max(a[4], b[4])
                        walls[j] = None; changed = True
            walls = [w for w in walls if w is not None]
    minlen = p['wall_min_len'] / scale
    W = [dict(A=A, v=v, u0=u0, u1=u1, t=t) for A, v, u0, u1, t in walls if u1 - u0 >= minlen]
    # quantise thickness
    TQ = np.array(p['wall_thicknesses']) / scale
    for w in W:
        w['t'] = float(TQ[np.argmin(abs(TQ - w['t']))])
    # align collinear
    for i, a in enumerate(W):
        for b in W[i+1:]:
            if a['A'] == b['A'] and abs(a['v'] - b['v']) < 2.0 and abs(a['t'] - b['t']) <= 2:
                la, lb = a['u1'] - a['u0'], b['u1'] - b['u0']
                v = (a['v']*la + b['v']*lb) / (la + lb); a['v'] = b['v'] = v
                a['t'] = b['t'] = max(a['t'], b['t'])

    def frame(A):
        t = math.radians(A); e = np.array([math.cos(t), math.sin(t)]); return e, np.array([-e[1], e[0]])

    def line_int(a, b):
        ea, na = frame(a['A']); eb, nb = frame(b['A'])
        M = np.array([na, nb])
        if abs(np.linalg.det(M)) < 0.2:
            return None
        return np.linalg.solve(M, [a['v'], b['v']])

    EXT = p['wall_join'] / scale
    for _ in range(2):
        for a in W:
            ea, _ = frame(a['A'])
            for end in ('u0', 'u1'):
                best = None
                for b in W:
                    if b is a or b['A'] == a['A']:
                        continue
                    q = line_int(a, b)
                    if q is None:
                        continue
                    eb, _ = frame(b['A'])
                    ub = q @ eb
                    if ub < b['u0'] - b['t']/2 - 3 or ub > b['u1'] + b['t']/2 + 3:
                        continue
                    ua = q @ ea
                    dist = (ua - a[end]) if end == 'u1' else (a[end] - ua)
                    if -b['t']/2 - 4 <= dist <= EXT + b['t']/2:
                        if best is None or abs(dist) < abs(best[0]):
                            best = (dist, ua, b['t'])
                if best:
                    _, ua, tb = best
                    a[end] = ua + (tb/2 if end == 'u1' else -tb/2) * 0  # meet at centre line
        for a in W:
            for b in W:
                if a is not b and a['A'] == b['A'] and abs(a['v'] - b['v']) < 0.5 and 0 < b['u0'] - a['u1'] < 5:
                    a['u1'] = b['u0']
    out = []
    for w in W:
        e, n = frame(w['A'])
        p0 = e*w['u0'] + n*w['v']; p1 = e*w['u1'] + n*w['v']
        out.append(Wall(pts=[tuple(map(float, p0)), tuple(map(float, p1))], t=round(w['t']*scale, 3)))
    return out


# =========================================================== columns
def detect_columns(gray, scale, p):
    dark = (gray < p['column_dark']).astype(np.uint8) * 255
    k = max(3, int(round(0.13 / scale)))
    core = cv2.erode(dark, np.ones((k, k), np.uint8))
    blobs = cv2.dilate(core, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cols = []
    smin, smax = 0.18 / scale, 0.9 / scale
    for c in cnts:
        (x, y), (w, h), a = cv2.minAreaRect(c)
        if min(w, h) < smin * 0.8 or max(w, h) > smax or max(w, h) > 2.2 * min(w, h):
            continue
        cols.append(Column(x=float(x), y=float(y), s=round(float(max(w, h)) * scale, 2), a=float(a)))
    return cols


# =========================================================== doors
def detect_doors(walls, gray, scale, p):
    from ..geometry import wall_polygon
    dark = (cv2.GaussianBlur(gray, (3, 3), 0) < 175).astype(np.float32)
    dark = cv2.dilate(dark, np.ones((3, 3), np.uint8))
    H, Wd = dark.shape

    def dk(pts):
        v = [dark[int(round(y)), int(round(x))] for x, y in pts if 0 <= int(round(x)) < Wd and 0 <= int(round(y)) < H]
        return float(np.mean(v)) if v else 0.0

    polys = [wall_polygon(w, scale) for w in walls]
    if not polys:
        return []
    U = unary_union(polys)
    wmin, wmax = p['door_min'] / scale, p['door_max'] / scale
    openings = []
    for k, w in enumerate(walls):
        a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
        if np.linalg.norm(b - a) < 1:
            continue
        d = (b - a) / np.linalg.norm(b - a); tp = w.t / scale
        others = None
        for end, dirv in ((b, d), (a, -d)):
            p0 = end + dirv * (tp/2 + 0.5)
            if U.contains(Point(p0)):
                continue
            if others is None:
                others = unary_union([pp for j, pp in enumerate(polys) if j != k])
            ray = LineString([p0, p0 + dirv * (wmax + 5)])
            hit = ray.intersection(others)
            if hit.is_empty:
                continue
            dist = min(Point(p0).distance(Point(c)) for g in getattr(hit, 'geoms', [hit]) for c in g.coords)
            if dist < wmin or dist > wmax:
                continue
            openings.append((end + dirv * tp/2, p0 + dirv * dist, np.array([-dirv[1], dirv[0]])))
    # dedupe
    ded = []
    for o in openings:
        m = (o[0] + o[1]) / 2
        if all(np.linalg.norm(m - (e[0] + e[1]) / 2) > 0.25 / scale for e in ded):
            ded.append(o)
    doors = []
    for q0, q1, n in ded:
        w = np.linalg.norm(q1 - q0); u = (q1 - q0) / w
        best = (0, None)
        for side in (1, -1):
            nn = n * side
            for hinge, other in ((q0, q1), (q1, q0)):
                uu = (other - hinge) / w
                a0 = math.atan2(uu[1], uu[0]); a1 = math.atan2(nn[1], nn[0])
                da = (a1 - a0 + math.pi) % (2*math.pi) - math.pi
                arc = [hinge + w*0.97*np.array([math.cos(t), math.sin(t)]) for t in np.linspace(a0 + da*.15, a0 + da*.85, 24)]
                sc = 0.75*dk(arc) + 0.25*dk([hinge + nn*w*s for s in np.linspace(.15, .9, 10)])
                if sc > best[0]:
                    best = (sc, ('s', hinge, other, nn))
            sc = 0
            for hinge, uu in ((q0, u), (q1, -u)):
                a0 = math.atan2(uu[1], uu[0]); a1 = math.atan2(nn[1], nn[0])
                da = (a1 - a0 + math.pi) % (2*math.pi) - math.pi
                sc += 0.5*dk([hinge + w/2*0.97*np.array([math.cos(t), math.sin(t)]) for t in np.linspace(a0 + da*.15, a0 + da*.85, 16)])
            if sc > best[0] + (0.1 if w * scale < 1.3 else -0.05):
                best = (sc, ('d', q0, q1, nn))
        if best[0] < p['door_score'] or best[1] is None:
            continue
        kind, h, o, nn = best[1]
        uu = (o - h) / np.linalg.norm(o - h)
        side = 1 if np.dot(np.array([-uu[1], uu[0]]), nn) > 0 else -1
        doors.append(Door(p0=tuple(map(float, h)), p1=tuple(map(float, o)), side=side, double=(kind == 'd')))
    return doors


# =========================================================== windows
def detect_windows(walls, columns, gray, scale, p):
    """A window shows as extra glazing lines *inside* a wall's thickness."""
    dark = cv2.GaussianBlur(gray, (3, 3), 0) < 165
    H, Wd = dark.shape
    cols = np.array([(c.x, c.y) for c in columns]) if columns else np.zeros((0, 2))
    out = []
    for w in walls:
        if w.t < p['window_min_wall']:
            continue
        a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
        L = np.linalg.norm(b - a)
        if L < 1:
            continue
        u = (b - a) / L; n = np.array([-u[1], u[0]]); tp = w.t / scale
        offs = np.linspace(-tp/2 + 1.5, tp/2 - 1.5, max(5, int(tp)))
        flags = []
        for s in np.arange(0, L, 1.0):
            q = a + u * s
            if s < tp or s > L - tp or (len(cols) and np.min(np.hypot(*(cols - q).T)) < 0.3/scale):
                flags.append(False); continue
            prof = []
            for o in offs:
                x, y = (q + n*o).round().astype(int)
                prof.append(bool(0 <= x < Wd and 0 <= y < H and dark[y, x]))
            # count separate dark runs strictly inside the wall
            runs = sum(1 for i in range(len(prof)) if prof[i] and (i == 0 or not prof[i-1]))
            frac = sum(prof) / len(prof)
            flags.append(runs >= 1 and frac <= 0.6)      # glazing lines with white between (not a filled wall)
        f = np.convolve(np.array(flags, float), np.ones(5), 'same') >= 3
        s = 0; minrun = p['window_min_len'] / scale
        while s < len(f):
            if f[s]:
                e2 = s
                while e2 < len(f) and f[e2]:
                    e2 += 1
                if e2 - s >= minrun:
                    out.append(Window(p0=tuple(map(float, a + u*s)), p1=tuple(map(float, a + u*(e2-1))), t=w.t))
                s = e2
            else:
                s += 1
    return out


# =========================================================== furniture
def _guess_block(lo, hi, circ, fill):
    """Pick a block type from the size of a shape (metres)."""
    if circ > 0.80 and hi / max(lo, 1e-6) < 1.15:
        return 'TABLE_ROUND' if hi >= 0.6 else 'CHAIR_OFFICE'
    if 0.8 <= lo <= 1.25 and 1.7 <= hi <= 2.25:
        return 'BED_SINGLE'
    if 1.25 < lo <= 2.1 and 1.8 <= hi <= 2.35:
        return 'BED_DOUBLE'
    if 0.65 <= lo <= 1.1 and 1.35 <= hi <= 2.7 and fill < 0.93:
        return 'SOFA_3' if hi >= 1.9 else 'SOFA_2'
    if 0.38 <= lo and hi <= 0.8 and hi / lo < 1.35:
        return 'CHAIR_ARM' if fill < 0.93 else 'TABLE_RECT'
    if 1.2 <= hi <= 2.2 and 0.6 <= lo <= 0.95:
        return 'DESK'
    return 'TABLE_RECT'


def detect_furniture(gray, scale, mask, p):
    """Ink islands left after removing walls/doors/windows -> clean blocks guessed by size."""
    ink = (cv2.GaussianBlur(gray, (3, 3), 0) < 175).astype(np.uint8) * 255
    ink[mask > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((2, 2), np.uint8)), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if max(w, h) * scale < 0.35 or max(w, h) * scale > 3.6:
            continue
        comp = (lab[y:y+h, x:x+w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        # text / level labels: several tiny letter holes
        cc, hh = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hh is not None:
            tiny = sum(1 for k, q in enumerate(cc) if hh[0][k][3] >= 0 and cv2.contourArea(q) < (0.12 / scale) ** 2)
            if tiny >= 3:
                continue
        (rx, ry), (rw, rh), ang = cv2.minAreaRect(c)
        if min(rw, rh) * scale < 0.3:
            continue
        hull_a = cv2.contourArea(cv2.convexHull(c))
        density = area / max(rw * rh, 1)
        if density > 0.55 or hull_a / max(rw * rh, 1) < 0.6:
            continue                                    # text / hatching / irregular scribble
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * cv2.contourArea(c) / max(per * per, 1e-6)
        fill = cv2.contourArea(c) / max(rw * rh, 1)
        lo, hi = sorted((rw * scale, rh * scale))
        blk = _guess_block(lo, hi, circ, fill)
        long_ang = ang if rw >= rh else ang + 90         # direction of the long side (image deg)
        if blk in ('BED_SINGLE', 'BED_DOUBLE'):
            w_, d_, face = lo, hi, long_ang
        else:
            w_, d_, face = hi, lo, long_ang + 90
        if blk in ('TABLE_ROUND', 'CHAIR_OFFICE'):
            w_ = d_ = (lo + hi) / 2
        from ..blocks import LIB
        out.append(Furn(blk, float(x + rx), float(y + ry), round(w_, 3), round(d_, 3), float(face) % 360,
                        layer=LIB[blk][0]))
    return out


def prune_isolated_walls(walls, scale, min_free_len=1.6):
    """Drop short wall pieces that touch no other wall (usually furniture edges)."""
    from ..geometry import wall_polygon
    polys = [wall_polygon(w, scale) for w in walls]
    keep = []
    for i, (w, pg) in enumerate(zip(walls, polys)):
        L = sum(math.dist(a, b) for a, b in zip(w.pts[:-1], w.pts[1:])) * scale
        if L >= min_free_len:
            keep.append(w); continue
        g = pg.buffer(1.5)
        if any(j != i and g.intersects(q) for j, q in enumerate(polys)):
            keep.append(w)
    return keep


# =========================================================== solid (filled / hatched) walls
def solid_wall_mask(gray, scale, p):
    """Thick dark strokes (filled walls), with thin lines removed."""
    dark = (cv2.GaussianBlur(gray, (3, 3), 0) < p.get('solid_dark', 125)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k = max(3, int(round(p.get('solid_t_min', 0.13) / scale)))
    core = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return core, dark


def _snap_dir(a, b, angles, tol=5.0):
    d = b - a; L = np.linalg.norm(d)
    if L < 1e-6:
        return a, b
    ang = math.degrees(math.atan2(d[1], d[0])) % 180
    best, bd = None, tol
    for A in _families(angles):
        diff = min(abs(ang - A), 180 - abs(ang - A))
        if diff < bd:
            best, bd = A, diff
    if best is None:
        return a, b
    m = (a + b) / 2
    e = np.array([math.cos(math.radians(best)), math.sin(math.radians(best))])
    if np.dot(e, d) < 0:
        e = -e
    return m - e * L / 2, m + e * L / 2


def detect_solid_walls(gray, scale, angles, p):
    core, dark = solid_wall_mask(gray, scale, p)
    if core.sum() == 0:
        return [], core
    dt = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    pseudo = np.where(core > 0, 0, 255).astype(np.uint8)
    polys = trace_lines(pseudo, thr=128, up=2, min_len=3, eps=1.2)
    TQ = np.array(p['wall_thicknesses'])
    out = []
    minlen = p['wall_min_len'] / scale
    for pl in polys:
        for a, b in zip(pl[:-1], pl[1:]):
            L = np.linalg.norm(b - a)
            n = max(3, int(L))
            xs = np.clip(np.linspace(a[0], b[0], n).round().astype(int), 0, gray.shape[1] - 1)
            ys = np.clip(np.linspace(a[1], b[1], n).round().astype(int), 0, gray.shape[0] - 1)
            t = 2 * float(np.median(dt[ys, xs])) + 0.5
            if L < max(minlen, 2.5 * t):
                continue
            a2, b2 = _snap_dir(np.array(a, float), np.array(b, float), angles)
            tm = float(TQ[np.argmin(abs(TQ - t * scale))])
            out.append(Wall(pts=[tuple(map(float, a2)), tuple(map(float, b2))], t=tm))
    return _merge_wall_list(out, scale, gray), core


def _merge_wall_list(walls, scale, gray=None, bridge=4.0):
    """Join collinear straight walls that overlap / touch."""
    items = []
    for w in walls:
        a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
        d = b - a; L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        A = math.degrees(math.atan2(d[1], d[0])) % 180
        e = np.array([math.cos(math.radians(A)), math.sin(math.radians(A))]); n = np.array([-e[1], e[0]])
        u0, u1 = sorted([a @ e, b @ e])
        items.append([A, (a @ n + b @ n) / 2, u0, u1, w.t])
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            if items[i] is None:
                continue
            for j in range(len(items)):
                if i == j or items[j] is None:
                    continue
                A, v, u0, u1, t = items[i]; B, v2, a0, a1, t2 = items[j]
                if min(abs(A - B), 180 - abs(A - B)) > 2 or abs(v - v2) > max(t, t2) / scale * 0.6:
                    continue
                ok = a0 <= u1 + 0.3 / scale and a1 >= u0 - 0.3 / scale
                if not ok and gray is not None:
                    gap0, gap1 = (u1, a0) if a0 > u1 else (a1, u0)
                    if 0 < gap1 - gap0 <= bridge / scale:
                        e = np.array([math.cos(math.radians(A)), math.sin(math.radians(A))]); nn = np.array([-e[1], e[0]])
                        vv = (v + v2) / 2
                        ok = _line_darkness(gray, e * gap0 + nn * vv, e * gap1 + nn * vv) < 205 and \
                             _line_darkness(cv2.erode(gray, np.ones((5, 5), np.uint8)), e * gap0 + nn * vv, e * gap1 + nn * vv) < 140
                if ok:
                    la, lb = u1 - u0, a1 - a0
                    items[i] = [A, (v * la + v2 * lb) / (la + lb), min(u0, a0), max(u1, a1), max(t, t2)]
                    items[j] = None; changed = True
    out = []
    for it in items:
        if it is None:
            continue
        A, v, u0, u1, t = it
        e = np.array([math.cos(math.radians(A)), math.sin(math.radians(A))]); n = np.array([-e[1], e[0]])
        out.append(Wall(pts=[tuple(map(float, e * u0 + n * v)), tuple(map(float, e * u1 + n * v))], t=t))
    return out


def join_walls(walls, scale, max_gap=0.6):
    """Extend free wall ends to the nearest wall face in their own direction (closes corners and T's)."""
    from ..geometry import wall_polygon
    from shapely.geometry import LineString as LS
    polys = [wall_polygon(w, scale) for w in walls]
    gmax = max_gap / scale
    for i, w in enumerate(walls):
        if len(w.pts) != 2:
            continue
        for end in (0, 1):
            a = np.array(w.pts[1 - end], float); b = np.array(w.pts[end], float)
            d = b - a; L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            d /= L
            tp = w.t / scale
            tip = b + d * (tp / 2 + 0.6)
            others = [q for j, q in enumerate(polys) if j != i and q is not None]
            if any(q.contains(Point(tip)) for q in others):
                continue
            ray = LS([b, b + d * (gmax + tp)])
            best = None
            for j, q in enumerate(polys):
                if j == i or q is None:
                    continue
                h = ray.intersection(q)
                if h.is_empty:
                    continue
                for g in getattr(h, 'geoms', [h]):
                    for c in g.coords:
                        dd = np.linalg.norm(np.array(c) - b)
                        if best is None or dd < best[0]:
                            best = (dd, j)
            if best is None:
                continue
            j = best[1]
            # go to the far side's centre line of the hit wall: penetrate by half its thickness
            new = b + d * (best[0] + walls[j].t / scale / 2)
            pts = list(w.pts); pts[end] = tuple(map(float, new)); w.pts = pts
            polys[i] = wall_polygon(w, scale)
    return walls


def keep_building(walls, scale, link=1.0, min_frac=0.06):
    """Keep only wall clusters that form the building; drops dimension chains, page edges, title blocks."""
    from ..geometry import wall_polygon
    from shapely.ops import unary_union as _uu
    if not walls:
        return walls
    polys = [wall_polygon(w, scale) for w in walls]
    U = _uu([p.buffer(link / scale / 2) for p in polys])
    comps = list(getattr(U, 'geoms', [U]))
    lens = []
    for c in comps:
        L = sum(math.dist(w.pts[0], w.pts[-1]) for w, p in zip(walls, polys) if c.intersects(p))
        lens.append(L)
    big = max(lens)
    cnt = [sum(1 for p in polys if c.intersects(p)) for c in comps]
    keep_c = [c for c, L, n in zip(comps, lens, cnt) if L >= min_frac * big and n >= 3]
    out = []
    for w, p in zip(walls, polys):
        if any(c.intersects(p) for c in keep_c):
            # a long thin isolated straight piece with nothing attached is a dimension line
            out.append(w)
    return out



# =========================================================== robust wall faces from rotated raster
def _rot_img(gray, ang):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s_ + w * c) + 2, int(h * c + w * s_) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255), M


def raster_wall_items(gray, scale, angles, p):
    """Find wall faces as thin horizontal strokes in images rotated to each drawing direction,
    then pair parallel faces with white between them.  Returns raw items [A, v, u0, u1, t]."""
    t_min = p['wall_t_min'] / scale; t_max = p['wall_t_max'] / scale
    min_face = max(8, p.get('raster_min_len', 0.8) / scale)
    items = []
    for A in _families(angles):
        img, M = _rot_img(gray, A)
        Mi = cv2.invertAffineTransform(M)
        ink = (img < p.get('face_thr', 175)).astype(np.uint8)
        hc = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
        # keep only horizontal structure: open with a long horizontal kernel
        horiz = cv2.morphologyEx(hc, cv2.MORPH_OPEN, np.ones((1, int(min_face)), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(horiz, 8)
        faces = []
        th_ = math.radians(A); e_ = np.array([math.cos(th_), math.sin(th_)]); nn_ = np.array([-e_[1], e_[0]])
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if w < min_face:
                continue
            comp = lab[y:y + h, x:x + w] == i
            prof = comp.mean(axis=1)                      # fraction of the span inked per row
            rows = prof >= 0.45
            k = 0
            while k < h:
                if not rows[k]:
                    k += 1; continue
                k2 = k
                while k2 + 1 < h and rows[k2 + 1]:
                    k2 += 1
                gh = k2 - k + 1
                cols = comp[k:k2 + 1].any(axis=0)
                xs = np.nonzero(cols)[0]
                xa, xb = x + xs.min(), x + xs.max() + 1
                if xb - xa >= min_face:
                    yc = y + (k + k2) / 2.0
                    raw = ink[y + k:y + k2 + 1, xa:xb]
                    cover = raw.max(axis=0).mean() if raw.size else 0      # continuity of the real ink
                    if gh <= 4:
                        if cover >= 0.8:                                   # dashed / dotted lines are not walls
                            faces.append((yc, float(xa), float(xb)))
                    elif t_min <= gh <= t_max and raw.mean() > 0.85 and (xb - xa) * scale >= 1.0:
                        # solid band = filled wall of this thickness
                        p0 = Mi @ np.array([xa, yc, 1.0]); p1 = Mi @ np.array([xb, yc, 1.0])
                        u0, u1 = sorted([p0 @ e_, p1 @ e_])
                        items.append([A, float(((p0 + p1) / 2) @ nn_), float(u0), float(u1), float(gh)])
                k = k2 + 1
        faces.sort()
        H, W = ink.shape
        for k, (y1, a0, a1) in enumerate(faces):
            best = None
            for j in range(k + 1, len(faces)):
                y2, b0, b1 = faces[j]
                dy = y2 - y1
                if dy < t_min:
                    continue
                if dy > t_max:
                    break
                o0, o1 = max(a0, b0), min(a1, b1)
                if o1 - o0 < min_face:
                    continue
                r0, r1 = int(round(y1 + 1.5)), int(round(y2 - 1.5))
                if r1 >= r0:
                    band = ink[r0:r1 + 1, int(o0):int(o1)]
                    if band.size and band.mean() > 0.35:      # not white inside -> not a wall
                        continue
                best = (dy, o0, o1)
                break
            if best is None:
                continue
            dy, o0, o1 = best
            yc = y1 + dy / 2
            p0 = Mi @ np.array([o0, yc, 1.0]); p1 = Mi @ np.array([o1, yc, 1.0])
            th = math.radians(A); e = np.array([math.cos(th), math.sin(th)]); nn = np.array([-e[1], e[0]])
            u0, u1 = sorted([p0 @ e, p1 @ e])
            items.append([A, float(((p0 + p1) / 2) @ nn), float(u0), float(u1), float(dy)])
    return items


def detect_walls_combined(polys, gray, scale, angles, p):
    q = dict(p); q['_raw'] = True
    raw = detect_walls(polys, gray, scale, angles, q) + raster_wall_items(gray, scale, angles, p)
    return _clean_walls(raw, scale, p, gray)
