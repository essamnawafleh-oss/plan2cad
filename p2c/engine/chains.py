"""Façade dimension chains -> exact window positions.

A chain is a row of dimensions on one line (0.9 | 0.3 | 0.9 | 1.8 ...).  Each segment
is either an opening (window) or a pier.  We find the façade wall the chain belongs
to, look at the scan inside the wall thickness for every segment, and place windows
with the written widths, spaced exactly by the written values.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

from ..model import Window


def group_chains(dims, scale, ang_tol=2.0, off_tol=2.5, gap_tol=4.0):
    items = []
    for k, d in enumerate(dims):
        p0, p1 = np.array(d.p0, float), np.array(d.p1, float)
        L = np.linalg.norm(p1 - p0)
        if L < 4:
            continue
        u = (p1 - p0) / L
        if u[0] < -1e-6 or (abs(u[0]) < 1e-6 and u[1] < 0):
            u = -u; p0, p1 = p1, p0
        a = math.degrees(math.atan2(u[1], u[0])) % 180
        items.append(dict(k=k, a=a, u=u, p0=p0, p1=p1, d=d))
    chains = []
    used = set()
    for it in items:
        if it['k'] in used:
            continue
        u = it['u']; n = np.array([-u[1], u[0]]); off = it['p0'] @ n
        grp = [j for j in items if j['k'] not in used and min(abs(j['a'] - it['a']), 180 - abs(j['a'] - it['a'])) < ang_tol
               and abs(j['p0'] @ n - off) < off_tol and abs(j['p1'] @ n - off) < off_tol]
        grp.sort(key=lambda j: j['p0'] @ u)
        # build rows of consecutive, non-overlapping segments (stacked rows / overall dims go to their own row)
        rows = []
        for j in grp:
            for r in rows:
                gap = j['p0'] @ u - r[-1]['p1'] @ u
                if -gap_tol <= gap <= gap_tol:
                    r.append(j); break
            else:
                rows.append([j])
        chains.extend(rows)
        for j in grp:
            used.add(j['k'])
    out = []
    for c in chains:
        u = c[0]['u']; n = np.array([-u[1], u[0]])
        off = float(np.mean([j['p0'] @ n for j in c]))
        # exact tick positions: anchor at measured ticks, spacing = written values
        vals = [j['d'].value / scale for j in c]
        meas = [c[0]['p0'] @ u] + [j['p1'] @ u for j in c]
        cum = np.concatenate([[0], np.cumsum(vals)])
        start = float(np.mean(np.array(meas) - cum))
        ticks = start + cum
        out.append(dict(u=u, n=n, off=off, ticks=ticks, vals=[j['d'].value for j in c],
                        texts=[j['d'].text for j in c], ks=[j['k'] for j in c]))
    return out


def extend_chain(c, ink, scale, texts=None, max_len=60.0):
    """Follow the dimension line beyond the read numbers and collect every tick on it."""
    H, W = ink.shape
    u, n, off = c['u'], c['n'], c['off']
    def on(t, o=0.0):
        q = u * t + n * (off + o)
        x, y = int(round(q[0])), int(round(q[1]))
        return 0 <= x < W and 0 <= y < H and ink[y, x]
    def line_at(t):
        return on(t) or on(t, 1) or on(t, -1)
    def is_tick(t):
        hits = 0
        for k in (3, 4, 5, 6):
            for dvec in ((0, 1), (1, 1), (-1, 1)):            # perpendicular or slash ticks
                a, b = dvec
                for sg in (1, -1):
                    q = u * (t + sg * a * k * 0.7) + n * (off + sg * b * k * 0.7)
                    x, y = int(round(q[0])), int(round(q[1]))
                    if 0 <= x < W and 0 <= y < H and ink[y, x]:
                        hits += 1
        return hits >= 6
    ticks = list(c['ticks'])
    for direction in (-1, 1):
        t = ticks[0] if direction < 0 else ticks[-1]
        gap = 0; last = t
        while abs(t - (ticks[0] if direction < 0 else ticks[-1])) < max_len / scale:
            t += direction
            if line_at(t):
                gap = 0
            else:
                gap += 1
                if gap > 4:
                    break
            if abs(t - last) > 0.25 / scale and is_tick(t):
                # refine to local best within +-1 px
                last = t
                (ticks.insert(0, t) if direction < 0 else ticks.append(t))
    cand = np.array(sorted(ticks))
    # segments proven by a written number sitting on the line between two ticks
    from .dims import parse_value
    segs = [(a_, b_, v, t) for a_, b_, v, t in zip(c['ticks'][:-1], c['ticks'][1:], c['vals'], c['texts'])]
    for txt, box in texts or []:
        v = parse_value(txt)
        if v is None:
            continue
        box = np.array(box, float); ctr = box.mean(axis=0)
        h = max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[1] - box[0]) * 0.25, 4)
        if abs(ctr @ n - off) > 1.1 * h:
            continue
        tc = ctr @ u
        if any(a_ - 1 <= tc <= b_ + 1 for a_, b_, _, _ in segs):
            continue
        best = None
        for i_, a_ in enumerate(cand):
            if a_ >= tc:
                break
            for b_ in cand[i_ + 1:]:
                if b_ <= tc:
                    continue
                err = abs((b_ - a_) * scale - v) / v
                if err < 0.08 and (best is None or err < best[0]):
                    best = (err, a_, b_)
        if best:
            segs.append((best[1], best[2], v, txt))
    segs.sort()
    # remove overlaps (keep the earlier / known one)
    clean = []
    for sg in segs:
        if clean and sg[0] < clean[-1][1] - 2:
            continue
        clean.append(sg)
    # exact positions: each segment keeps its measured centre, width = written value
    out_segs = []
    for a_, b_, v, t in clean:
        m_ = (a_ + b_) / 2; half = v / scale / 2
        out_segs.append((m_ - half, m_ + half, v, t))
    c2 = dict(c)
    c2['segs'] = out_segs
    c2['ticks'] = sorted({x for sg in out_segs for x in sg[:2]})
    return c2


def _wall_frame(w):
    a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
    e = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    return a, b, e, np.array([-e[1], e[0]])


def facade_line(chain, walls, scale, max_dist=7.0):
    """All collinear wall pieces parallel to the chain, as one façade line nearest to it.
    Returns dict(idx=[wall indices], off=centre offset along m, t=thickness, m=normal, lo, hi) or None."""
    u, n, off = chain['u'], chain['n'], chain['off']
    t0, t1 = chain['ticks'][0], chain['ticks'][-1]
    cand = []
    for i, w in enumerate(walls):
        if len(w.pts) != 2 or w.t < 0.15:
            continue
        a, b, e, m = _wall_frame(w)
        if abs(e @ u) < 0.985:
            continue
        cand.append((float(((a + b) / 2) @ n), i))
    cand.sort()
    groups = []
    for o, i in cand:
        if groups and abs(o - groups[-1][-1][0]) < 3.0:
            groups[-1].append((o, i))
        else:
            groups.append([(o, i)])
    best = None
    for g in groups:
        idx = [i for _, i in g]
        spans = []
        for i in idx:
            a, b, e, m = _wall_frame(walls[i]); spans.append(sorted([a @ u, b @ u]))
        lo = min(x for x, _ in spans); hi = max(y for _, y in spans)
        cov = min(hi, t1) - max(lo, t0)
        if cov < 0.5 * (t1 - t0):
            continue
        o = float(np.mean([o for o, _ in g]))
        dist = abs(o - off) * scale
        if dist > max_dist:
            continue
        if best is None or dist < best[0]:
            best = (dist, dict(idx=idx, off=o, t=float(np.median([walls[i].t for i in idx])), lo=lo, hi=hi, spans=spans))
    return None if best is None else best[1]


def classify_segment(gat, centre_line, u, m, t_px, s0, s1):
    """'window' if the inside of the wall shows continuous glazing lines with white between."""
    L = s1 - s0
    if L < 6:
        return 'wall'
    ss = np.linspace(s0 + 0.15 * L, s1 - 0.15 * L, max(8, int(L * 0.7)))
    offs = np.linspace(-t_px / 2 + 1.5, t_px / 2 - 1.5, max(5, int(t_px)))
    cover = []
    for o in offs:
        pts = centre_line[None, :] + ss[:, None] * u + o * m
        cover.append((gat(pts) < 170).mean())
    cover = np.array(cover)
    inner = cover[1:-1] if len(cover) > 4 else cover
    has_line = (inner > 0.7).any()
    has_gap = (inner < 0.25).any()
    if inner.mean() > 0.6:             # solid fill or hatching
        return 'wall'
    return 'window' if has_line and has_gap else 'wall'


def windows_from_chains(project, gray, min_w=0.45, max_w=4.5):
    s = project.scale
    if not project.dims or not project.walls:
        return dict(chains=0, windows=0)
    gb = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)

    def gat(pts):
        pts = pts.reshape(-1, 2)
        return cv2.remap(gb, pts[:, 0].astype(np.float32).reshape(1, -1), pts[:, 1].astype(np.float32).reshape(1, -1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255).ravel()

    from shapely.geometry import Point
    from shapely.ops import unary_union
    from ..geometry import wall_polygon
    hull = unary_union([wall_polygon(w, s) for w in project.walls]).convex_hull.buffer(-0.3 / s)
    def outside(c):
        mid = c['u'] * (c['ticks'][0] + c['ticks'][-1]) / 2 + c['n'] * c['off']
        return not hull.contains(Point(mid))
    ink = (cv2.GaussianBlur(gray, (3, 3), 0) < 180)
    chains = [extend_chain(c, ink, s, project.texts_raw) for c in group_chains(project.dims, s) if len(c['vals']) >= 2 or outside(c)]
    # merge chains that became the same line after extension
    uniq = []
    for c in sorted(chains, key=lambda c: -len(c['segs'])):
        if any(abs(abs(c['u'] @ d['u']) - 1) < 1e-3 and abs(c['off'] - d['off']) < 3 and
               set(round(x) for x in c['ticks']) <= set(round(x) for x in d['ticks']) for d in uniq):
            continue
        uniq.append(c)
    chains = uniq
    project.chains = [dict(u=list(map(float, c['u'])), n=list(map(float, c['n'])), off=float(c['off']),
                           ticks=list(map(float, c['ticks'])), texts=[sg[3] for sg in c['segs']]) for c in chains]
    n_win = n_ch = 0
    new_wins = []
    from ..model import Wall
    for c in chains:
        F = facade_line(c, project.walls, s)
        if F is None:
            continue
        u, n = c['u'], c['n']
        t_px = F['t'] / s
        origin = n * F['off']                    # centre line: origin + u * t
        m = n
        found = []
        for s0, s1, v, txt in c['segs']:
            if s0 < F['lo'] - 2 or s1 > F['hi'] + 2:
                continue
            if not (min_w <= v <= max_w):
                continue
            if classify_segment(gat, origin, u, m, t_px, s0, s1) == 'window':
                found.append((s0, s1))
        if not found:
            continue
        n_ch += 1
        keep = []
        for wi in project.windows:
            p0, p1 = np.array(wi.p0), np.array(wi.p1)
            mid = (p0 + p1) / 2
            d_ = p1 - p0; Ld = np.linalg.norm(d_)
            a0, a1 = sorted([p0 @ u, p1 @ u])
            hit = Ld > 0 and abs(mid @ n - F['off']) < t_px and abs(d_ @ u) > 0.9 * Ld and \
                any(min(a1, s1) - max(a0, s0) > 0 for s0, s1 in found)
            if not hit:
                keep.append(wi)
        project.windows = keep
        for s0, s1 in found:
            q0 = origin + u * s0; q1 = origin + u * s1
            new_wins.append(Window(tuple(map(float, q0)), tuple(map(float, q1)), F['t']))
            n_win += 1
            # window in a gap between wall pieces: fill the wall around it
            covered = sum(max(0, min(y, s1) - max(x, s0)) for x, y in F['spans'])
            if covered < 0.9 * (s1 - s0):
                project.walls.append(Wall(pts=[tuple(map(float, origin + u * (s0 - 1))), tuple(map(float, origin + u * (s1 + 1)))], t=F['t']))
    project.windows.extend(new_wins)
    info = dict(chains=n_ch, windows=n_win, chains_total=len(chains))
    project.chain_info = info
    return info
