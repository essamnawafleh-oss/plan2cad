"""Evidence-based wall geometry: measure every wall on the image itself.

The network (or the rule engine) proposes where walls are; this module checks each proposal against
the drawing and fixes what the evidence supports:

* centre line and thickness are re-measured from many cross-sections (filled, hatched and
  double-line walls, windows included), using a robust median -> different walls keep different
  thicknesses, nothing is forced to one value;
* each wall gets a confidence from how consistently the cross-sections agree (and from the network
  probability when available);
* a wall end is only extended to meet another wall when the gap is tiny (a scanning break) or the
  image / network shows wall material in the gap;
* every geometric change is logged so it can be reviewed.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

STD_T = [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _ink_threshold(gray):
    paper = float(np.percentile(gray, 90)); ink = float(np.percentile(gray, 2))
    return min(175.0, (paper + ink) / 2.0)


def _profiles(gray, a, b, scale, R_px, K=17, step=0.5, f0=0.12, f1=0.88):
    """Gray cross-sections (K x M) perpendicular to a->b, offsets -R..R px."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = np.linalg.norm(b - a)
    u = (b - a) / max(L, 1e-9); n = np.array([-u[1], u[0]])
    ts = np.linspace(f0, f1, K) * L
    offs = np.arange(-R_px, R_px + 1e-9, step)
    P = a[None, None, :] + ts[:, None, None] * u[None, None, :] + offs[None, :, None] * n[None, None, :]
    mx = P[..., 0].astype(np.float32); my = P[..., 1].astype(np.float32)
    prof = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return prof.astype(np.float32), offs


def _runs(row):
    """Contiguous True runs as (start, end) index pairs (end inclusive)."""
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    s = np.nonzero(d == 1)[0]; e = np.nonzero(d == -1)[0] - 1
    return list(zip(s, e))


def measure_wall(w, gray, scale, thr=None, t_min=0.07, t_max=0.45, step=0.5):
    """Cross-section evidence for one wall.  Returns dict with keys
    ok, offset_px (centre shift along the wall normal), t_m, conf, kind, valid_frac."""
    a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
    L = np.linalg.norm(b - a)
    res = dict(ok=False, offset_px=0.0, t_m=w.t, conf=0.0, kind='none', valid_frac=0.0)
    if L * scale < 0.25:
        return res
    thr = _ink_threshold(gray) if thr is None else thr
    R = max(t_max / scale * 0.8, w.t / scale * 1.2) + 2
    prof, offs = _profiles(gray, a, b, scale, R, step=step)
    ink = prof < thr
    tmin, tmax = t_min / scale, t_max / scale
    gap = max(1, int(round(1.0 / step)))
    los, his, fills = [], [], []
    for row, g in zip(ink, prof):
        rr = _runs(row)
        if not rr:
            continue
        merged = [list(rr[0])]
        for s_, e_ in rr[1:]:                          # 1 px breaks inside a stroke are noise
            if s_ - merged[-1][1] <= gap + 1:
                merged[-1][1] = e_
            else:
                merged.append([s_, e_])
        best = None
        for i in range(len(merged)):
            for j in range(i, len(merged)):
                lo, hi = offs[merged[i][0]], offs[merged[j][1]]
                wdt = hi - lo + step
                if wdt < tmin or wdt > tmax:
                    continue
                if not (lo - 1.5 <= 0 <= hi + 1.5) and not (abs((lo + hi) / 2) <= 0.6 * w.t / scale):
                    continue
                # outer neighbours must be paper (the band is bounded by its faces)
                il, ih = merged[i][0], merged[j][1]
                outside_l = row[max(0, il - 3):il].mean() if il > 0 else 0
                outside_h = row[ih + 1:ih + 4].mean() if ih + 1 < len(row) else 0
                if outside_l > 0.5 or outside_h > 0.5:
                    continue
                if best is None or wdt > best[0]:
                    # sub-pixel face positions: where the profile crosses the ink threshold
                    fl = 0.5
                    if il > 0 and g[il - 1] != g[il]:
                        fl = float(np.clip((thr - g[il]) / (g[il - 1] - g[il]), 0, 1))
                    fh = 0.5
                    if ih + 1 < len(g) and g[ih + 1] != g[ih]:
                        fh = float(np.clip((thr - g[ih]) / (g[ih + 1] - g[ih]), 0, 1))
                    best = (wdt, lo - fl * step, hi + fh * step, row[merged[i][0]:merged[j][1] + 1].mean())
        if best:
            los.append(best[1]); his.append(best[2]); fills.append(best[3])
    K = len(prof)
    if len(los) < max(3, 0.3 * K):
        res['valid_frac'] = len(los) / K
        return res
    los, his = np.array(los), np.array(his)
    lo, hi = float(np.median(los)), float(np.median(his))
    agree = float(np.mean((np.abs(los - lo) <= 2.0) & (np.abs(his - hi) <= 2.0)))
    fill = float(np.median(fills))
    kind = 'filled' if fill > 0.75 else ('hatched' if fill > 0.35 else 'double')
    res.update(ok=True, offset_px=(lo + hi) / 2.0, t_m=(hi - lo) * scale, valid_frac=len(los) / K,
               conf=(len(los) / K) * agree, kind=kind, agree=agree, fill=fill)
    return res


def snap_thickness(t):
    """Nearest standard thickness when within 2.5 cm, else rounded to the centimetre."""
    k = int(np.argmin([abs(t - s) for s in STD_T]))
    return STD_T[k] if abs(STD_T[k] - t) <= 0.025 else round(t, 2)


def nn_support(w, prob_wall, scale):
    """Mean network wall probability along the centre line (0..1), or None."""
    if prob_wall is None:
        return None
    a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
    L = np.linalg.norm(b - a)
    n = max(3, int(L / 2))
    P = a + (b - a) * np.linspace(0.05, 0.95, n)[:, None]
    H, W = prob_wall.shape
    x = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
    return float(np.mean(prob_wall[y, x].astype(np.float32)))


def refine_walls(walls, gray, scale, prob_wall=None, log=None, max_shift_m=0.15):
    """Re-measure centre and thickness of every wall from the image, set confidence and origin.
    Returns the list of walls to keep (walls with neither image nor network evidence are dropped)."""
    thr = _ink_threshold(gray)
    keep = []
    for w in walls:
        if len(w.pts) != 2:
            keep.append(w); continue
        m = measure_wall(w, gray, scale, thr)
        sup = nn_support(w, prob_wall, scale)
        if m['ok']:
            a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-u[1], u[0]])
            off = m['offset_px']
            lim = max(0.7 * w.t / scale, max_shift_m / scale)
            t_new = snap_thickness(m['t_m'])
            if abs(off) <= lim and m['agree'] >= 0.5:
                old = (list(w.pts), w.t)
                if abs(off) > 0.75 or abs(t_new - w.t) > 0.009:
                    w.pts = [tuple(map(float, a + n * off)), tuple(map(float, b + n * off))]
                    w.t = t_new
                    if log is not None:
                        log.append(dict(action='refine_wall', shift_m=round(off * scale, 3),
                                        t_old=old[1], t_new=t_new, kind=m['kind'],
                                        before=[list(p) for p in old[0]], after=[list(p) for p in w.pts]))
        img_conf = m['conf'] if m['ok'] else 0.25 * m['valid_frac']
        w.conf = round(float(img_conf if sup is None else 0.5 * img_conf + 0.5 * sup), 3)
        w.origin = 'auto' if w.conf >= 0.5 else 'uncertain'
        if not m['ok'] and m['valid_frac'] < 0.2 and (sup is None or sup < 0.35):
            if log is not None:
                log.append(dict(action='drop_wall', reason='no wall evidence in the image or network',
                                pts=[list(p) for p in w.pts], t=w.t))
            continue
        keep.append(w)
    return keep


def gap_evidence(gray, a, b, t_px, thr, prob_wall=None):
    """Is there wall material between points a and b (a band of width t_px)?"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = np.linalg.norm(b - a)
    if L < 1:
        return True
    n_s = max(3, int(L))
    u = (b - a) / L; nrm = np.array([-u[1], u[0]])
    P = a + (b - a) * np.linspace(0, 1, n_s)[:, None]
    H, W = gray.shape
    cov = []
    for o in np.linspace(-t_px / 3, t_px / 3, 3):
        Q = P + nrm * o
        x = np.clip(np.round(Q[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(Q[:, 1]).astype(int), 0, H - 1)
        cov.append(gray[y, x] < thr)
    ink = float(np.mean(np.any(cov, axis=0)))
    if ink >= 0.6:
        return True
    if prob_wall is not None:
        x = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
        if float(np.mean(prob_wall[y, x].astype(np.float32))) >= 0.4:
            return True
    return False


def join_with_evidence(walls, gray, scale, max_gap=0.6, free_gap=0.12, prob_wall=None, log=None):
    """Like detect.join_walls, but an end is extended over more than `free_gap` metres only when the
    gap itself shows wall material.  Logs every extension."""
    from shapely.geometry import LineString as LS, Point
    from ..geometry import wall_polygon
    thr = _ink_threshold(gray)
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
            if any(j != i and q is not None and q.contains(Point(tip)) for j, q in enumerate(polys)):
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
            dd, j = best
            if dd * scale > free_gap and not gap_evidence(gray, b, b + d * dd, tp, thr, prob_wall):
                if log is not None:
                    log.append(dict(action='join_refused', reason='no wall material in the gap',
                                    gap_m=round(dd * scale, 3), at=list(map(float, b))))
                continue
            new = b + d * (dd + walls[j].t / scale / 2)
            pts = list(w.pts); pts[end] = tuple(map(float, new)); w.pts = pts
            polys[i] = wall_polygon(w, scale)
            if log is not None and dd * scale > 0.02:
                log.append(dict(action='join_wall', gap_m=round(dd * scale, 3), at=list(map(float, b)),
                                to=list(map(float, new))))
    return walls
