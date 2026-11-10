"""Read dimension annotations (numbers + dimension lines) before drawing.

For every number found by OCR we look for the dimension line it sits on and the
two ticks / extension lines that bound it.  value / pixel-length gives a scale
estimate; the median over all dimensions is the drawing scale.
"""
from __future__ import annotations

import math
import re
import numpy as np
import cv2

_OCR = None
NUM_RE = re.compile(r'^\d{1,3}(?:[.,]\d{1,3})?$')


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def ocr_available():
    try:
        import rapidocr_onnxruntime  # noqa
        return True
    except Exception:
        return False


def _rot(gray, ang):
    """Rotate so that image direction `ang` (deg, y down) becomes horizontal. Returns img, M (orig->rot)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255), M


def _inv(M):
    return cv2.invertAffineTransform(M)


def _tf(M, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def read_texts(gray, angles, up=3, min_conf=0.5, progress=None):
    """OCR in every drawing direction. Returns list of dicts (orig px)."""
    dirs = []
    for a in angles:
        for d in (a % 180, (a + 90) % 180):
            if all(min(abs(d - e), 180 - abs(d - e)) > 3 for e in dirs):
                dirs.append(d)
    out = []
    eng = _ocr()
    for k, d in enumerate(dirs):
        if progress:
            progress(k / len(dirs))
        # text running along image direction d (y down). Rotating by +d (cv2 = CCW on screen) aligns it.
        for flip in (0, 180):
            img, M = _rot(gray, d + flip)
            big = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            res, _ = eng(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR), use_cls=False)
            Mi = _inv(M)
            for box, txt, conf in res or []:
                t = txt.strip().replace(' ', '')
                conf = float(conf)
                if conf < min_conf:
                    continue
                b = np.array(box, float) / up
                wbox = np.linalg.norm(b[1] - b[0]); hbox = np.linalg.norm(b[3] - b[0])
                if wbox < hbox * 0.9:            # not horizontal in this frame
                    continue
                ob = _tf(Mi, b)
                out.append(dict(text=t, conf=conf, box=ob, dir=d, flip=flip, h=hbox, w=wbox,
                                rbox=b, M=M))
    if progress:
        progress(1.0)
    # dedupe: same place -> keep the best
    out.sort(key=lambda r: -r['conf'])
    keep = []
    for r in out:
        c = r['box'].mean(axis=0)
        if all(np.linalg.norm(c - q['box'].mean(axis=0)) > max(4, 0.5 * r['h']) for q in keep):
            keep.append(r)
    return keep


UNIT_RE = re.compile(r'^(\d{1,5}(?:[.,]\d{1,3})?)(mm|cm|m)?$', re.I)


def parse_hypotheses(t):
    """All readings of a dimension text as (value_m, unit, prior).
    '3.45' / '3,45' -> metres; '3.45m', '345cm', '3450mm' -> explicit unit;
    a plain integer is ambiguous: '345' may be cm, '3450' may be mm (or cm) -> several hypotheses with
    lower priors; the drawing-wide scale vote decides which reading is consistent."""
    t = t.strip().replace(' ', '').replace('\u066b', '.').replace('\u060c', ',')
    t = t.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789'))
    m = UNIT_RE.match(t)
    if not m:
        return []
    num, unit = m.group(1).replace(',', '.'), (m.group(2) or '').lower()
    try:
        v = float(num)
    except ValueError:
        return []
    out = []
    if unit:
        f = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}[unit]
        out.append((v * f, unit, 1.0))
    elif '.' in num:
        if len(num.split('.')[1]) <= 2 or v < 100:
            out.append((v, 'm', 1.0))
    else:                                           # integer
        n = len(num)
        if 2 <= n <= 4 and num[0] != '0':
            out.append((v * 0.01, 'cm', 0.7))
        if 3 <= n <= 5 and num[0] != '0':
            out.append((v * 0.001, 'mm', 0.5 if n >= 4 else 0.3))
    return [(val, u, pr) for val, u, pr in out if 0.03 <= val <= 200]


def parse_value(t):
    """Unambiguous value in metres (decimal or explicit unit), else None.  Kept for callers that need
    a sure reading (e.g. wall-thickness labels)."""
    h = [x for x in parse_hypotheses(t) if x[2] >= 1.0]
    return h[0][0] if h else None


def measure_dimensions(gray, texts, progress=None, scale_hint=None):
    """For every numeric text, locate its dimension segment. Adds 'value', 'seg' (2 pts orig px), 'len_px'."""
    ink_full = (cv2.GaussianBlur(gray, (3, 3), 0) < 175).astype(np.uint8)
    cache = {}
    dims = []
    for r in texts:
        hyps = parse_hypotheses(r['text'])
        if not hyps:
            continue
        v = hyps[0][0]
        key = (round(r['dir'], 2), r['flip'])
        if key not in cache:
            img, M = _rot(gray, r['dir'] + r['flip'])
            cache[key] = ((cv2.GaussianBlur(img, (3, 3), 0) < 175).astype(np.uint8), M)
        ink, M = cache[key]
        H, W = ink.shape
        b = r['rbox']
        x0, x1 = b[:, 0].min(), b[:, 0].max(); y0, y1 = b[:, 1].min(), b[:, 1].max()
        th = max(y1 - y0, 4)
        cx = (x0 + x1) / 2
        # erase the text itself
        work = ink.copy()
        work[int(max(0, y0 - 1)):int(min(H, y1 + 2)), int(max(0, x0 - 1)):int(min(W, x1 + 2))] = 0
        best = None
        for side, rng in (('below', range(int(y1), int(min(H - 1, y1 + 1.4 * th)))),
                          ('above', range(int(y0), int(max(0, y0 - 1.4 * th)), -1))):
            for y in rng:
                xa, xb = int(max(0, x0 - th)), int(min(W, x1 + th))
                row = work[max(0, y - 1):y + 2, xa:xb].max(axis=0)
                if row.mean() > 0.85:
                    best = (y, side)
                    break
            if best:
                break
        if not best:
            continue
        ly = best[0]
        line = work[max(0, ly - 1):ly + 2].max(axis=0)
        reach = int(th * 0.9)

        def is_tick(x):
            up_ = work[max(0, ly - reach):max(0, ly - 2), max(0, x - 1):x + 2]
            dn_ = work[ly + 3:ly + reach, max(0, x - 1):x + 2]
            return up_.size and dn_.size and up_.max() and dn_.max() and \
                (up_.mean() > 0.18 or dn_.mean() > 0.18)

        def ticks(step):
            x = int(cx); gap = 0; found = []
            while 0 < x < W - 1 and len(found) < 8:
                x += step
                if line[x]:
                    gap = 0
                else:
                    gap += 1
                    if gap > 3:
                        break
                if abs(x - cx) > th * 1.0 and is_tick(x) and (not found or abs(x - found[-1]) > 3):
                    found.append(x)
                if abs(x - cx) > 80 * th:
                    break
            return found
        TL, TR = ticks(-1), ticks(1)
        if not TL or not TR:
            continue
        pairs = [(xl, xr) for xl in TL for xr in TR if xr - xl >= max(x1 - x0, 6)]
        if not pairs:
            continue
        if scale_hint:
            err, xl, xr, v, unit = min((abs(math.log(hv / (xr_ - xl_) / scale_hint)), xl_, xr_, hv, hu)
                                       for xl_, xr_ in pairs for hv, hu, hp in hyps)
            if err > 0.06:
                continue
        else:
            d = dict(r); d.update(value=v, unit=hyps[0][1],
                                  hyps=[(hv, hu, hp, [hv / (xr_ - xl_) for xl_, xr_ in pairs]) for hv, hu, hp in hyps],
                                  cands=[v / (xr_ - xl_) for xl_, xr_ in pairs])
            dims.append(d)
            continue
        L = xr - xl
        seg_r = np.array([[xl, ly], [xr, ly]], float)
        seg = _tf(_inv(M), seg_r)
        d = dict(r); d.update(value=v, unit=unit, seg=seg, len_px=float(np.linalg.norm(seg[1] - seg[0])),
                              line_y=ly, side=best[1])
        dims.append(d)
    return dims


def vote_scale(cand_dims, tol=0.035):
    """Hough-style vote: each dimension votes once (its best reading) for every scale one of its
    tick pairs implies.  Readings carry priors (decimal metres 1.0, integer-as-cm 0.7, ...).
    Returns (scale, n_supporting, info) where info tells which unit won and how close the runner-up
    unit hypothesis came (an ambiguous drawing is reported, not silently guessed)."""
    if not cand_dims:
        return None, 0, {}
    items = []
    for d in cand_dims:
        hyps = d.get('hyps') or [(d['value'], 'm', 1.0, d['cands'])]
        items.append([(np.log(np.array(c)), pr, u) for hv, u, pr, c in hyps])
    allv = np.concatenate([lc for it in items for lc, pr, u in it])
    best, bv = None, 0.0
    scores = {}
    for c in np.unique(np.round(allv, 3)):
        votes = 0.0; units = {}
        for d, it in zip(cand_dims, items):
            w = 0.0; wu = None
            for lc, pr, u in it:
                if np.any(np.abs(lc - c) < tol) and pr > w:
                    w, wu = pr, u
            if w:
                votes += d['conf'] * w * (1.0 + min(d['value'], 5.0) / 5.0)
                units[wu] = units.get(wu, 0) + 1
        scores[float(c)] = (votes, units)
        if votes > bv:
            bv, best = votes, c
    sel = []
    for it in items:
        cand = [(abs(lc[np.argmin(np.abs(lc - best))] - best), lc[np.argmin(np.abs(lc - best))]) for lc, pr, u in it]
        e, v = min(cand)
        if e < tol:
            sel.append(v)
    # runner-up far from the winner (a different unit reading differs by a factor ~10)
    runner = max([v for c, (v, u) in scores.items() if abs(c - best) > 1.0] or [0.0])
    info = dict(units=scores[float(best)][1], votes=bv, runner_up=runner,
                ambiguous=bool(runner > 0.8 * bv))
    return float(np.exp(np.median(sel))), len(sel), info


def estimate_scale(dims, tol=0.06):
    """Agreement check of measured dims around their median scale."""
    if not dims:
        return None, [], 0.0
    s = np.array([d['value'] / d['len_px'] for d in dims])
    med = float(np.median(s))
    sel = np.abs(np.log(s / med)) < tol
    for d, g in zip(dims, sel):
        d['inlier'] = bool(g)
    return med, [d for d, g in zip(dims, sel) if g], float(sel.mean())


def clutter_mask(gray, texts, dims):
    """Pixels belonging to text and to isolated dimension lines (to be ignored when finding walls)."""
    m = np.zeros(gray.shape, np.uint8)
    for r in texts:
        pts = np.round(r['box']).astype(np.int32)
        cv2.fillPoly(m, [pts], 255)
        cv2.polylines(m, [pts], True, 255, 2)
    ink = (cv2.GaussianBlur(gray, (3, 3), 0) < 175).astype(np.uint8)
    for d in dims:
        if not d.get('inlier', True):
            continue
        a, b = d['seg']
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-u[1], u[0]])
        # only erase when the line has white on both sides (not one face of a wall)
        L = np.linalg.norm(b - a)
        ts = np.linspace(0.1, 0.9, 25)
        side_ink = 0
        for t in ts:
            q = a + (b - a) * t
            for o in (3, 4, 5, -3, -4, -5):
                x, y = (q + n * o).round().astype(int)
                if 0 <= x < ink.shape[1] and 0 <= y < ink.shape[0] and ink[y, x]:
                    side_ink += 1
        if side_ink / (len(ts) * 6) > 0.25:
            continue
        cv2.line(m, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 255, 3)
    return m


def robust_scale(dims):
    """Weighted median of per-dimension scales with MAD outlier rejection.
    Marks every dim with 'status' ('used' or 'rejected:<reason>') and returns (scale, report)."""
    if not dims:
        return None, dict(n=0)
    r = np.array([d['value'] / d['len_px'] for d in dims])
    w = np.array([d.get('conf', 1.0) * (0.5 + min(d['value'], 6.0) / 6.0) for d in dims])
    lr = np.log(r)
    o = np.argsort(lr); cw = np.cumsum(w[o]) / w.sum()
    med = lr[o][np.searchsorted(cw, 0.5)]
    mad = float(np.median(np.abs(lr - med))) * 1.4826
    lim = max(0.03, 3.0 * mad)
    used = np.abs(lr - med) <= lim
    for d, u, x in zip(dims, used, lr):
        if u:
            d['status'] = 'used'
        else:
            dev = (math.exp(x - med) - 1) * 100
            ratio = math.exp(x - med)
            why = 'unit?' if abs(math.log10(ratio)) > 0.8 else ('ocr?' if abs(dev) > 8 else 'inconsistent')
            d['status'] = f'rejected:{why} {dev:+.1f}%'
    if used.sum() == 0:
        return None, dict(n=len(dims))
    s = float(np.exp(np.sum(lr[used] * w[used]) / w[used].sum()))
    spread = float(np.std(lr[used])) if used.sum() > 1 else 0.0
    return s, dict(n=len(dims), used=int(used.sum()), rejected=int((~used).sum()), spread_pct=spread * 100)


def analyse(gray, angles, progress=None):
    """Full dimension understanding: OCR -> tick candidates -> voted scale -> exact segments ->
    robust calibration with a per-dimension status."""
    say = progress or (lambda f: None)
    texts = read_texts(gray, angles, progress=lambda f: say(0.85 * f))
    cand = measure_dimensions(gray, texts)
    scale, nvotes, vinfo = vote_scale(cand)
    dims = measure_dimensions(gray, texts, scale_hint=scale) if scale else []
    calib = {}
    if len(dims) >= 5:
        more = measure_with_scale(gray, texts, scale, units=vinfo.get('units'))
        seen = [d['box'].mean(axis=0) for d in dims]
        for d in more:
            c = d['box'].mean(axis=0)
            if all(np.linalg.norm(c - q) > 3 for q in seen):
                dims.append(d); seen.append(c)
        scale, calib = robust_scale(dims)
        for d in dims:
            d['inlier'] = d.get('status') == 'used'
    else:
        for d in dims:
            d['inlier'] = True; d['status'] = 'used'
        scale = None                                # not enough evidence -> do not trust
    if scale and vinfo.get('ambiguous'):
        calib['warning'] = 'unit ambiguous: integer dimensions fit both cm and mm readings'
        scale = None
    say(1.0)
    return dict(texts=texts, dims=dims, scale=scale, n_texts=len(texts),
                n_numbers=sum(1 for t in texts if parse_hypotheses(t['text'])),
                n_dims=len(dims), agree=(nvotes / max(len(cand), 1)), units=vinfo.get('units', {}),
                calib=calib)


def measure_with_scale(gray, texts, scale, tol=0.07, units=None):
    """Second pass: the scale is known, so every number predicts the length of its dimension line.
    Search rows near the text for a line segment of that length that covers the text, bounded by ticks."""
    cache = {}
    out = []
    for r in texts:
        hyps = parse_hypotheses(r['text'])
        if not hyps:
            continue
        # the reading in the drawing's established unit first, then by prior
        hyps = sorted(hyps, key=lambda h: (h[1] not in (units or {}), -h[2]))
        v, unit, _ = hyps[0]
        want = v / scale
        if want < 8:
            continue
        key = (round(r['dir'], 2), r['flip'])
        if key not in cache:
            img, M = _rot(gray, r['dir'] + r['flip'])
            cache[key] = ((cv2.GaussianBlur(img, (3, 3), 0) < 180).astype(np.uint8), M)
        ink, M = cache[key]
        H, W = ink.shape
        b = r['rbox']
        x0, x1 = b[:, 0].min(), b[:, 0].max(); y0, y1 = b[:, 1].min(), b[:, 1].max()
        th = max(y1 - y0, 4); cx = (x0 + x1) / 2
        best = None
        for y in list(range(int(y1), int(min(H - 2, y1 + 1.6 * th)))) + list(range(int(y0), int(max(1, y0 - 1.6 * th)), -1)):
            row = ink[y - 1:y + 2].max(axis=0).astype(bool)
            row_t = row.copy()
            row_t[int(max(0, x0 - 2)):int(min(W, x1 + 3))] = True          # text may hide the line
            lo = int(max(0, cx - want * (1 + tol) - 2)); hi = int(min(W - 1, cx + want * (1 + tol) + 2))
            reach = int(max(3, th * 0.8))

            def tick(x):
                a = ink[max(0, y - reach):max(0, y - 2), max(0, x - 1):x + 2]
                c = ink[y + 3:y + reach, max(0, x - 1):x + 2]
                return (a.size and a.any(axis=1).mean() > 0.6) or (c.size and c.any(axis=1).mean() > 0.6)
            ticks = [x for x in range(lo, hi + 1) if row[x] and tick(x)]
            if len(ticks) < 2:
                continue
            for i, xl in enumerate(ticks):
                if xl > cx:
                    break
                for xr in ticks[i + 1:]:
                    if xr < cx:
                        continue
                    L = xr - xl
                    if abs(math.log(L / want)) > tol:
                        continue
                    cov = row_t[xl:xr + 1].mean()
                    if cov < 0.85:
                        continue
                    sc = cov - abs(math.log(L / want)) - 0.01 * abs(y - (y1 if y >= y1 else y0))
                    if best is None or sc > best[0]:
                        best = (sc, xl, xr, y)
            if best and best[0] > 0.95:
                break
        if best is None:
            continue
        _, xl, xr, y = best
        seg = _tf(_inv(M), np.array([[xl, y], [xr, y]], float))
        d = dict(r); d.update(value=v, unit=unit, seg=seg, len_px=float(np.linalg.norm(seg[1] - seg[0])), inlier=True)
        out.append(d)
    return out
