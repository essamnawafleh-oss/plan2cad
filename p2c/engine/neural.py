"""Local neural network (no internet): segments walls, doors and windows.

The model (ONNX, runs on onnxruntime) was trained on CubiCasa5K plans augmented to
look like scans.  Its wall mask replaces the hand-made wall rules; the clean vector
geometry is then produced by the same tools as the rule engine.
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import cv2

TGT_T = 7.0          # wall thickness (px) the model was trained at
_SESS = None


def model_path():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, 'assets', 'plan_seg.onnx')
    if not os.path.exists(p) and getattr(sys, '_MEIPASS', None):
        p = os.path.join(sys._MEIPASS, 'p2c', 'assets', 'plan_seg.onnx')
    return p


def available():
    try:
        import onnxruntime  # noqa
    except Exception:
        return False
    return os.path.exists(model_path())


def _session():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        _SESS = ort.InferenceSession(model_path(), so, providers=['CPUExecutionProvider'])
    return _SESS


# safety limits for the network input (the resize factor follows the scale, which may be wrong)
F_MIN, F_MAX = 0.12, 3.0
MAX_SIDE = 6000              # longest side of the resized image fed to the network (px)


def input_factor(scale, shape, wall_t=0.18):
    """Resize factor so a wall of `wall_t` m becomes TGT_T px, clamped so a wrong scale can neither
    blow the image up to gigabytes nor shrink it to nothing."""
    f = TGT_T / (wall_t / max(float(scale), 1e-6))
    f = min(max(f, F_MIN), F_MAX)
    return min(f, MAX_SIDE / max(shape[:2]))


def segment(gray, scale, wall_t=0.18, tile=768, pad=64, progress=None, factor=None):
    """Class probability maps (4, H, W) float16 at the original resolution: bg, wall, door, window."""
    sess = _session()
    f = factor if factor is not None else input_factor(scale, gray.shape, wall_t)
    H0, W0 = gray.shape
    img = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    x = 1.0 - img.astype(np.float32) / 255.0
    del img
    H, W = x.shape
    out = np.zeros((4, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    step = tile - 2 * pad
    ys = list(range(0, max(1, H - 2 * pad), step)); xs = list(range(0, max(1, W - 2 * pad), step))
    n = len(ys) * len(xs); k = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            c = x[y0:y1, x0:x1]
            ph, pw = (-c.shape[0]) % 16, (-c.shape[1]) % 16
            c = np.pad(c, ((0, ph), (0, pw)))
            logit = sess.run(None, {'x': c[None, None]})[0][0][:, :y1 - y0, :x1 - x0]
            e = np.exp(logit - logit.max(0, keepdims=True)); p = e / e.sum(0, keepdims=True)
            out[:, y0:y1, x0:x1] += p; cnt[y0:y1, x0:x1] += 1
            k += 1
            if progress:
                progress(k / n)
    out /= np.maximum(cnt, 1)
    del cnt, x
    res = np.empty((4, H0, W0), np.float16)
    for i in range(4):
        res[i] = cv2.resize(out[i], (W0, H0), interpolation=cv2.INTER_LINEAR)
    return res


def walls_from_prob(prob, gray, scale, angles, params):
    """Wall mask -> clean double-line walls.  Windows sit inside walls, so they are part of the wall body
    here (they are cut out again when the drawing is built)."""
    wall = ((prob[1] > 0.5) | (prob[3] > 0.5)).astype(np.uint8)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    H, W = wall.shape; m = max(3, int(0.008 * max(H, W)))
    wall[:m] = 0; wall[-m:] = 0; wall[:, :m] = 0; wall[:, -m:] = 0          # page edge
    n, lab, st, _ = cv2.connectedComponentsWithStats(wall, 8)
    for i in range(1, n):                     # isolated frame strokes (a lone long thin line along the edge)
        x, y, w, h, a = st[i]
        if (w > 0.6 * W and h < 0.03 * H and (y < 0.06 * H or y + h > 0.94 * H)) or \
           (h > 0.6 * H and w < 0.03 * W and (x < 0.06 * W or x + w > 0.94 * W)):
            wall[lab == i] = 0
    return mask_walls(wall, scale, angles, params)


def _rot(m, A):
    h, w = m.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), A, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s_ + w * c) + 2, int(h * c + w * s_) + 2
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(m, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0), M


def _runs(comp, t_hint=None):
    """Split a horizontal band component into straight pieces: (x0, x1, y_centre, thickness)."""
    h, w = comp.shape
    has = comp.any(axis=0)
    top = np.where(has, comp.argmax(axis=0), 0)
    bot = np.where(has, h - 1 - comp[::-1].argmax(axis=0), 0)
    out, cur = [], None
    for x in range(w):
        if not has[x]:
            if cur is not None and x - cur['last'] > 2:
                out.append(cur); cur = None
            continue
        c, t = (top[x] + bot[x] + 1) / 2.0, float(bot[x] - top[x] + 1)
        if cur is not None:
            mc, mt = np.median(cur['c'][-25:]), np.median(cur['t'][-25:])
            if abs(c - mc) > 0.35 * mt + 1.0 or abs(t - mt) > 0.5 * mt + 2.0:
                # a short disturbance (junction, text) does not break the run if it settles back
                look = [(top[k] + bot[k] + 1) / 2.0 for k in range(x, min(w, x + 6)) if has[k]]
                if not look or abs(np.median(look) - mc) > 0.35 * mt + 1.0:
                    out.append(cur); cur = None
                else:
                    cur['last'] = x
                    continue
        if cur is None:
            cur = dict(x0=x, last=x, c=[], t=[])
        cur['c'].append(c); cur['t'].append(t); cur['last'] = x
    if cur is not None:
        out.append(cur)
    return [(r['x0'], r['last'] + 1, float(np.median(r['c'])), float(np.median(r['t']))) for r in out]


def mask_walls(wall, scale, angles, params, min_len=0.5, t_max=0.5):
    """Straight wall runs in a binary mask, one pass per drawing direction.  A piece that lies mostly
    inside a wall of another direction (a junction blob) is not a wall of its own."""
    from . import detect as D
    from ..model import Wall
    L = max(5, int(min_len / scale)); tmax = t_max / scale
    TQ = np.array(params.get('wall_thicknesses', [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35]))
    H, W = wall.shape
    fams = D._families(angles)
    dt = cv2.distanceTransform(wall.astype(np.uint8), cv2.DIST_L2, 3)
    try:
        sk = cv2.ximgproc.thinning(wall.astype(np.uint8) * 255) > 0
        t_est = float(2 * np.median(dt[sk])) if sk.any() else 6.0
    except Exception:
        t_est = 6.0
    # a band of thickness t crossed at ~35 deg gives runs of ~1.8 t: kernels must be longer than that
    L = max(L, int(2.6 * t_est) + 1)
    L2 = max(int(1.0 / scale), int(4.0 * t_est))
    rot, opened, back = {}, {}, {}
    for A in fams:
        r, M = _rot(wall.astype(np.uint8), A)
        o = cv2.morphologyEx(r, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
        o2 = cv2.morphologyEx(r, cv2.MORPH_OPEN, np.ones((1, L2), np.uint8))
        rot[A], opened[A] = (r, M), o
        back[A] = cv2.warpAffine(o2, cv2.invertAffineTransform(M), (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    out = []
    for A in fams:
        (r, M), o = rot[A], opened[A]
        Mi = cv2.invertAffineTransform(M)
        oth = np.zeros((H, W), np.uint8)
        for B in fams:
            if B != A:
                oth |= back[B]
        oth_r = cv2.warpAffine(oth, M, (r.shape[1], r.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
        n, lab, st, _ = cv2.connectedComponentsWithStats(o, 8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if w < L:
                continue
            comp = lab[y:y + h, x:x + w] == i
            for x0, x1, yc, t in _runs(comp):
                if t > tmax or t < 2 or x1 - x0 < max(L, 2.5 * t):
                    continue
                band = comp[:, x0:x1]
                if (band & (oth_r[y:y + h, x + x0:x + x1] > 0)).sum() > 0.5 * band.sum():
                    continue
                p0 = Mi @ np.array([x + x0, y + yc - 0.5, 1.0]); p1 = Mi @ np.array([x + x1, y + yc - 0.5, 1.0])
                tm = float(TQ[np.argmin(abs(TQ - t * scale))])
                out.append(Wall(pts=[tuple(map(float, p0)), tuple(map(float, p1))], t=tm))
    return D._merge_wall_list(out, scale)


def openings_from_prob(prob, walls, scale):
    """Door / window masks -> Door and Window objects projected onto the walls."""
    from ..model import Door, Window
    from ..geometry import wall_polygon
    doors, windows = [], []
    if not walls:
        return doors, windows
    segs = []
    for w in walls:
        a, b = np.array(w.pts[0], float), np.array(w.pts[-1], float)
        L = np.linalg.norm(b - a)
        if L > 1:
            segs.append((a, b, (b - a) / L, L, w.t))
    for cls in (2, 3):
        m = (prob[cls] > 0.5).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if st[i][4] < (0.15 / scale) ** 2:
                continue
            ys, xs = np.nonzero(lab == i)
            pts = np.c_[xs, ys].astype(float)
            c = pts.mean(axis=0)
            best = None
            for a, b, u, L, t in segs:
                tt = np.clip((c - a) @ u, 0, L); q = a + u * tt
                d = np.linalg.norm(c - q)
                if best is None or d < best[0]:
                    best = (d, a, u, L, t)
            if best is None or best[0] > 0.6 / scale:
                continue
            d, a, u, L, t = best
            proj = (pts - a) @ u
            s0, s1 = np.percentile(proj, 2), np.percentile(proj, 98)
            if (s1 - s0) * scale < 0.4:
                continue
            p0, p1 = a + u * s0, a + u * s1
            if cls == 3:
                windows.append(Window(tuple(map(float, p0)), tuple(map(float, p1)), t))
            else:
                doors.append(Door(tuple(map(float, p0)), tuple(map(float, p1)), 1, bool((s1 - s0) * scale > 1.3)))
    return doors, windows


def orient_doors(doors, gray, scale):
    """Pick hinge side and swing direction of each door from the drawn arc."""
    dark = cv2.dilate((cv2.GaussianBlur(gray, (3, 3), 0) < 175).astype(np.float32), np.ones((3, 3), np.uint8))
    H, W = dark.shape

    def dk(pts):
        v = [dark[int(round(y)), int(round(x))] for x, y in pts if 0 <= int(round(x)) < W and 0 <= int(round(y)) < H]
        return float(np.mean(v)) if v else 0.0
    out = []
    for d in doors:
        q0, q1 = np.array(d.p0, float), np.array(d.p1, float)
        w = np.linalg.norm(q1 - q0)
        if w < 2:
            continue
        u = (q1 - q0) / w; n = np.array([-u[1], u[0]])
        best = None
        for side in (1, -1):
            nn = n * side
            if d.double:
                sc = 0
                for h, uu in ((q0, u), (q1, -u)):
                    a0 = math.atan2(uu[1], uu[0]); a1 = math.atan2(nn[1], nn[0]); da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                    sc += 0.5 * dk([h + w / 2 * 0.97 * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(a0 + da * .15, a0 + da * .85, 16)])
                cand = (sc, q0, q1, side)
                if best is None or cand[0] > best[0]:
                    best = cand
            else:
                for h, o in ((q0, q1), (q1, q0)):
                    uu = (o - h) / w
                    a0 = math.atan2(uu[1], uu[0]); a1 = math.atan2(nn[1], nn[0]); da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                    sc = dk([h + w * 0.97 * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(a0 + da * .15, a0 + da * .85, 24)])
                    s2 = 1 if np.dot(np.array([-uu[1], uu[0]]), nn) > 0 else -1
                    cand = (sc, h, o, s2)
                    if best is None or cand[0] > best[0]:
                        best = cand
        sc, h, o, sd = best
        d.p0, d.p1, d.side = tuple(map(float, h)), tuple(map(float, o)), int(sd)
        out.append(d)
    return out


def trim_overshoots(walls, scale):
    """Clean corners: a wall end that pokes past a crossing wall (the '+' stubs) or stops just short of it
    is moved to that wall's outer face, so L-corners close neatly and T-junctions meet flush."""
    segs = [(np.array(w.pts[0], float), np.array(w.pts[-1], float)) for w in walls]
    new = []
    for i, w in enumerate(walls):
        a, b = segs[i]
        L = np.linalg.norm(b - a)
        if L < 1:
            new.append((a, b)); continue
        u = (b - a) / L
        ti = w.t / scale
        pos = [0.0, L]
        for end in (0, 1):
            best = None
            for j, (c, d) in enumerate(segs):
                if j == i:
                    continue
                v = d - c; Lv = np.linalg.norm(v)
                if Lv < 1:
                    continue
                sinang = abs(u[0] * v[1] - u[1] * v[0]) / Lv
                if sinang < 0.5:
                    continue
                tt, ss = np.linalg.solve(np.array([u, -v]).T, c - a)
                ext = ti / 2 + 0.15 / scale
                if not (-ext <= ss * Lv <= Lv + ext):
                    continue
                if (tt > 0.5 * L) != (end == 1):
                    continue
                tj = walls[j].t / scale
                face = tj / 2 / sinang
                off = max(0.0, face - ti / 2)          # walls are drawn with square caps of t/2
                target = tt + off if end == 1 else tt - off
                dist = abs(pos[end] - target)
                if dist < 1.6 * tj + 0.12 / scale and (best is None or dist < best[0]):
                    best = (dist, target)
            if best:
                pos[end] = best[1]
        if pos[1] - pos[0] > 0.3 / scale:
            new.append((a + u * pos[0], a + u * pos[1]))
        else:
            new.append((a, b))
    for w, (a, b) in zip(walls, new):
        w.pts = [tuple(map(float, a)), tuple(map(float, b))]
    return walls


def _door_scale(prob_door, door_w=0.82):
    n, lab, st, _ = cv2.connectedComponentsWithStats((prob_door > 0.5).astype(np.uint8), 8)
    Ls = []
    for i in range(1, n):
        if st[i][4] < 20:
            continue
        ys, xs = np.nonzero(lab[st[i][1]:st[i][1] + st[i][3], st[i][0]:st[i][0] + st[i][2]] == i)
        (_, _), (w, h), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        if max(w, h) > 2.5 * min(w, h) + 1:
            Ls.append(max(w, h))
    if len(Ls) < 3:
        return None, len(Ls)
    return door_w / float(np.median(Ls)), len(Ls)


def _ink_crops(gray, side_px, k=4):
    """Up to k square crops (in original px) of the most ink-dense parts of the drawing."""
    H, W = gray.shape
    if max(H, W) <= side_px:
        return [gray]
    side = int(min(side_px, H, W))
    ink = (gray < 180).astype(np.float32)
    small = cv2.resize(ink, (max(1, W // 16), max(1, H // 16)), interpolation=cv2.INTER_AREA)
    win = max(1, side // 16)
    dens = cv2.blur(small, (win, win))
    crops, taken = [], np.zeros_like(dens, bool)
    for _ in range(k):
        d = np.where(taken, -1, dens)
        yy, xx = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[yy, xx] <= 0:
            break
        cy, cx = yy * 16, xx * 16
        y0 = int(np.clip(cy - side // 2, 0, H - side)); x0 = int(np.clip(cx - side // 2, 0, W - side))
        crops.append(gray[y0:y0 + side, x0:x0 + side])
        taken[max(0, yy - win):yy + win + 1, max(0, xx - win):xx + win + 1] = True
    return crops


def estimate_scale(gray, progress=None, max_input=2048):
    """Scale (m/px) of a drawing without dimensions: the network is run at several trial scales and the
    one at which the doors it finds are a normal door width (self-consistent) is kept.  Each trial looks
    at no more than ~4 crops of max_input network pixels, so the cost stays bounded on big images."""
    grid = [0.006, 0.0085, 0.012, 0.017, 0.024, 0.034, 0.045]
    res = []
    for k, s in enumerate(grid):
        f = input_factor(s, gray.shape)
        Ls_all, n_all = [], 0
        for crop in _ink_crops(gray, int(max_input / f)):
            p = segment(crop, s, factor=f)
            imp, n = _door_scale(p[2].astype(np.float32))
            if imp:
                Ls_all += [imp] * n
            n_all += n
        imp = float(np.median(Ls_all)) if len(Ls_all) >= 3 else None
        res.append((s, imp, n_all))
        if progress:
            progress((k + 1) / len(grid))
    best = None
    for (s0, i0, n0), (s1, i1, n1) in zip(res, res[1:]):
        if not i0 or not i1:
            continue
        r0, r1 = math.log(i0 / s0), math.log(i1 / s1)
        if r0 >= 0 >= r1:                                   # implied scale crosses the trial scale
            f = r0 / (r0 - r1) if r0 != r1 else 0.5
            s = math.exp(math.log(s0) + f * (math.log(s1) - math.log(s0)))
            if best is None or n0 + n1 > best[1]:
                best = (s, n0 + n1)
    if best is None:                                        # no crossing: closest agreement
        ok = [(abs(math.log(i / s)), s, i) for s, i, n in res if i]
        if not ok:
            return None
        e, s, i = min(ok)
        if e > 0.35:
            return None
        return math.sqrt(s * i)
    return best[0]
