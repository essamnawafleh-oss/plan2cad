"""Deterministic synthetic floor plans with ground truth, for tests and benchmarks.

A plan is a rectangle split recursively into rooms.  Ground truth = wall centre lines with
thickness (m), door openings, window positions.  The image can be drawn in several styles
(filled black walls, hatched, hollow double lines, colour CAD with light-gray fill) and then
degraded like a scan or a phone photo.  Every degradation is a known transform, so ground truth
can be mapped into the degraded image.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

SCALE = 0.02                 # m / px of the clean rendering


class Plan:
    def __init__(self):
        self.walls = []      # (p0, p1, t_m)   centre lines in px
        self.doors = []      # (p0, p1)        opening on a wall centre line
        self.windows = []    # (p0, p1, t_m)
        self.size = (0, 0)   # (W, H)
        self.scale = SCALE


def _split(rng, x0, y0, x1, y1, depth, out):
    w, h = x1 - x0, y1 - y0
    if depth == 0 or (w < 7 and h < 7):
        out.append((x0, y0, x1, y1)); return
    if w >= h and w >= 6:
        c = x0 + rng.uniform(0.38, 0.62) * w
        _split(rng, x0, y0, c, y1, depth - 1, out); _split(rng, c, y0, x1, y1, depth - 1, out)
    elif h >= 6:
        c = y0 + rng.uniform(0.38, 0.62) * h
        _split(rng, x0, y0, x1, c, depth - 1, out); _split(rng, x0, c, x1, y1, depth - 1, out)
    else:
        out.append((x0, y0, x1, y1))


def make_plan(seed=0, angled=False, margin_m=2.0, thin=False):
    """Room layout in metres -> Plan in px of the clean image."""
    rng = np.random.default_rng(seed)
    Wm, Hm = rng.uniform(12, 18), rng.uniform(9, 14)
    rooms = []
    _split(rng, 0.0, 0.0, Wm, Hm, 3, rooms)
    s = SCALE
    P = lambda x, y: ((x + margin_m) / s, (y + margin_m) / s)
    pl = Plan()
    T_EXT = 0.30 if not thin else 0.20
    T_INT = [0.10, 0.12, 0.15, 0.20]
    # collect unique interior edges
    edges = {}
    for (x0, y0, x1, y1) in rooms:
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))):
            key = tuple(sorted((tuple(np.round(a, 3)), tuple(np.round(b, 3)))))
            edges[key] = edges.get(key, 0) + 1
    # interior walls: maximal collinear segments
    segs = []
    for (a, b) in edges:
        on_border = (a[0] == b[0] and a[0] in (0.0, round(Wm, 3))) or (a[1] == b[1] and a[1] in (0.0, round(Hm, 3)))
        if not on_border:
            segs.append((a, b))
    merged = _merge_axis(segs)
    t_int = {}
    for a, b in merged:
        t_int[(a, b)] = float(rng.choice(T_INT)) if not thin else 0.10
    # exterior
    corners = [(0, 0), (Wm, 0), (Wm, Hm), (0, Hm)]
    if angled:                                   # cut the bottom-left corner at 45 deg
        c = min(Wm, Hm) * 0.3
        corners = [(0, 0), (Wm, 0), (Wm, Hm), (c, Hm), (0, Hm - c)]
    for i in range(len(corners)):
        a, b = corners[i], corners[(i + 1) % len(corners)]
        pl.walls.append((P(*a), P(*b), T_EXT))
    for (a, b), t in t_int.items():
        if angled:
            c = min(Wm, Hm) * 0.3
            # clip interior walls against the cut corner (keep it simple: drop walls fully inside the cut)
            if a[0] + (Hm - a[1]) < c - 1e-6 and b[0] + (Hm - b[1]) < c - 1e-6:
                continue
        pl.walls.append((P(*a), P(*b), t))
    # openings: one door per interior wall long enough, windows on exterior walls
    for (a, b, t) in list(pl.walls):
        L = math.dist(a, b) * s
        if L < 1.6:
            continue
        u = (np.array(b) - np.array(a)) / (L / s)
        if t == T_EXT:
            n = int(L // 3.5)
            for k in range(n):
                f = (k + 0.5) / n
                c = np.array(a) + (np.array(b) - np.array(a)) * f
                hw = rng.uniform(0.6, 0.8) / s
                pl.windows.append((tuple(c - u * hw), tuple(c + u * hw), t))
        else:
            f = rng.uniform(0.25, 0.75)
            c = np.array(a) + (np.array(b) - np.array(a)) * f
            hw = 0.45 / s
            if math.dist(c, a) * s > 0.7 and math.dist(c, b) * s > 0.7:
                pl.doors.append((tuple(c - u * hw), tuple(c + u * hw)))
    pl.size = (int((Wm + 2 * margin_m) / s), int((Hm + 2 * margin_m) / s))
    pl.rooms = [(P(x0, y0), P(x1, y1)) for x0, y0, x1, y1 in rooms]
    pl.extent_m = (Wm, Hm)
    return pl


def _merge_axis(segs):
    out = []
    used = [False] * len(segs)
    for i, (a, b) in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        a, b = list(a), list(b)
        horiz = a[1] == b[1]
        changed = True
        while changed:
            changed = False
            for j, (c, d) in enumerate(segs):
                if used[j]:
                    continue
                if horiz and c[1] == d[1] == a[1] and (abs(c[0] - b[0]) < 1e-6 or abs(d[0] - a[0]) < 1e-6):
                    a[0], b[0] = min(a[0], c[0]), max(b[0], d[0]); used[j] = True; changed = True
                elif not horiz and c[0] == d[0] == a[0] and (abs(c[1] - b[1]) < 1e-6 or abs(d[1] - a[1]) < 1e-6):
                    a[1], b[1] = min(a[1], c[1]), max(b[1], d[1]); used[j] = True; changed = True
        out.append((tuple(a), tuple(b)))
    return out


# ------------------------------------------------------------------ rendering
def _wall_poly(a, b, t_px, ext=True):
    a, b = np.array(a, float), np.array(b, float)
    u = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-u[1], u[0]])
    e = u * t_px / 2 if ext else 0
    return np.array([a - e - n * t_px / 2, b + e - n * t_px / 2, b + e + n * t_px / 2, a - e + n * t_px / 2])


def wall_mask(pl, shape=None, cut_doors=True, include_windows=True):
    W, H = pl.size if shape is None else (shape[1], shape[0])
    m = np.zeros((H, W), np.uint8)
    for a, b, t in pl.walls:
        cv2.fillPoly(m, [np.round(_wall_poly(a, b, t / pl.scale)).astype(np.int32)], 255)
    if cut_doors:
        for a, b in pl.doors:
            t = 0.5 / pl.scale
            cv2.fillPoly(m, [np.round(_wall_poly(a, b, t, ext=False)).astype(np.int32)], 0)
    if not include_windows:
        for a, b, t in pl.windows:
            cv2.fillPoly(m, [np.round(_wall_poly(a, b, t / pl.scale * 1.05, ext=False)).astype(np.int32)], 0)
    return m


def render(pl, style='filled', seed=0, furniture=True, text=True, dims=True):
    """Return a BGR image of the plan."""
    rng = np.random.default_rng(seed + 1000)
    W, H = pl.size
    img = np.full((H, W, 3), 255, np.uint8)
    s = pl.scale
    m = wall_mask(pl)
    win = np.zeros_like(m)
    for a, b, t in pl.windows:
        cv2.fillPoly(win, [np.round(_wall_poly(a, b, t / s * 1.02, ext=False)).astype(np.int32)], 255)
    body = cv2.bitwise_and(m, cv2.bitwise_not(win))
    edge = cv2.morphologyEx(body, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    wall_col = (0, 0, 0)
    if style == 'filled':
        img[body > 0] = wall_col
    elif style == 'hatched':
        yy, xx = np.mgrid[0:H, 0:W]
        hatch = ((xx + yy) % 7) < 2
        img[(body > 0) & hatch] = wall_col
        img[edge] = wall_col
    elif style == 'hollow':
        img[edge] = wall_col
    elif style == 'colour':
        img[body > 0] = (210, 210, 210)
        img[edge] = (150, 150, 150)
        wall_col = (150, 150, 150)
    # windows: thin lines across the opening
    for a, b, t in pl.windows:
        a, b = np.array(a), np.array(b); u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]])
        tp = t / s
        for o in (-tp / 2, -tp / 8, tp / 8, tp / 2):
            cv2.line(img, tuple(np.round(a + n * o).astype(int)), tuple(np.round(b + n * o).astype(int)), wall_col, 1)
    fcol = (40, 110, 210) if style == 'colour' else (60, 60, 60)
    # doors: leaf + arc
    for a, b in pl.doors:
        a, b = np.array(a), np.array(b); w = np.linalg.norm(b - a); u = (b - a) / w; n = np.array([-u[1], u[0]])
        if rng.random() < 0.5:
            n = -n
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(a + n * w).astype(int)), fcol, 2)
        a0 = math.degrees(math.atan2(u[1], u[0])); a1 = math.degrees(math.atan2(n[1], n[0]))
        da = (a1 - a0 + 180) % 360 - 180
        pts = [a + w * np.array([math.cos(math.radians(a0 + da * k / 20)), math.sin(math.radians(a0 + da * k / 20))]) for k in range(21)]
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, fcol, 1)
    if furniture:
        for (p0, p1) in pl.rooms:
            x0, y0 = p0; x1, y1 = p1
            if rng.random() < 0.7:
                w = rng.uniform(0.8, 2.0) / s; d = rng.uniform(0.5, 1.6) / s
                cx = rng.uniform(x0 + 0.5 / s + w / 2, max(x0 + 0.5 / s + w / 2 + 1, x1 - 0.5 / s - w / 2))
                cy = rng.uniform(y0 + 0.5 / s + d / 2, max(y0 + 0.5 / s + d / 2 + 1, y1 - 0.5 / s - d / 2))
                cv2.rectangle(img, (int(cx - w / 2), int(cy - d / 2)), (int(cx + w / 2), int(cy + d / 2)), fcol, 1)
                cv2.rectangle(img, (int(cx - w / 2 + 6), int(cy - d / 2 + 6)), (int(cx + w / 2 - 6), int(cy - d / 2 + 20)), fcol, 1)
            if rng.random() < 0.4:
                cv2.circle(img, (int((x0 + x1) / 2), int((y0 + y1) / 2)), int(0.4 / s), fcol, 1)
    if text:
        for (p0, p1) in pl.rooms:
            cv2.putText(img, 'ROOM', (int((p0[0] + p1[0]) / 2) - 25, int((p0[1] + p1[1]) / 2) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, fcol, 1, cv2.LINE_AA)
    if dims:
        _draw_dims(img, pl)
    return img


def dim_chain(pl):
    """Overall dimensions along the top and left (value in m, endpoints px on the dimension line)."""
    xs = sorted({round(a[0], 1) for a, b, t in pl.walls if abs(a[0] - b[0]) < 1e-6 and t >= 0.2} |
                {round(b[0], 1) for a, b, t in pl.walls if abs(a[0] - b[0]) < 1e-6 and t >= 0.2})
    ys = sorted({round(a[1], 1) for a, b, t in pl.walls if abs(a[1] - b[1]) < 1e-6 and t >= 0.2})
    out = []
    if len(xs) >= 2:
        y = min(a[1] for a, b, t in pl.walls) - 1.0 / pl.scale
        out.append(((xs[0], y), (xs[-1], y), (xs[-1] - xs[0]) * pl.scale))
    if len(ys) >= 2:
        x = min(a[0] for a, b, t in pl.walls) - 1.0 / pl.scale
        out.append(((x, ys[0]), (x, ys[-1]), (ys[-1] - ys[0]) * pl.scale))
    return out


def _draw_dims(img, pl):
    for (a, b, v) in dim_chain(pl):
        a = np.array(a); b = np.array(b)
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (0, 0, 0), 1)
        u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]])
        for q in (a, b):
            cv2.line(img, tuple(np.round(q - n * 12).astype(int)), tuple(np.round(q + n * 12).astype(int)), (0, 0, 0), 1)
            cv2.line(img, tuple(np.round(q - (u + n) * 6).astype(int)), tuple(np.round(q + (u + n) * 6).astype(int)), (0, 0, 0), 2)
        c = (a + b) / 2
        txt = f"{v:.2f}"
        if abs(u[0]) > 0.5:
            cv2.putText(img, txt, (int(c[0]) - 30, int(c[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            tmp = np.full((40, 120, 3), 255, np.uint8)
            cv2.putText(tmp, txt, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            tmp = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            y0, x0 = int(c[1]) - 60, int(c[0]) - 48
            if 0 <= y0 and y0 + 120 <= img.shape[0] and 0 <= x0 and x0 + 40 <= img.shape[1]:
                img[y0:y0 + 120, x0:x0 + 40] = np.minimum(img[y0:y0 + 120, x0:x0 + 40], tmp)


# ------------------------------------------------------------------ degradations (each returns img, 3x3 H)
def deg_rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s_ + w * c), int(h * c + w * s_)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return out, np.vstack([M, [0, 0, 1]])


def deg_perspective(img, strength=0.08, seed=0, background=True):
    """Phone photo of a sheet lying on a darker table: pinhole camera (principal point at the photo
    centre, focal ~ 1.1 x photo width) tilted by up to ~strength*250 degrees about x and y."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    W2, H2 = int(w * 1.3), int(h * 1.3)
    f = 1.1 * W2
    tilt = strength * 250.0
    rx, ry = math.radians(rng.uniform(-tilt, tilt)), math.radians(rng.uniform(-tilt, tilt))
    rz = math.radians(rng.uniform(-4, 4))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    D = f * 1.0                                   # distance so the page fills most of the frame
    K = np.array([[f, 0, W2 / 2], [0, f, H2 / 2], [0, 0, 1]])
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])        # page centre at the origin
    Hm = K @ np.c_[R[:, 0], R[:, 1], np.array([0, 0, D])] @ T
    Hm /= Hm[2, 2]
    bg = (95, 105, 115) if background else (255, 255, 255)
    out = cv2.warpPerspective(img, Hm, (W2, H2), flags=cv2.INTER_LINEAR, borderValue=bg)
    return out, Hm


def deg_lighting(img, strength=0.45, shadow=True, seed=0):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 1.0 - strength * (0.6 * xx / w + 0.4 * yy / h)
    if shadow:
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        r = 0.35 * max(w, h)
        g *= 1.0 - 0.35 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    out = np.clip(img.astype(np.float32) * g[..., None], 0, 255).astype(np.uint8)
    return out, np.eye(3)


def deg_noise(img, sigma=18, seed=0):
    rng = np.random.default_rng(seed)
    out = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return out, np.eye(3)


def deg_jpeg(img, q=25):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), np.eye(3)


def deg_lowres(img, f=0.5):
    out = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return out, np.diag([f, f, 1.0])


def deg_blur(img, k=3):
    return cv2.GaussianBlur(img, (k * 2 + 1, k * 2 + 1), 0), np.eye(3)


def transform_pts(Hm, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return q[:, :2] / q[:, 2:3]


def make_case(name, seed=0):
    """Named benchmark case -> (bgr image, plan, H from clean plan px to image px, kind)."""
    style = 'filled'; angled = False; degs = []; kind = 'scan'
    if name == 'clean_filled': pass
    elif name == 'clean_hatched': style = 'hatched'
    elif name == 'clean_hollow': style = 'hollow'
    elif name == 'colour_cad': style = 'colour'
    elif name == 'angled': angled = True
    elif name == 'thin_walls': pass
    elif name == 'scan_rot3': degs = [('rot', 3.0), ('noise', 8), ('jpeg', 60)]
    elif name == 'scan_noisy': degs = [('noise', 22), ('jpeg', 35)]
    elif name == 'scan_lowres': degs = [('lowres', 0.5), ('jpeg', 50)]
    elif name == 'photo_persp': degs = [('persp', 0.07), ('light', 0.35), ('noise', 6), ('blur', 1), ('jpeg', 70)]; kind = 'photo'
    elif name == 'photo_shadow': degs = [('rot', 1.5), ('light', 0.55), ('noise', 10), ('jpeg', 60)]; kind = 'photo'
    elif name == 'photo_rot90': degs = [('rot', 90.0), ('light', 0.3), ('jpeg', 70)]; kind = 'photo'
    else:
        raise KeyError(name)
    pl = make_plan(seed, angled=angled, thin=(name == 'thin_walls'))
    img = render(pl, style=style, seed=seed)
    Htot = np.eye(3)
    for k, v in degs:
        if k == 'rot': img, Hm = deg_rotate(img, v)
        elif k == 'persp': img, Hm = deg_perspective(img, v, seed)
        elif k == 'light': img, Hm = deg_lighting(img, v, seed=seed)
        elif k == 'noise': img, Hm = deg_noise(img, v, seed)
        elif k == 'jpeg': img, Hm = deg_jpeg(img, v)
        elif k == 'lowres': img, Hm = deg_lowres(img, v)
        elif k == 'blur': img, Hm = deg_blur(img, v)
        Htot = Hm @ Htot
    return img, pl, Htot, kind


CASES = ['clean_filled', 'clean_hatched', 'clean_hollow', 'colour_cad', 'angled', 'thin_walls',
         'scan_rot3', 'scan_noisy', 'scan_lowres', 'photo_persp', 'photo_shadow', 'photo_rot90']
