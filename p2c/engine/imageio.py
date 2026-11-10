"""Image loading / preprocessing (unicode-safe on Windows)."""
import math
import numpy as np
import cv2


def imread_gray(path):
    data = np.fromfile(path, dtype=np.uint8)          # works with Arabic/unicode paths
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def imwrite(path, img):
    ext = '.' + path.rsplit('.', 1)[-1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def rotate_bound(gray, angle_deg):
    """Rotate keeping the whole image (white background)."""
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h*s + w*c), int(h*c + w*s)
    M[0, 2] += nw/2 - w/2
    M[1, 2] += nh/2 - h/2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)


def estimate_skew(gray, max_angle=8.0):
    """Small skew of a scanned drawing (deg). Positive = rotate counter-clockwise to fix."""
    bw = (gray < 160).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(bw, 1, np.pi/720, threshold=80, minLineLength=max(40, gray.shape[1]//25), maxLineGap=3)
    if lines is None:
        return 0.0
    angs, wts = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = (a + 45) % 90 - 45                  # fold to [-45, 45)
        if abs(a) <= max_angle:
            angs.append(a); wts.append(math.hypot(x2 - x1, y2 - y1))
    if not angs:
        return 0.0
    h, e = np.histogram(angs, bins=int(max_angle*2*4), range=(-max_angle, max_angle), weights=wts)
    i = int(np.argmax(h))
    sel = [(a, w) for a, w in zip(angs, wts) if e[i] - 0.25 <= a <= e[i+1] + 0.25]
    a = sum(a*w for a, w in sel) / sum(w for _, w in sel)
    return a if abs(a) > 0.15 else 0.0


def clean_background(gray):
    """Flatten uneven scan illumination and remove border shadows."""
    bg = cv2.medianBlur(gray, 31)
    bg = cv2.max(bg, 200)
    norm = cv2.divide(gray, bg, scale=255)
    # black scanner borders -> white
    dark = (gray < 60).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    h, w = gray.shape
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches and area > 0.002 * w * h and (bw > 0.3*w or bh > 0.3*h):
            norm[lab == i] = 255
    return norm


def imread_color(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def wall_view(bgr):
    """Colour drawings (CAD exports): furniture/text are often coloured and walls neutral gray, sometimes
    only lightly filled.  Returns a gray image with coloured ink removed and the neutral ink darkened,
    or None when the drawing is not a colour drawing (plain scans are left alone)."""
    if bgr is None or bgr.ndim != 3:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat, val = hsv[..., 1], hsv[..., 2]
    ink = val < 235
    if ink.sum() < 500:
        return None
    coloured = ink & (sat > 50)
    if coloured.sum() < 0.12 * ink.sum():
        return None
    neutral = ink & (sat <= 35)
    out = np.full_like(g, 255)
    v = g[neutral].astype(np.float32)
    lo = float(np.percentile(v, 2)) if v.size else 0.0
    hi = 238.0
    out[neutral] = np.clip((v - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return out


def wall_view_path(work_path):
    import os
    return os.path.splitext(work_path)[0] + '_walls.png'


def load_wall_view(work_path, shape):
    import os
    if not work_path:
        return None
    p = wall_view_path(work_path)
    if not os.path.exists(p):
        return None
    try:
        w = imread_gray(p)
    except Exception:
        return None
    return w if w.shape == tuple(shape) else None
