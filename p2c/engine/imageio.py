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
