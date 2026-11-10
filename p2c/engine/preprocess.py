"""Image preprocessing for scans and phone photos, with every geometric step recorded.

All geometric operations are 3x3 matrices composed into ``H`` (input image px -> working image px),
so any coordinate found on the working image can be mapped back to the original photo/scan
(``to_source``) and vice versa.  Photometric steps (lighting, noise) never move pixels.

Each step decides from measurements whether it is needed, and reports why it ran or not:
  * orientation  – 90 deg rotations: manual, or automatic from OCR text direction (optional)
  * perspective  – sheet boundary of a phone photo -> rectified top-down view; only when a clear
                   four-sided page boundary is found, otherwise nothing is warped
  * deskew       – small rotation of scanned drawings
  * lighting     – flat-field correction when the background is uneven (shadows, vignetting)
  * denoise      – non-local means only when measured noise is high
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import cv2


@dataclass
class PreprocessOptions:
    rotate: int = 0                 # manual clockwise rotation (0/90/180/270), applied first
    auto_orient: bool = False       # try 0/90/180/270 with OCR and keep the most readable (slow)
    perspective: str = 'auto'       # 'auto' | 'off' | 'manual'
    quad: list | None = None        # manual page corners (4 points, input px) for 'manual'
    deskew: bool = True
    lighting: str = 'auto'          # 'auto' | 'off' | 'on'
    denoise: str = 'auto'           # 'auto' | 'off' | 'on'
    max_side: int = 9000            # larger images are reduced (recorded) to keep memory bounded


@dataclass
class PreprocessResult:
    gray: np.ndarray                # working grayscale image
    color: np.ndarray               # working colour image (same geometry)
    H: np.ndarray                   # 3x3: input px -> working px
    size_in: tuple                  # (w, h) of the input
    steps: list = field(default_factory=list)    # [{'step', 'applied', 'reason', ...}]

    def to_source(self, pts):
        return apply_h(np.linalg.inv(self.H), pts)

    def to_work(self, pts):
        return apply_h(self.H, pts)

    def record(self):
        """JSON-able description (stored in the project)."""
        return dict(H=self.H.tolist(), size_in=list(self.size_in),
                    size_work=[int(self.gray.shape[1]), int(self.gray.shape[0])], steps=self.steps)


def apply_h(Hm, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(Hm, float).T
    return q[:, :2] / q[:, 2:3]


# ------------------------------------------------------------------ geometry helpers
def _rot90(img, k, border):
    """Rotate clockwise by k*90 deg. Returns image and 3x3 matrix."""
    k %= 4
    h, w = img.shape[:2]
    if k == 0:
        return img, np.eye(3)
    codes = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
    out = cv2.rotate(img, codes[k])
    if k == 1:   # (x, y) -> (h-1-y, x)
        M = np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], float)
    elif k == 2:
        M = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], float)
    else:        # (x, y) -> (y, w-1-x)
        M = np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], float)
    return out, M


def rotate_bound(img, angle_deg, border=255):
    """Rotate keeping the whole image. Returns image and 3x3 matrix."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    bv = (border,) * 3 if img.ndim == 3 else border
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=bv)
    return out, np.vstack([M, [0, 0, 1]])


# ------------------------------------------------------------------ perspective
def _order_quad(q):
    q = np.asarray(q, np.float32).reshape(4, 2)
    s = q.sum(1); d = np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def find_page_quad(gray):
    """Four corners of a sheet of paper that is clearly brighter than its surroundings, or None.
    Returns (quad, info).  Conservative: needs a convex 4-gon covering 25..97% of the image with
    high contrast between inside and outside, and not simply the image frame."""
    h, w = gray.shape
    f = 900.0 / max(h, w)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray.copy()
    f = min(f, 1.0)
    sh, sw = small.shape
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    # the drawing's own lines must not split the page: close dark strokes first
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    thr, bw = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    info = dict(found=False)
    if not cnts:
        info['reason'] = 'no bright region'; return None, info
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c) / float(sh * sw)
    if not 0.25 <= area <= 0.97:
        info['reason'] = f'page area {area:.2f} outside 0.25..0.97'; return None, info
    peri = cv2.arcLength(c, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.04):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            quad = ap.reshape(4, 2).astype(np.float32); break
    if quad is None:
        info['reason'] = 'boundary is not a clean quadrilateral'; return None, info
    quad = _order_quad(quad)
    # fit quality: contour points must lie close to the quad edges
    poly_area = cv2.contourArea(quad) / float(sh * sw)
    if abs(poly_area - area) > 0.04:
        info['reason'] = 'quadrilateral does not match the page outline'; return None, info
    # contrast inside vs. outside
    mask = np.zeros_like(small); cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) & ~mask
    inner = cv2.erode(mask, np.ones((25, 25), np.uint8))
    if ring.sum() == 0 or inner.sum() == 0:
        info['reason'] = 'page touches the image border on all sides'; return None, info
    cin = float(np.median(blur[inner > 0])); cout = float(np.median(blur[ring > 0]))
    info.update(contrast=cin - cout, area=area)
    if cin - cout < 25:
        info['reason'] = f'low page/background contrast ({cin - cout:.0f})'; return None, info
    corners_img = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32)
    if np.max(np.linalg.norm(quad - corners_img, axis=1)) < 0.015 * max(sh, sw):
        info['reason'] = 'page already fills the image'; return None, info
    info['found'] = True
    return quad / f, info


def page_aspect(quad, image_shape):
    """Real width/height ratio of a rectangle seen in perspective (Zhang & He, 2007), with the
    principal point at the image centre.  Falls back to edge lengths when the view is near-affine."""
    q = _order_quad(quad).astype(float)
    m1, m2, m4, m3 = [np.array([p[0], p[1], 1.0]) for p in q]          # tl, tr, br, bl -> m1 m2 m4 m3
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2 = k2 * m2 - m1; n3 = k3 * m3 - m1
    h, w = image_shape[:2]; u0, v0 = w / 2.0, h / 2.0
    edge_ar = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / \
              max(np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]), 1e-9)
    if abs(n2[2]) < 1e-9 or abs(n3[2]) < 1e-9:
        return edge_ar
    f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0) +
           (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
    if not np.isfinite(f2) or f2 <= 0 or not (0.3 * max(h, w)) ** 2 < f2 < (20 * max(h, w)) ** 2:
        return edge_ar
    A = np.array([[math.sqrt(f2), 0, u0], [0, math.sqrt(f2), v0], [0, 0, 1.0]])
    Ai = np.linalg.inv(A); B = Ai.T @ Ai
    ar = math.sqrt(max(n2 @ B @ n2, 1e-12) / max(n3 @ B @ n3, 1e-12))
    return ar if 0.2 < ar < 5 else edge_ar


def rectify(img, quad, border=255):
    """Warp the page quad to a rectangle with the recovered real aspect ratio (resolution of the
    longer side kept)."""
    q = _order_quad(quad)
    wt = np.linalg.norm(q[1] - q[0]); wb = np.linalg.norm(q[2] - q[3])
    hl = np.linalg.norm(q[3] - q[0]); hr = np.linalg.norm(q[2] - q[1])
    ar = page_aspect(q, img.shape)
    W0, H0 = max(wt, wb), max(hl, hr)
    if W0 / ar >= H0:
        W, H = int(round(W0)), int(round(W0 / ar))
    else:
        W, H = int(round(H0 * ar)), int(round(H0))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    bv = (border,) * 3 if img.ndim == 3 else border
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=bv), M


# ------------------------------------------------------------------ photometric
def background_map(gray, max_stroke_px=None):
    """Smooth paper-brightness map: grey closing removes dark strokes narrower than the kernel."""
    h, w = gray.shape
    f = min(1.0, 600.0 / max(h, w))
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    k = int(max(15, (max_stroke_px or 0.04 * max(h, w)) * f)) | 1
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def lighting_unevenness(gray):
    """Relative spread of the paper brightness (0 = perfectly flat)."""
    bg = background_map(gray).astype(np.float32)
    lo, hi = np.percentile(bg, [5, 95])
    return float((hi - lo) / max(hi, 1.0))


def flatten_lighting(gray):
    bg = background_map(gray).astype(np.float32)
    out = gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8), bg


def noise_sigma(gray):
    """Noise level of the paper: median local standard deviation over flat paper areas (windows that
    contain no drawing strokes).  Works for correlated (JPEG) noise, unlike Laplacian estimators."""
    g = gray.astype(np.float32)
    f = min(1.0, 1500.0 / max(g.shape))
    if f < 1:                                   # sample a reduced-size centre region, not a resize (keeps noise)
        h, w = g.shape; ch, cw = int(h * f), int(w * f)
        g = g[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    m = cv2.blur(g, (7, 7)); m2 = cv2.blur(g * g, (7, 7))
    sd = np.sqrt(np.maximum(m2 - m * m, 0))
    lo = cv2.erode(g, np.ones((7, 7), np.uint8))
    paper_level = np.percentile(g, 75)
    flat = (lo > paper_level - 60) & (m > paper_level - 30)
    return float(np.median(sd[flat])) if flat.sum() > 500 else 0.0


def denoise(gray, sigma):
    h = float(np.clip(0.9 * sigma, 5, 25))
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def remove_scan_borders(gray):
    """Large dark regions that touch the image edge (scanner lid, table around a photo) -> white."""
    dark = (gray < 60).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    h, w = gray.shape
    out = gray; k = 0
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches and area > 0.002 * w * h and (bw > 0.3 * w or bh > 0.3 * h) and area > 0.5 * bw * bh * 0.2:
            fill = area / float(bw * bh)
            if fill > 0.6 or min(bw, bh) < 0.04 * min(w, h):      # a solid band, not the drawing's frame line
                if out is gray:
                    out = gray.copy()
                out[lab == i] = 255; k += 1
    return out, k


# ------------------------------------------------------------------ orientation
def detect_orientation(gray):
    """Clockwise quarter turns that make most text upright, from OCR on a reduced copy.
    Returns (k, scores) or (0, None) when OCR is unavailable or undecided."""
    try:
        from .dims import _ocr, ocr_available
        if not ocr_available():
            return 0, None
        eng = _ocr()
    except Exception:
        return 0, None
    f = min(1.0, 1400.0 / max(gray.shape))
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    scores = []
    for k in range(4):
        img, _ = _rot90(small, k, 255)
        res, _ = eng(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), use_cls=False)
        sc = 0.0
        for box, txt, conf in res or []:
            b = np.array(box, float)
            if np.linalg.norm(b[1] - b[0]) < np.linalg.norm(b[3] - b[0]):
                continue
            t = txt.strip()
            if len(t) >= 2 and float(conf) > 0.75:
                sc += float(conf) * min(len(t), 8)
        scores.append(sc)
    best = int(np.argmax(scores))
    if best != 0 and scores[best] > 1.5 * max(scores[0], 1.0) and scores[best] > 12:
        return best, scores
    return 0, scores


# ------------------------------------------------------------------ main entry
def preprocess(color, opts: PreprocessOptions | None = None, progress=None):
    """color: BGR uint8 (EXIF orientation already applied by the decoder)."""
    o = opts or PreprocessOptions()
    say = progress or (lambda f: None)
    if color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    h0, w0 = color.shape[:2]
    H = np.eye(3)
    steps = []
    img = color
    # size limit
    if max(h0, w0) > o.max_side:
        f = o.max_side / max(h0, w0)
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        H = np.diag([f, f, 1.0]) @ H
        steps.append(dict(step='downscale', applied=True, factor=f, reason=f'longest side > {o.max_side}px'))
    # manual rotation
    if o.rotate % 360:
        img, M = _rot90(img, o.rotate // 90, 255); H = M @ H
        steps.append(dict(step='rotate', applied=True, degrees=o.rotate, reason='requested'))
    say(0.1)
    # perspective (page detection)
    if o.perspective != 'off':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        quad, info = (None, dict(found=False, reason='no quad given'))
        if o.perspective == 'manual' and o.quad:
            quad = apply_h(H, o.quad).astype(np.float32); info = dict(found=True, reason='manual corners')
        else:
            quad, info = find_page_quad(gray)
        if quad is not None:
            img, M = rectify(img, quad, 255); H = M @ H
            steps.append(dict(step='perspective', applied=True, quad=quad.tolist(),
                              reason=info.get('reason', 'page boundary found'),
                              contrast=info.get('contrast')))
        else:
            steps.append(dict(step='perspective', applied=False, reason=info.get('reason', '')))
    say(0.3)
    # automatic orientation (after rectification, before deskew)
    if o.auto_orient:
        k, sc = detect_orientation(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if k:
            img, M = _rot90(img, k, 255); H = M @ H
        steps.append(dict(step='orientation', applied=bool(k), quarter_turns=k, scores=sc,
                          reason='text direction' if k else 'text already upright or undecided'))
    say(0.45)
    # deskew
    if o.deskew:
        from .imageio import estimate_skew
        sk = estimate_skew(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if sk:
            img, M = rotate_bound(img, sk, 255); H = M @ H
        steps.append(dict(step='deskew', applied=bool(sk), degrees=float(sk),
                          reason='dominant lines tilted' if sk else 'no measurable tilt'))
    say(0.6)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # lighting
    if o.lighting != 'off':
        une = lighting_unevenness(gray)
        do = o.lighting == 'on' or une > 0.08
        if do:
            gray, bg = flatten_lighting(gray)
            img = np.clip(img.astype(np.float32) / np.maximum(bg[..., None].astype(np.float32), 1) * 255,
                          0, 255).astype(np.uint8)
        steps.append(dict(step='lighting', applied=bool(do), unevenness=une,
                          reason='uneven background' if do else 'background already even'))
        # paper level: a grey (yellowed / dim) sheet is brought to white with a global gain
        paper = float(np.percentile(gray, 90))
        if 120 < paper < 235:
            gain = 250.0 / paper
            gray = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            steps.append(dict(step='paper_level', applied=True, gain=gain, reason=f'paper grey level {paper:.0f}'))
        # black scanner borders touching the image edge -> white
        gray, nb = remove_scan_borders(gray)
        if nb:
            steps.append(dict(step='borders', applied=True, n=nb, reason='dark scanner border at the image edge'))
    say(0.75)
    # noise
    if o.denoise != 'off':
        sg = noise_sigma(gray)
        do = o.denoise == 'on' or sg > 6.0
        if do:
            gray = denoise(gray, sg)
        steps.append(dict(step='denoise', applied=bool(do), sigma=sg,
                          reason='noisy image' if do else 'noise low'))
    say(1.0)
    return PreprocessResult(gray=gray, color=img, H=H, size_in=(w0, h0), steps=steps)
