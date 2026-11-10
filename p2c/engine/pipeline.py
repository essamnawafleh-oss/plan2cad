"""Run the full automatic recognition on a project."""
from __future__ import annotations

import math
import numpy as np
import cv2

from . import detect as D
from ..geometry import wall_polygon, door_geometry, window_cut, rings
from shapely.ops import unary_union

DEFAULTS = dict(
    threshold=185,            # ink threshold (0-255)
    wall_dark=150,            # wall lines must be at least this dark (median gray)
    wall_t_min=0.09,          # m
    wall_t_max=0.40,          # m
    wall_min_len=0.45,        # m
    wall_join=0.45,           # m  max gap closed at corners / T-junctions
    wall_thicknesses=[0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35],
    column_dark=105,
    door_min=0.6, door_max=2.1, door_score=0.45,
    window_min_wall=0.2, window_min_len=0.5,
    do_walls=True, do_columns=True, do_doors=True, do_windows=True, do_furniture=False,
)


def run(project, gray, params=None, progress=None):
    """Fill *project* from the working image *gray* (uint8)."""
    p = dict(DEFAULTS); p.update(params or {})
    say = progress or (lambda f, msg='': None)
    say(0.02, 'binarize')
    # ---- 1. understand the drawing first: directions + dimensions ----
    quick = D.trace_lines(gray, p['threshold'], up=1)
    project.angles = D.dominant_angles(quick)
    project.dims, project.texts_raw = [], []
    clean = gray
    if p.get('read_dims', True):
        from . import dims as DM
        if DM.ocr_available():
            say(0.04, 'dims')
            info = DM.analyse(gray, project.angles, progress=lambda f: say(0.04 + 0.30 * f, 'dims'))
            project.dims_info = {k: info.get(k) for k in ('scale', 'n_texts', 'n_numbers', 'n_dims', 'agree', 'units', 'calib')}
            if info['scale'] and info['n_dims'] >= 5 and not project.scale_calibrated:
                project.scale = info['scale']
                project.scale_calibrated = True
                project.scale_source = 'dims'
            from ..model import Dim
            project.dims = [Dim(tuple(map(float, d['seg'][0])), tuple(map(float, d['seg'][1])), d['text'],
                                float(d['value']), conf=round(float(d.get('conf', 1.0)), 3),
                                unit=d.get('unit', 'm'), status=d.get('status', ''))
                            for d in info['dims'] if d.get('status', 'used') == 'used' or 'unit' not in d.get('status', '')]
            project.texts_raw = [(t['text'], [tuple(map(float, q)) for q in t['box']]) for t in info['texts']]
            mask = DM.clutter_mask(gray, info['texts'], info['dims'])
            clean = gray.copy(); clean[mask > 0] = 255
    s = project.scale
    # ---- 2. trace the remaining drawing ----
    say(0.36, 'trace')
    polys = D.trace_lines(clean, p['threshold'], up=p.get('trace_up', 3), progress=lambda f: say(0.36 + 0.2*f, 'trace'))
    project.trace = [[tuple(map(float, q)) for q in pl] for pl in D.trace_lines(gray, p['threshold'], up=2)]
    say(0.58, 'angles')
    gray_full, gray = gray, clean
    # colour CAD drawings: walls are found on a view without the coloured furniture / text
    from . import imageio as _io
    wv = p.get('wall_view')
    if wv is None:
        wv = _io.load_wall_view(getattr(project, 'image_path', None), gray_full.shape)
    if wv is not None and wv.shape == gray_full.shape:
        wv = wv.copy(); wv[(clean == 255) & (gray_full != 255)] = 255          # text / dimension clutter removed
        gray = wv
    nn_img = gray if wv is not None else gray_full
    # no calibration and no dimensions: estimate the scale from the doors the network sees
    if not project.scale_calibrated and p.get('use_nn', True) and p.get('auto_scale', True):
        from . import neural as NN
        if NN.available():
            say(0.37, 'scale')
            se = NN.estimate_scale(nn_img, progress=lambda f: say(0.37 + 0.15 * f, 'scale'))
            if se and 0.002 < se < 0.2:
                project.scale = se; project.scale_source = 'doors'
                s = se
    if p['do_walls']:
        say(0.55, 'walls')
        nn_prob = None
        if p.get('use_nn', True):
            from . import neural as NN
            if NN.available():
                say(0.56, 'nn')
                nn_prob = NN.segment(nn_img, s, progress=lambda f: say(0.56 + 0.06 * f, 'nn'))
        project._nn_prob = nn_prob
        solid, core = D.detect_solid_walls(gray, s, project.angles, p) if p.get('do_solid', True) else ([], None)
        if nn_prob is not None:
            nn_walls = NN.walls_from_prob(nn_prob, nn_img, s, project.angles, p)
            # the network decides where walls are; rule walls only add what it clearly missed
            from ..geometry import wall_polygon as _wp
            from shapely.ops import unary_union as _uu
            NU = _uu([_wp(w, s) for w in nn_walls]).buffer(2) if nn_walls else None
            extra = solid if p.get('nn_keep_rules', True) else []
            if NU is not None:
                extra = [w for w in extra if _wp(w, s).difference(NU).area > 0.6 * _wp(w, s).area]
            solid, lined = nn_walls + extra, []
        else:
            lined = D.detect_walls_combined(polys, gray, s, project.angles, p)
        if solid:
            from ..geometry import wall_polygon
            from shapely.ops import unary_union as _uu
            SU = _uu([wall_polygon(w, s) for w in solid]).buffer(2)
            lined = [w for w in lined if wall_polygon(w, s).intersection(SU).area < 0.5 * wall_polygon(w, s).area]
        walls = solid + lined
        project.corrections = []
        ev = p.get('evidence', True)
        pw = None
        if nn_prob is not None:
            pw = np.maximum(nn_prob[1], nn_prob[3])
        if ev:                                # measure every proposed wall on the drawing itself
            from . import walls as WG
            say(0.63, 'walls')
            walls = WG.refine_walls(walls, gray, s, pw, log=project.corrections)
            project._dbg_after_refine = [(list(w.pts), w.t) for w in walls]

        def _join(ws):
            if ev:
                return WG.join_with_evidence(ws, gray, s, p['wall_join'] + 0.15, prob_wall=pw, log=project.corrections)
            return D.join_walls(ws, s, p['wall_join'] + 0.15)
        if nn_prob is not None or ev:
            walls = _join(walls)              # close junction gaps before judging connectivity
        walls = D.prune_isolated_walls(walls, s)
        H, W = gray.shape[:2]; m = max(12, 0.012 * max(H, W))
        walls = [w for w in walls if not all(x < m or y < m or x > W - m or y > H - m for x, y in w.pts)]   # page border
        m2 = 0.05 * min(H, W)
        def _edge_line(w):
            (x0, y0), (x1, y1) = w.pts[0], w.pts[-1]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 0.25 * min(H, W):
                return False
            vert = abs(x1 - x0) < 0.03 * L; hor = abs(y1 - y0) < 0.03 * L
            return (vert and (max(x0, x1) < m2 or min(x0, x1) > W - m2)) or (hor and (max(y0, y1) < m2 or min(y0, y1) > H - m2))
        def _attached(w):            # a real outer wall has other walls meeting it; a page frame line does not
            from shapely.geometry import LineString as _LS
            ln = _LS(w.pts); tol = 0.35 / s
            return sum(1 for o in walls if o is not w and ln.distance(_LS(o.pts)) < tol) >= 2
        walls = [w for w in walls if not (_edge_line(w) and not _attached(w))]
        project.walls = D.keep_building(_join(walls), s)
        if p.get('complete_from_dims', True) and project.dims:
            from .complete import complete_walls
            say(0.66, 'complete')
            n0 = len(project.walls)
            project.completed = complete_walls(project, gray_full, p)
            for w in project.walls[n0:]:
                w.origin = 'fixed'; w.conf = min(getattr(w, 'conf', 1.0), 0.6)
                project.corrections.append(dict(action='complete_wall', reason='dimension end without a wall',
                                                pts=[list(q) for q in w.pts], t=w.t))
            project.walls = _join(project.walls)
        if nn_prob is not None or ev:
            from . import neural as NN
            NN.trim_overshoots(project.walls, s)
        _apply_wall_labels(project)
    if p['do_columns']:
        say(0.68, 'columns')
        cols = D.detect_columns(gray, s, p)
        if project.walls:
            from ..geometry import wall_polygon
            from shapely.ops import unary_union as _uu
            from shapely.geometry import Point as _P
            WU = _uu([wall_polygon(w, s) for w in project.walls if w.t >= 0.2])
            cols = [c for c in cols if not WU.contains(_P(c.x, c.y))]
        H, W = gray.shape[:2]; mb = 0.04 * min(H, W)
        cols = [c for c in cols if mb < c.x < W - mb and mb < c.y < H - mb]          # page-edge blobs
        g_dark = gray < 110
        for c in cols:                                                 # confidence = how solid the blob is
            r = max(1, int(c.s / s / 2))
            y0, x0 = int(c.y), int(c.x)
            patch = g_dark[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1]
            c.conf = round(float(patch.mean()) if patch.size else 0.0, 3)
            c.origin = 'auto' if c.conf >= 0.6 else 'uncertain'
        project.columns = cols
    if p['do_doors'] and project.walls:
        say(0.74, 'doors')
        project.doors = D.detect_doors(project.walls, gray, s, p)
        if getattr(project, '_nn_prob', None) is not None:
            from . import neural as NN
            nd, nw = NN.openings_from_prob(project._nn_prob, project.walls, s)
            nd = NN.orient_doors(nd, gray_full, s)
            keep = [d for d in project.doors if all(math.dist(((d.p0[0]+d.p1[0])/2, (d.p0[1]+d.p1[1])/2),
                    ((e.p0[0]+e.p1[0])/2, (e.p0[1]+e.p1[1])/2)) > 0.4 / s for e in nd)]
            for d in keep:                                             # rule doors next to network ones
                d.conf, d.origin = 0.55, 'auto'
            project.doors = nd + keep
            project._nn_windows = nw
    if p['do_windows'] and project.walls:
        say(0.82, 'windows')
        project.windows = D.detect_windows(project.walls, project.columns, gray, s, p)
        nw = getattr(project, '_nn_windows', None)
        if nw is not None:
            project.windows = nw
    if p.get('fit_dims', True) and getattr(project, 'dims', None) and project.walls:
        from .fit import fit_to_dimensions
        say(0.88, 'fit')
        fit_to_dimensions(project)
    if p.get('chain_windows', True) and getattr(project, 'dims', None) and project.walls:
        from .chains import windows_from_chains
        say(0.89, 'chains')
        n0 = len(project.windows)
        windows_from_chains(project, gray_full)
        for w in project.windows[n0:]:
            w.origin = 'fixed'; w.conf = 0.7
            project.corrections.append(dict(action='window_from_chain', p0=list(w.p0), p1=list(w.p1),
                                            reason='facade dimension chain'))
        if p.get('fit_dims', True):
            from .fit import fit_to_dimensions
            fit_to_dimensions(project)
    if p['do_furniture']:
        say(0.90, 'furniture')
        mask = np.zeros_like(gray)
        geo = [wall_polygon(w, s) for w in project.walls]
        geo += [window_cut(w, s) for w in project.windows]
        for d in project.doors:
            lv, arcs = door_geometry(d, s)
            from shapely.geometry import LineString, Polygon
            geo += [Polygon(l).buffer(2) for l in lv] + [LineString(a).buffer(2.5) for a in arcs]
        U = unary_union([g.buffer(2.5) for g in geo if g is not None]) if geo else None
        for gpoly in (getattr(U, 'geoms', [U]) if U is not None else []):
            if gpoly.geom_type != 'Polygon':
                continue
            cv2.fillPoly(mask, [np.round(np.array(gpoly.exterior.coords)).astype(np.int32)], 255)
            for r in gpoly.interiors:
                cv2.fillPoly(mask, [np.round(np.array(r.coords)).astype(np.int32)], 0)
        for c in project.columns:
            sp = c.s / s
            cv2.circle(mask, (int(c.x), int(c.y)), int(sp * 0.8), 255, -1)
        project.furniture = D.detect_furniture(gray, s, mask, p)
    for kind in ('walls', 'doors', 'windows', 'columns'):
        for o in getattr(project, kind):
            if getattr(o, 'origin', '') == '':
                o.origin = 'auto'; o.conf = min(getattr(o, 'conf', 1.0), 0.8)
    # ---- geometric validation: safe fixes are applied (and logged), everything else is reported
    from . import validate as VAL
    say(0.95, 'validate')
    if not hasattr(project, 'corrections') or project.corrections is None:
        project.corrections = []
    VAL.auto_fix(project, log=project.corrections)
    project.validation = VAL.validate(project)
    project._nn_prob = None; project._nn_windows = None        # free the probability maps (hundreds of MB)
    say(1.0, 'done')
    return project


def estimate_scale_from_doors(project):
    """If the user did not calibrate: assume the median door is 0.90 m wide."""
    ws = [np.hypot(d.p1[0]-d.p0[0], d.p1[1]-d.p0[1]) / (2 if d.double else 1) for d in project.doors]
    if len(ws) < 3:
        return None
    return 0.90 / float(np.median(ws))


def _apply_wall_labels(project):
    """Numbers like 0.1 / 0.2 written on a wall give its thickness."""
    from ..geometry import wall_polygon
    from shapely.geometry import Point
    from . import dims as DM
    s = project.scale
    polys = [wall_polygon(w, s) for w in project.walls]
    for txt, box in getattr(project, 'texts_raw', []):
        v = DM.parse_value(txt)
        if v is None or not (0.05 <= v <= 0.45):
            continue
        c = Point(np.mean([q[0] for q in box]), np.mean([q[1] for q in box]))
        best = None
        for w, pg in zip(project.walls, polys):
            d = pg.distance(c)
            if d <= 0.25 / s and (best is None or d < best[0]):
                best = (d, w)
        if best and abs(best[1].t - v) <= 0.12:
            best[1].t = v
