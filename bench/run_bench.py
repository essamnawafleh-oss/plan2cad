"""Repeatable benchmark: synthetic plans with exact ground truth (+ optional real plans).

    python bench/run_bench.py --code . --out bench_results/current.json
    python bench/run_bench.py --code /path/to/old/checkout --out bench_results/baseline.json
    python bench/report.py bench_results/baseline.json bench_results/current.json

Each case runs in its own subprocess so peak memory is measured per case.  Predictions are
mapped back into the clean synthetic frame through the recorded transforms, so the same metric
code applies whatever preprocessing (rotation, rectification, resizing) an implementation does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


# ------------------------------------------------------------------ import step (old vs new code)
def import_like_app(path, code_has_workflow):
    """Returns gray work image, optional wall view, and 3x3 H mapping INPUT image px -> WORK px."""
    if code_has_workflow:
        from p2c.engine import workflow
        o = workflow.ImportOptions()
        for k, v in json.loads(os.environ.get('P2C_PP', '{}')).items():      # preprocessing configuration
            setattr(o, k, v)
        r = workflow.import_image(path, o)
        return r.gray, r.wall_view, r.H_in_to_work, r
    from p2c.engine import imageio            # baseline: replicate MainWindow.load_image (deskew + clean)
    gray = imageio.imread_gray(path)
    try:
        wv = imageio.wall_view(imageio.imread_color(path))
    except Exception:
        wv = None
    Htot = np.eye(3)
    sk = imageio.estimate_skew(gray)
    if sk:
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), sk, 1.0)
        c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * s + w * c), int(h * c + w * s)
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        gray = imageio.rotate_bound(gray, sk)
        if wv is not None:
            wv = imageio.rotate_bound(wv, sk)
        Htot = np.vstack([M, [0, 0, 1]]) @ Htot
    gray = imageio.clean_background(gray)
    return gray, wv, Htot, None


def local_scale(Hm, pt):
    """Linear scale factor of a homography around pt."""
    from bench.synth import transform_pts
    p = np.array(pt, float)
    q = transform_pts(Hm, [p, p + [10, 0], p + [0, 10]])
    a, b = q[1] - q[0], q[2] - q[0]
    return math.sqrt(abs(a[0] * b[1] - a[1] * b[0])) / 10.0


# ------------------------------------------------------------------ worker
def run_case(case, seed, calibrated, out_json):
    from bench import synth
    from bench.synth import transform_pts
    import resource
    from p2c.model import Project
    from p2c.engine import pipeline
    from p2c import geometry as G
    has_wf = os.path.exists(os.path.join(os.path.dirname(pipeline.__file__), 'workflow.py'))
    if case.startswith('real:'):
        img_path, gt_walls, gscale = _real_case(case[5:])
        Hc = np.eye(3); kind = 'real'; pl = None
    else:
        img, pl, Hc, kind = synth.make_case(case, seed)
        tmpd = tempfile.mkdtemp(prefix='p2cbench_')
        img_path = os.path.join(tmpd, case + '.jpg')
        cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        gt_walls = [(np.array(a), np.array(b), t) for a, b, t in pl.walls]
        gscale = pl.scale
    t0 = time.time()
    gray, wv, Hw, imp = import_like_app(img_path, has_wf)
    t_import = time.time() - t0
    H_clean_to_work = Hw @ Hc
    centre = np.array([gray.shape[1] / 2, gray.shape[0] / 2])
    Hinv = np.linalg.inv(H_clean_to_work)
    s_true = gscale / local_scale(H_clean_to_work, transform_pts(Hinv, [centre])[0])
    pr = Project(image_path=img_path, scale=s_true if calibrated else 0.03, scale_calibrated=calibrated)
    params = dict(read_dims=True)
    if wv is not None:
        params['wall_view'] = wv
    # time the network separately when present
    t_nn = [0.0]
    try:
        from p2c.engine import neural as NN
        _seg = NN.segment

        def timed(*a, **k):
            t = time.time(); r = _seg(*a, **k); t_nn[0] += time.time() - t; return r
        NN.segment = timed
    except Exception:
        pass
    t0 = time.time()
    err = None
    try:
        pipeline.run(pr, gray, params)
    except Exception as e:                       # a crash is a result too
        import traceback
        err = traceback.format_exc()
    t_run = time.time() - t0
    t0 = time.time()
    dxf_ok = None
    try:
        from p2c.dxf_export import export_dxf
        dxf_path = os.path.join(tempfile.mkdtemp(prefix='p2cdxf_'), 'out.dxf')
        export_dxf(pr, dxf_path, (gray.shape[1], gray.shape[0]), None, include_scan=False)
        import ezdxf
        d = ezdxf.readfile(dxf_path)
        dxf_ok = len(list(d.modelspace())) > 0
    except Exception:
        dxf_ok = False
    t_dxf = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # MB on Linux
    res = dict(case=case, seed=seed, kind=kind, calibrated=calibrated, error=err,
               t_import=t_import, t_run=t_run, t_nn=t_nn[0], t_dxf=t_dxf, peak_mb=peak, dxf_ok=dxf_ok,
               scale_true=s_true, scale_used=pr.scale, scale_err=abs(pr.scale / s_true - 1.0),
               n_walls=len(pr.walls), n_doors=len(pr.doors), n_windows=len(pr.windows))
    if err is None:
        # predictions -> clean frame
        M = np.linalg.inv(H_clean_to_work)
        pw = []
        for w in pr.walls:
            if len(w.pts) < 2:
                continue
            q = transform_pts(M, w.pts)
            loc = local_scale(M, w.pts[0])
            for a, b in zip(q[:-1], q[1:]):
                pw.append((a, b, w.t, w.t / pr.scale * loc))       # thickness in clean px
        res.update(wall_metrics(gt_walls, pw, gscale, (pl.doors if pl else []),
                                shape=(pl.size[1], pl.size[0]) if pl else gray.shape))
        if pl is not None:
            dpts = [transform_pts(M, [((d.p0[0] + d.p1[0]) / 2, (d.p0[1] + d.p1[1]) / 2)])[0] for d in pr.doors]
            wpts = [transform_pts(M, [((d.p0[0] + d.p1[0]) / 2, (d.p0[1] + d.p1[1]) / 2)])[0] for d in pr.windows]
            res.update(point_pr('door', [((np.array(a) + np.array(b)) / 2) for a, b in pl.doors], dpts, 0.45 / gscale))
            res.update(point_pr('window', [((np.array(a) + np.array(b)) / 2) for a, b, t in pl.windows], wpts, 0.5 / gscale))
    with open(out_json, 'w') as f:
        json.dump(res, f)


def _real_case(name):
    """Real plans with hand-traced wall centre lines (available on the dev machine only)."""
    root = os.environ.get('P2C_REAL_DIR', '')
    if name == 'plan1':
        img = os.path.join(root, 'work.png')
        ns = {}
        cwd = os.getcwd(); os.chdir(root)
        try:
            exec(open(os.path.join(root, 'wall_data.py')).read(), ns)
        finally:
            os.chdir(cwd)
        gt = []
        for t, pts in ns['W']:
            for a, b in zip(pts[:-1], pts[1:]):
                gt.append((np.array(a, float), np.array(b, float), t))
        return img, gt, 0.03
    raise KeyError(name)


def _sample(a, b, step=2.0):
    L = np.linalg.norm(b - a)
    n = max(2, int(L / step))
    return a + (b - a) * np.linspace(0, 1, n)[:, None], L


def _seg_dist(P, a, b):
    ab = b - a; L2 = max(ab @ ab, 1e-9)
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def wall_metrics(gt, pred, gscale, doors, shape):
    """gt: (a, b, t_m) clean px ; pred: (a, b, t_m, t_px_clean)."""
    tol_m = 0.15
    door_segs = [(np.array(a), np.array(b)) for a, b in doors]

    def in_door(P):
        m = np.zeros(len(P), bool)
        for a, b in door_segs:
            m |= _seg_dist(P, a, b) < 0.25 / gscale
        return m
    # recall over GT length
    cov_len = tot_len = 0.0
    missing = 0
    t_err = []
    for a, b, t in gt:
        P, L = _sample(a, b)
        keep = ~in_door(P)
        if keep.sum() < 2:
            continue
        P = P[keep]
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        tol = max(0.6 * t, tol_m) / gscale
        best = np.full(len(P), np.inf); bt = np.full(len(P), np.nan)
        for pa, pb, pt, ptp in pred:
            v = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
            if abs(u @ v) < math.cos(math.radians(12)):
                continue
            d = _seg_dist(P, pa, pb)
            upd = d < best
            best[upd] = d[upd]; bt[upd] = ptp * gscale
        hit = best < tol
        frac = hit.mean()
        cov_len += frac * L * keep.mean(); tot_len += L * keep.mean()
        if frac < 0.5:
            missing += 1
        t_err += list(np.abs(bt[hit] - t))
    recall = cov_len / max(tot_len, 1e-9)
    # precision over predicted length
    ok_len = p_len = 0.0
    false = 0
    for pa, pb, pt, ptp in pred:
        P, L = _sample(pa, pb)
        v = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
        best = np.full(len(P), np.inf)
        for a, b, t in gt:
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            if abs(u @ v) < math.cos(math.radians(12)):
                continue
            tol = max(0.6 * t, tol_m) / gscale
            d = _seg_dist(P, a, b) / tol
            best = np.minimum(best, d)
        good = (best < 1.0) & ~in_door(P)
        ok_len += good.mean() * L; p_len += L
        if good.mean() < 0.5:
            false += 1
    precision = ok_len / max(p_len, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # endpoints and junctions
    ends = [p for a, b, t in gt for p in (a, b)]
    pends = [p for pa, pb, pt, ptp in pred for p in (pa, pb)]
    e_err = []
    for p in ends:
        if pends:
            e_err.append(min(1.0, min(np.linalg.norm(p - q) for q in pends) * gscale))
    junction = []
    for p in ends:
        n_gt = sum(1 for a, b, t in gt if _seg_dist(p[None], a, b)[0] < 2.0)
        if n_gt < 2:
            continue
        n_pr = sum(1 for pa, pb, pt, ptp in pred if _seg_dist(p[None], pa, pb)[0] < max(ptp / 2 + 0.1 / gscale, 3))
        junction.append(n_pr >= 2)
    # pixel IoU
    H, W = shape[:2]
    gm = np.zeros((H, W), np.uint8)
    for a, b, t in gt:
        _fill_wall(gm, a, b, t / gscale)
    for a, b in door_segs:
        _fill_wall(gm, a, b, 0.6 / gscale, ext=False, val=0)
    pm = np.zeros((H, W), np.uint8)
    for pa, pb, pt, ptp in pred:
        _fill_wall(pm, pa, pb, ptp)
    inter = np.logical_and(gm > 0, pm > 0).sum(); uni = np.logical_or(gm > 0, pm > 0).sum()
    return dict(wall_recall=recall, wall_precision=precision, wall_f1=f1, wall_iou=inter / max(uni, 1),
                thick_err_m=float(np.mean(t_err)) if t_err else None,
                endpoint_err_m=float(np.mean(e_err)) if e_err else None,
                connectivity=float(np.mean(junction)) if junction else None,
                missing_walls=missing, false_walls=false, n_gt_walls=len(gt), n_pred_segs=len(pred))


def _fill_wall(m, a, b, tp, ext=True, val=255):
    a, b = np.array(a, float), np.array(b, float)
    L = np.linalg.norm(b - a)
    if L < 1e-6:
        return
    u = (b - a) / L; n = np.array([-u[1], u[0]]); e = u * tp / 2 if ext else 0
    poly = np.array([a - e - n * tp / 2, b + e - n * tp / 2, b + e + n * tp / 2, a - e + n * tp / 2])
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], val)


def point_pr(name, gt, pred, tol):
    used = set(); tp = 0
    for g in gt:
        best = None
        for i, q in enumerate(pred):
            if i in used:
                continue
            d = np.linalg.norm(np.asarray(g) - q)
            if d < tol and (best is None or d < best[0]):
                best = (d, i)
        if best:
            used.add(best[1]); tp += 1
    p = tp / max(len(pred), 1); r = tp / max(len(gt), 1)
    return {f'{name}_precision': p, f'{name}_recall': r, f'{name}_f1': 2 * p * r / max(p + r, 1e-9)}


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', default=os.path.dirname(HERE), help='checkout whose p2c package is benchmarked')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases', default='all')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--uncalibrated', action='store_true', help='also run every case without a known scale')
    ap.add_argument('--real', default='', help='comma list of real cases (needs P2C_REAL_DIR)')
    ap.add_argument('--pp', default='', help='preprocessing options as JSON, e.g. {"perspective": "off"}')
    ap.add_argument('--worker', nargs=4, metavar=('CASE', 'SEED', 'CAL', 'OUT'))
    a = ap.parse_args()
    if a.worker:
        sys.path.insert(0, os.path.abspath(a.code))
        case, seed, cal, out = a.worker
        run_case(case, int(seed), cal == '1', out)
        return
    from bench.synth import CASES
    cases = CASES if a.cases == 'all' else a.cases.split(',')
    jobs = [(c, int(s), 1) for c in cases for s in a.seeds.split(',')]
    if a.uncalibrated:
        jobs += [(c, int(s), 0) for c in cases for s in a.seeds.split(',')]
    jobs += [('real:' + r, 0, 1) for r in a.real.split(',') if r]
    results = []
    for case, seed, cal in jobs:
        tmp = tempfile.mktemp(suffix='.json')
        env = dict(os.environ, PYTHONPATH=os.path.abspath(a.code) + os.pathsep + os.path.dirname(HERE))
        if a.pp:
            env['P2C_PP'] = a.pp
        t = time.time()
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--code', a.code, '--out', a.out,
                            '--worker', case, str(seed), str(cal), tmp], env=env, capture_output=True, text=True,
                           timeout=1800)
        if p.returncode != 0 or not os.path.exists(tmp):
            r = dict(case=case, seed=seed, calibrated=bool(cal), error=p.stderr[-2000:])
        else:
            r = json.load(open(tmp))
        results.append(r)
        print(f"{case:14s} s{seed} cal={cal} {time.time() - t:5.1f}s  "
              f"F1={r.get('wall_f1', float('nan')):.3f} IoU={r.get('wall_iou', float('nan')):.3f} "
              f"P={r.get('wall_precision', float('nan')):.3f} R={r.get('wall_recall', float('nan')):.3f} "
              f"scale_err={r.get('scale_err', float('nan')):.3f}" + ('  ERROR' if r.get('error') else ''), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
