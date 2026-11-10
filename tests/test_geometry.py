"""Wall measurement, evidence-gated joins and geometric validation on synthetic drawings."""
import math
import numpy as np
import cv2
import pytest

from p2c.model import Project, Wall, Door, Window
from p2c.engine import walls as WG
from p2c.engine import validate as V
from p2c.engine.neural import trim_overshoots

S = 0.01


def _canvas(h=600, w=800):
    return np.full((h, w), 255, np.uint8)


def _wall(img, a, b, t_m, style='filled'):
    a, b = np.array(a, float), np.array(b, float)
    u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]]); tp = t_m / S - 1
    poly = np.array([a - n * tp / 2, b - n * tp / 2, b + n * tp / 2, a + n * tp / 2])
    if style == 'filled':
        cv2.fillPoly(img, [np.round(poly).astype(np.int32)], 0)
    else:
        cv2.polylines(img, [np.round(poly).astype(np.int32)], True, 0, 1)


# ------------------------------------------------------------------ measurement
@pytest.mark.parametrize('style', ['filled', 'hollow'])
@pytest.mark.parametrize('t', [0.10, 0.20, 0.30])
def test_measure_recovers_centre_and_thickness(style, t):
    img = _canvas(); _wall(img, (100, 300), (700, 300), t, style)
    w = Wall([(100, 300 + 0.3 * t / S), (700, 300 + 0.3 * t / S)], 0.15)      # proposal off-centre, wrong t
    m = WG.measure_wall(w, img, S)
    assert m['ok']
    assert abs((300 + 0.3 * t / S + m['offset_px']) - 300) < 1.0
    assert abs(m['t_m'] - t) < 0.015


def test_measure_angled_wall():
    img = _canvas(); a, b = (150, 500), (650, 150)
    _wall(img, a, b, 0.20)
    w = Wall([a, b], 0.20)
    m = WG.measure_wall(w, img, S)
    assert m['ok'] and abs(m['t_m'] - 0.20) < 0.02 and abs(m['offset_px']) < 1.0


def test_measure_ignores_text_and_furniture_next_to_wall():
    img = _canvas(); _wall(img, (100, 300), (700, 300), 0.20)
    cv2.putText(img, 'ROOM 3.45', (250, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.rectangle(img, (420, 330), (560, 420), 0, 1)
    m = WG.measure_wall(Wall([(100, 300), (700, 300)], 0.2), img, S)
    assert m['ok'] and abs(m['t_m'] - 0.20) < 0.015 and abs(m['offset_px']) < 1.0


def test_single_line_is_not_wall_evidence():
    img = _canvas(); cv2.line(img, (100, 300), (700, 300), 0, 1)      # furniture / annotation line
    m = WG.measure_wall(Wall([(100, 300), (700, 300)], 0.2), img, S)
    assert not m['ok']


def test_different_walls_keep_different_thickness():
    img = _canvas(); _wall(img, (100, 150), (700, 150), 0.30); _wall(img, (100, 400), (700, 400), 0.10)
    ws = [Wall([(100, 150), (700, 150)], 0.2), Wall([(100, 400), (700, 400)], 0.2)]
    out = WG.refine_walls(ws, img, S)
    assert [round(w.t, 2) for w in out] == [0.30, 0.10]


def test_refine_drops_walls_without_any_evidence():
    img = _canvas(); _wall(img, (100, 150), (700, 150), 0.20)
    ws = [Wall([(100, 150), (700, 150)], 0.2), Wall([(100, 450), (700, 450)], 0.2)]   # second: blank paper
    log = []
    out = WG.refine_walls(ws, img, S, log=log)
    assert len(out) == 1 and any(e['action'] == 'drop_wall' for e in log)


# ------------------------------------------------------------------ continuity
def test_join_bridges_scan_break_but_not_empty_gap():
    img = _canvas()
    _wall(img, (100, 300), (700, 300), 0.2)                 # horizontal wall
    _wall(img, (400, 300), (400, 120), 0.2)                 # vertical wall drawn up to it
    img[290:310, 395:405] = 255                             # a 10 px hole in the drawing at the T (scan defect)?
    ws = [Wall([(100, 300), (700, 300)], 0.2), Wall([(400, 230), (400, 120)], 0.2)]   # detected 60 cm short
    log = []
    WG.join_with_evidence(ws, img, S, max_gap=0.8, log=log)
    assert any(e['action'] == 'join_wall' for e in log)    # ink in the gap -> joined
    img2 = _canvas(); _wall(img2, (100, 300), (700, 300), 0.2); _wall(img2, (400, 230), (400, 120), 0.2)
    ws2 = [Wall([(100, 300), (700, 300)], 0.2), Wall([(400, 230), (400, 120)], 0.2)]
    log2 = []
    WG.join_with_evidence(ws2, img2, S, max_gap=0.8, log=log2)
    assert ws2[1].pts[0][1] == 230 and any(e['action'] == 'join_refused' for e in log2)   # paper -> refused


def test_tiny_gap_closed_without_evidence():
    img = _canvas()
    ws = [Wall([(100, 300), (700, 300)], 0.2), Wall([(400, 270), (400, 120)], 0.2)]       # 10 cm short of the face
    WG.join_with_evidence(ws, img, S, max_gap=0.8, free_gap=0.12)
    assert ws[1].pts[0][1] > 295


@pytest.mark.parametrize('kind', ['L', 'T', 'X'])
def test_junctions_are_connected_and_clean(kind):
    if kind == 'L':
        ws = [Wall([(100, 100), (500, 100)], 0.2), Wall([(100, 100), (100, 400)], 0.2)]
    elif kind == 'T':
        ws = [Wall([(100, 100), (500, 100)], 0.2), Wall([(300, 100), (300, 400)], 0.2)]
    else:
        ws = [Wall([(100, 250), (500, 250)], 0.2), Wall([(300, 100), (300, 400)], 0.2)]
    p = Project(scale=S, walls=ws)
    iss = V.validate(p)
    assert not [d for d in iss if d['code'] in ('open_gap', 'isolated_wall', 'duplicate_wall', 'invalid_polygon')]


def test_overshoot_stub_trimmed_at_corner():
    ws = [Wall([(100, 100), (520, 100)], 0.2), Wall([(100, 80), (100, 400)], 0.2)]       # '+' stubs
    trim_overshoots(ws, S)
    assert abs(ws[0].pts[1][0] - 520) < 1e-6                 # free end untouched
    assert ws[1].pts[0][1] == pytest.approx(100, abs=0.5)    # stub cut back to the corner


def test_angled_geometry_not_forced_orthogonal():
    a = Wall([(100, 400), (400, 120)], 0.2)
    p = Project(scale=S, walls=[a, Wall([(400, 120), (700, 120)], 0.2)])
    V.auto_fix(p)
    assert p.walls[0].pts == [(100, 400), (400, 120)]


# ------------------------------------------------------------------ validation
def test_validation_finds_and_fixes():
    ws = [Wall([(100, 100), (500, 100)], 0.2),
          Wall([(100, 100), (500, 100)], 0.2),                # duplicate
          Wall([(200, 200), (200, 200.2)], 0.2),              # zero length
          Wall([(float('nan'), 0), (1, 1)], 0.2),             # invalid
          Wall([(600, 300), (610, 300)], 0.2),                # tiny isolated
          Wall([(100, 100), (100, 300)], 0.9)]                # implausible thickness
    p = Project(scale=S, walls=ws)
    p.doors = [Door((2000, 2000), (2090, 2000))]                # not on any wall
    codes = {d['code'] for d in V.validate(p)}
    assert {'duplicate_wall', 'zero_length', 'invalid_coords', 'tiny_wall', 'bad_thickness', 'off_wall'} <= codes
    sev = V.summary(V.validate(p))
    assert sev['critical'] >= 1
    log = []
    V.auto_fix(p, log)
    left = {d['code'] for d in V.validate(p)}
    assert not ({'duplicate_wall', 'zero_length', 'invalid_coords'} & left)
    assert 'tiny_wall' in left and 'bad_thickness' in left     # not "safe" -> reported, never silently changed
    assert all(e['auto'] for e in log)


def test_open_gap_reported_and_fixable():
    p = Project(scale=S, walls=[Wall([(100, 100), (500, 100)], 0.2), Wall([(300, 300), (300, 125)], 0.2)])
    gaps = [d for d in V.validate(p) if d['code'] == 'open_gap']
    assert gaps and abs(gaps[0]['gap_m'] - 0.05) < 0.01          # 125 - (100 + 10) - 10 px
    V.apply_fix(p, gaps[0])
    assert not [d for d in V.validate(p) if d['code'] == 'open_gap']
    assert p.walls[1].origin == 'fixed'


def test_overlapping_collinear_walls_merged_on_request():
    p = Project(scale=S, walls=[Wall([(100, 100), (400, 100)], 0.2), Wall([(300, 100), (600, 100)], 0.2)])
    ov = [d for d in V.validate(p) if d['code'] == 'overlap_wall']
    assert ov
    V.apply_fix(p, ov[0])
    assert len(p.walls) == 1 and p.walls[0].pts[0][0] == pytest.approx(100) and p.walls[0].pts[1][0] == pytest.approx(600)
