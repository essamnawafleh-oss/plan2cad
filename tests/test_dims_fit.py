"""Dimension parsing, robust calibration and robust fitting (synthetic, deterministic)."""
import math
import numpy as np
import pytest

from p2c.engine import dims as DM
from p2c.engine.fit import fit_to_dimensions, dimension_report
from p2c.model import Project, Wall, Dim


@pytest.mark.parametrize('txt,val,unit', [
    ('3.45', 3.45, 'm'), ('3,45', 3.45, 'm'), ('12.5', 12.5, 'm'), ('3.45m', 3.45, 'm'),
    ('345cm', 3.45, 'cm'), ('3450mm', 3.45, 'mm'), ('٣٫٤٥', 3.45, 'm'), ('0.9', 0.9, 'm')])
def test_parse_sure_values(txt, val, unit):
    h = DM.parse_hypotheses(txt)
    assert h and abs(h[0][0] - val) < 1e-9 and h[0][1] == unit and h[0][2] == 1.0


def test_integer_is_ambiguous_not_guessed():
    h = DM.parse_hypotheses('345')
    assert {u for _, u, _ in h} == {'cm', 'mm'} and all(p < 1 for *_, p in h)
    assert DM.parse_value('345') is None              # no sure reading
    assert DM.parse_hypotheses('abc') == [] and DM.parse_hypotheses('') == []


def _cand(value_txt, true_len_px, scale, conf=0.95):
    """A dimension candidate as measure_dimensions(…) would produce (tick pairs -> scales)."""
    hyps = DM.parse_hypotheses(value_txt)
    return dict(text=value_txt, conf=conf, value=hyps[0][0], hyps=[(v, u, p, [v / true_len_px]) for v, u, p in hyps],
                cands=[hyps[0][0] / true_len_px])


def test_vote_picks_unit_from_consistent_evidence():
    s = 0.02
    c = [_cand('3.45', 3.45 / s, s), _cand('2.10', 2.10 / s, s), _cand('410', 4.10 / s, s), _cand('290', 2.90 / s, s)]
    scale, n, info = DM.vote_scale(c)
    assert abs(scale / s - 1) < 0.01 and n == 4 and info['units'].get('cm') == 2


def test_integer_only_drawing_flags_ambiguity():
    s = 0.02
    c = [_cand(t, int(t) / 100 / s, s) for t in ('345', '410', '290', '515', '600')]
    scale, n, info = DM.vote_scale(c)
    assert abs(scale / s - 1) < 0.01                    # cm prior wins…
    assert 'ambiguous' in info                          # …and the runner-up reading is reported


def _dim(value, len_px, conf=0.9):
    return dict(value=value, len_px=len_px, conf=conf)


def test_robust_scale_rejects_misread():
    s = 0.025
    ds = [_dim(v, v / s) for v in (3.45, 2.10, 4.80, 1.20, 5.55, 2.95)] + [_dim(5.43, 4.53 / s)]   # swapped digits
    sc, rep = DM.robust_scale(ds)
    assert abs(sc / s - 1) < 0.003 and rep['rejected'] == 1
    assert ds[-1]['status'].startswith('rejected') and all(d['status'] == 'used' for d in ds[:-1])


def test_robust_scale_reports_unit_error():
    s = 0.025
    ds = [_dim(v, v / s) for v in (3.45, 2.10, 4.80, 1.20, 5.55)] + [_dim(34.5, 3.45 / s)]
    sc, rep = DM.robust_scale(ds)
    assert 'unit' in ds[-1]['status']


def _room(s=0.01):
    """5.00 x 4.00 m room (centre lines), walls 0.20 m; dimensions measured face to face inside."""
    W, H = 5.0 / s, 4.0 / s
    o = 100.0
    walls = [Wall([(o, o), (o + W, o)], 0.2), Wall([(o + W, o), (o + W, o + H)], 0.2),
             Wall([(o, o + H), (o + W, o + H)], 0.2), Wall([(o, o), (o, o + H)], 0.2)]
    return walls, o, W, H


def test_fit_moves_wall_to_match_and_rejects_wrong_dimension():
    s = 0.01
    walls, o, W, H = _room(s)
    walls[1].pts = [(p[0] - 6, p[1]) for p in walls[1].pts]         # east wall drawn 6 cm too far west
    t = 0.2 / s
    dims = [Dim((o + t / 2, o + 100), (o + W - t / 2, o + 100), '4.80', 4.80),         # inner width (true)
            Dim((o + t / 2, o + 300), (o + W - t / 2, o + 300), '4.80', 4.80),
            Dim((o + 100, o + t / 2), (o + 100, o + H - t / 2), '3.80', 3.80),         # inner height (true)
            Dim((o + 200, o + t / 2), (o + 200, o + H - t / 2), '3.08', 3.08)]         # misread (3.80 -> 3.08)
    p = Project(scale=s, scale_calibrated=True, walls=walls, dims=dims)
    p.corrections = []
    info = fit_to_dimensions(p)
    rep = {i: r for i, r in enumerate(dimension_report(p))}
    assert abs(rep[0]['diff']) < 0.005 and abs(rep[2]['diff']) < 0.005
    assert p.dims[3].status.startswith('rejected')
    assert info['walls_moved'] >= 1
    # the north/south walls were not distorted to satisfy the wrong 3.08
    assert abs(p.walls[2].pts[0][1] - (o + H)) * s < 0.02


def test_fit_never_moves_more_than_cap():
    s = 0.01
    walls, o, W, H = _room(s)
    t = 0.2 / s
    dims = [Dim((o + t / 2, o + 100), (o + W - t / 2, o + 100), '5.10', 5.10)]       # 30 cm off: beyond 'far'?
    p = Project(scale=s, scale_calibrated=True, walls=walls, dims=dims); p.corrections = []
    before = [list(w.pts) for w in p.walls]
    fit_to_dimensions(p, max_shift=0.15)
    moved = max(abs(a[0] - b[0]) for w, bw in zip(p.walls, before) for a, b in zip(w.pts, bw)) * s
    assert moved <= 0.15 + 1e-6


def test_fit_reports_dimension_without_walls():
    s = 0.01
    walls, o, W, H = _room(s)
    dims = [Dim((o + 900, o + 900), (o + 1200, o + 900), '3.00', 3.0)]                # nowhere near walls
    p = Project(scale=s, scale_calibrated=True, walls=walls, dims=dims); p.corrections = []
    fit_to_dimensions(p)
    assert p.dims[0].status == 'rejected:no_wall_at_end'
