"""DXF export integrity: units, layers, styles, blocks, entity counts, coordinates, re-open + audit."""
import math
import numpy as np
import ezdxf
import pytest

from p2c.model import Project, Wall, Door, Window, Column, Furn, Text, Level, Dim
from p2c.dxf_export import export_dxf, verify_dxf
from p2c.geometry import door_geometry


def _project():
    s = 0.01
    o = 100
    walls = [Wall([(o, o), (o + 500, o)], 0.2), Wall([(o + 500, o), (o + 500, o + 400)], 0.2),
             Wall([(o + 500, o + 400), (o, o + 400)], 0.2), Wall([(o, o + 400), (o, o)], 0.2)]
    p = Project(scale=s, walls=walls)
    p.doors = [Door((o + 150, o + 400), (o + 240, o + 400), 1, False), Door((o, o + 100), (o, o + 260), -1, True)]
    p.windows = [Window((o + 300, o), (o + 420, o), 0.2)]
    p.columns = [Column(o + 250, o + 200, 0.3)]
    p.furniture = [Furn('SOFA_3', o + 250, o + 300, 2.0, 0.9, 90)]
    p.texts = [Text(o + 250, o + 150, 'غرفة نوم', 0.3)]
    p.levels = [Level(o + 100, o + 100, '+0.45')]
    p.dims = [Dim((o, o - 50), (o + 500, o - 50), '5.00', 5.0)]
    return p


def test_export_reopens_clean(tmp_path):
    p = _project()
    f = str(tmp_path / 'a.dxf')
    export_dxf(p, f, (800, 700), None, include_scan=False)
    rep = verify_dxf(f, p)
    assert rep['ok'], rep['problems']
    d = ezdxf.readfile(f)
    assert d.header['$INSUNITS'] == 6
    for lay in ('A-WALL', 'A-DOOR', 'A-DOOR-SWING', 'A-WINDOW', 'A-COLUMN', 'A-FURN', 'A-TEXT', 'A-DIM', 'A-LEVEL'):
        assert lay in d.layers
    assert 'P2C_TEXT' in d.styles and 'SOFA_3' in d.blocks


def test_entity_counts(tmp_path):
    p = _project()
    f = str(tmp_path / 'a.dxf')
    export_dxf(p, f, (800, 700), None, include_scan=False)
    msp = ezdxf.readfile(f).modelspace()
    assert len(msp.query('ARC[layer=="A-DOOR-SWING"]')) == 3            # single + double leaf
    assert len(msp.query('INSERT')) == 1
    assert len(msp.query('LWPOLYLINE[layer=="A-COLUMN"]')) == 1
    assert len(msp.query('DIMENSION')) == 1
    assert len(msp.query('LWPOLYLINE[layer=="A-WALL"]')) == 1           # window cuts the ring open -> one outline


def test_coordinates_metres_y_up(tmp_path):
    p = _project()
    f = str(tmp_path / 'a.dxf')
    H = 700
    export_dxf(p, f, (800, H), None, include_scan=False)
    pts = [v for e in ezdxf.readfile(f).modelspace().query('LWPOLYLINE[layer=="A-WALL"]') for v in e.get_points('xy')]
    xs = [x for x, y in pts]; ys = [y for x, y in pts]
    # room centre lines 100..600 px, 0.20 m walls -> outer face at (100-10) px * 0.01
    assert min(xs) == pytest.approx(0.90, abs=1e-3) and max(xs) == pytest.approx(6.10, abs=1e-3)
    assert min(ys) == pytest.approx((H - 510) * 0.01, abs=1e-3) and max(ys) == pytest.approx((H - 90) * 0.01, abs=1e-3)


def test_door_arc_matches_drawn_swing(tmp_path):
    p = _project()
    f = str(tmp_path / 'a.dxf')
    H = 700
    export_dxf(p, f, (800, H), None, include_scan=False)
    arc = list(ezdxf.readfile(f).modelspace().query('ARC'))[0]
    leaves, arcs = door_geometry(p.doors[0], p.scale)
    poly = [(x * p.scale, (H - y) * p.scale) for x, y in arcs[0]]
    c = np.array([arc.dxf.center.x, arc.dxf.center.y]); r = arc.dxf.radius
    assert all(abs(np.linalg.norm(np.array(q) - c) - r) < 1e-3 for q in poly)
    # every polyline point lies inside the arc's CCW sweep
    a0, a1 = arc.dxf.start_angle % 360, arc.dxf.end_angle % 360
    for q in poly[1:-1]:
        a = math.degrees(math.atan2(q[1] - c[1], q[0] - c[0])) % 360
        assert (a - a0) % 360 <= (a1 - a0) % 360 + 1e-6


def test_no_duplicate_wall_outlines(tmp_path):
    p = _project()
    p.walls.append(Wall(list(p.walls[0].pts), 0.2))                    # exact duplicate is absorbed by the union
    f = str(tmp_path / 'a.dxf')
    export_dxf(p, f, (800, 700), None, include_scan=False)
    assert len(ezdxf.readfile(f).modelspace().query('LWPOLYLINE[layer=="A-WALL"]')) == 1


def test_verify_detects_missing_block(tmp_path):
    p = _project()
    f = str(tmp_path / 'a.dxf')
    export_dxf(p, f, (800, 700), None, include_scan=False)
    d = ezdxf.readfile(f)
    d.modelspace().add_blockref('NOPE', (0, 0)) if 'NOPE' in d.blocks else None
    d.blocks.new('TMP'); d.modelspace().add_blockref('TMP', (0, 0)); d.blocks.delete_block('TMP', safe=False)
    d.saveas(f)
    rep = verify_dxf(f)
    assert not rep['ok']
