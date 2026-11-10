"""Write a project to a layered DXF (metres)."""
from __future__ import annotations

import os
import shutil
import ezdxf
from ezdxf.enums import TextEntityAlignment

from .blocks import GEO, LIB
from .geometry import (walls_union, rings, column_polygon, window_lines, door_geometry,
                       stair_geometry, furn_rotation_dxf)

LAYER_STYLE = {  # name: (aci color, lineweight 1/100 mm, plot)
    'A-WALL': (7, 50, True), 'A-COLUMN': (7, 35, True), 'A-WINDOW': (4, 18, True),
    'A-DOOR': (3, 25, True), 'A-DOOR-SWING': (3, 13, True), 'A-STAIR': (6, 25, True),
    'A-STAIR-ARROW': (6, 13, True), 'A-FURN': (30, 13, True), 'A-SANITARY': (5, 13, True),
    'A-KITCHEN': (40, 13, True), 'A-COURT': (82, 13, True), 'A-TEXT': (2, 18, True),
    'A-LEVEL': (3, 13, True), 'A-DIM': (1, 13, True), 'A-SCAN': (9, -3, False), 'A-TRACE': (8, -3, False),
}


def _arc_from_points(pts, T, s):
    """Circle through the first, middle and last point of a swing polyline -> (centre, r, a0, a1) in DXF
    space with a counter-clockwise sweep, or None when degenerate."""
    import math
    import numpy as np
    if len(pts) < 3:
        return None
    P = np.array([T(*p) for p in pts], float)
    a, b, c = P[0], P[len(P) // 2], P[-1]
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    ctr = np.array([ux, uy]); r = float(np.linalg.norm(a - ctr))
    ang = lambda q: math.degrees(math.atan2(q[1] - uy, q[0] - ux)) % 360
    a0, am, a1 = ang(a), ang(b), ang(c)
    ccw = (am - a0) % 360 < (a1 - a0) % 360            # the sweep a0->a1 CCW passes the middle point
    return (tuple(map(float, ctr)), r, a0, a1) if ccw else (tuple(map(float, ctr)), r, a1, a0)


def verify_dxf(path, project=None):
    """Re-open an exported file and check it: parse + ezdxf audit, units, referenced layers / text styles /
    blocks / linetypes exist, finite coordinates, expected entity counts.  Returns a report dict with
    'ok' and a list of 'problems'."""
    import math
    import ezdxf
    rep = dict(ok=True, problems=[], counts={})
    try:
        doc = ezdxf.readfile(path)
    except Exception as e:
        return dict(ok=False, problems=[f'cannot read: {e}'], counts={})
    aud = doc.audit()
    if aud.has_errors:
        rep['problems'] += [f'audit: {e.message}' for e in aud.errors][:20]
    if aud.has_fixes:                                     # something was broken and had to be repaired on load
        rep['problems'] += [f'audit fix: {e.message}' for e in aud.fixes][:20]
    if doc.header.get('$INSUNITS') != 6:
        rep['problems'].append('drawing units are not metres')
    msp = doc.modelspace()
    for e in msp:
        lay = e.dxf.get('layer', '0')
        if lay not in doc.layers:
            rep['problems'].append(f'{e.dxftype()} on missing layer {lay}')
        st = e.dxf.get('style', None) if e.dxftype() in ('TEXT', 'MTEXT') else None
        if st and st not in doc.styles:
            rep['problems'].append(f'text style {st} missing')
        lt = e.dxf.get('linetype', None)
        if lt and lt not in ('BYLAYER', 'BYBLOCK') and lt not in doc.linetypes:
            rep['problems'].append(f'linetype {lt} missing')
        if e.dxftype() == 'INSERT' and e.dxf.name not in doc.blocks:
            rep['problems'].append(f'block {e.dxf.name} missing')
        if e.dxftype() == 'LWPOLYLINE':
            if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in e.get_points('xy')):
                rep['problems'].append('non-finite polyline vertex')
        key = f"{e.dxftype()}@{lay}"
        rep['counts'][key] = rep['counts'].get(key, 0) + 1
    if project is not None:
        exp = dict(doors=len(project.doors), columns=len(project.columns), furniture=len(project.furniture))
        got_cols = rep['counts'].get('LWPOLYLINE@A-COLUMN', 0)
        if got_cols != exp['columns']:
            rep['problems'].append(f'columns: expected {exp["columns"]}, found {got_cols}')
        arcs = rep['counts'].get('ARC@A-DOOR-SWING', 0) + rep['counts'].get('LWPOLYLINE@A-DOOR-SWING', 0)
        want = sum(2 if d.double else 1 for d in project.doors)
        if arcs != want:
            rep['problems'].append(f'door swings: expected {want}, found {arcs}')
        ins = sum(v for k, v in rep['counts'].items() if k.startswith('INSERT@'))
        want_f = sum(1 for f in project.furniture if f.block in GEO)
        if ins != want_f:
            rep['problems'].append(f'furniture blocks: expected {want_f}, found {ins}')
        if project.walls and rep['counts'].get('LWPOLYLINE@A-WALL', 0) == 0:
            rep['problems'].append('walls missing')
    rep['ok'] = not rep['problems']
    return rep


def _max_y(project):
    """Image height to flip y when no image size is given: the lowest point of the geometry."""
    ys = [0.0]
    for w in project.walls:
        ys += [p[1] for p in w.pts]
    for o in list(project.doors) + list(project.windows):
        ys += [o.p0[1], o.p1[1]]
    for c in project.columns:
        ys.append(c.y)
    for f in project.furniture:
        ys.append(f.y)
    for p in project.polys:
        ys += [q[1] for q in p.pts]
    for t in list(project.texts) + list(project.levels):
        ys.append(t.y)
    for d in project.dims:
        ys += [d.p0[1], d.p1[1]]
    return max(ys)


def export_dxf(project, path, image_size=None, working_image_file=None,
               include_scan=True, include_trace=True):
    s = project.scale
    H = image_size[1] if image_size else _max_y(project)
    T = lambda x, y: (round(float(x) * s, 4), round((H - float(y)) * s, 4))

    doc = ezdxf.new('R2010', setup=True)
    doc.header['$INSUNITS'] = 6
    doc.header['$MEASUREMENT'] = 1
    for name, (col, lw, plot) in LAYER_STYLE.items():
        L = doc.layers.add(name, color=col)
        L.dxf.lineweight = lw
        L.dxf.plot = 1 if plot else 0
    doc.layers.get('A-SCAN').off(); doc.layers.get('A-TRACE').off()
    if 'P2C_TEXT' not in doc.styles:                     # Arial ships with Windows and covers Arabic
        doc.styles.new('P2C_TEXT', dxfattribs={'font': 'arial.ttf'})
    msp = doc.modelspace()

    def pl(pts, layer, closed=False):
        msp.add_lwpolyline([T(*p) for p in pts], close=closed, dxfattribs={'layer': layer})

    # walls (windows cut out)
    for r in rings(walls_union(project)):
        pl(r[:-1], 'A-WALL', True)
    for c in project.columns:
        pts = list(column_polygon(c, s).exterior.coords)[:-1]
        pl(pts, 'A-COLUMN', True)
        h = msp.add_hatch(color=7, dxfattribs={'layer': 'A-COLUMN'})
        h.paths.add_polyline_path([T(*p) for p in pts], is_closed=True)
    for w in project.windows:
        for ln in window_lines(w, s):
            pl(ln, 'A-WINDOW')
    for d in project.doors:
        leaves, arcs = door_geometry(d, s)
        for l in leaves: pl(l, 'A-DOOR', True)
        for a in arcs:                                   # true ARC entities (editable radius / angles)
            arc = _arc_from_points(a, T, s)
            if arc is None:
                pl(a, 'A-DOOR-SWING')
            else:
                c, r, a0, a1 = arc
                msp.add_arc(c, r, a0, a1, dxfattribs={'layer': 'A-DOOR-SWING'})
    for st in project.stairs:
        lines, arrow = stair_geometry(st, s)
        for l in lines: pl(l, 'A-STAIR')
        for a in arrow: pl(a, 'A-STAIR-ARROW')
    # furniture blocks
    used = sorted({f.block for f in project.furniture if f.block in GEO})
    for name in used:
        blk = doc.blocks.new(name=name)
        for g in GEO[name]:
            a = {'layer': '0', 'color': 256}
            if g[0] in ('pl', 'dpl'):
                if g[0] == 'dpl':
                    a['linetype'] = 'DASHED'
                blk.add_lwpolyline([(float(x), float(y)) for x, y in g[1]], close=g[2], dxfattribs=a)
            elif g[0] == 'circle':
                blk.add_circle(g[1], g[2], dxfattribs=a)
    for f in project.furniture:
        if f.block not in GEO:
            continue
        msp.add_blockref(f.block, T(f.x, f.y), dxfattribs={
            'layer': f.layer or LIB.get(f.block, ('A-FURN',))[0],
            'xscale': -f.w if f.mirror else f.w, 'yscale': f.d, 'zscale': 1,
            'rotation': furn_rotation_dxf(f)})
    for p in project.polys:
        if p.layer not in doc.layers:
            doc.layers.add(p.layer)
        pl(p.pts, p.layer, p.closed)
    for t in project.texts:
        msp.add_text(t.text, height=t.h, dxfattribs={'layer': 'A-TEXT', 'style': 'P2C_TEXT'}).set_placement(
            T(t.x, t.y), align=TextEntityAlignment.MIDDLE_CENTER)
    for lv in project.levels:
        x, y = T(lv.x, lv.y); r = 0.15
        msp.add_circle((x, y), r, dxfattribs={'layer': 'A-LEVEL'})
        msp.add_line((x - r*1.4, y), (x + r*1.4, y), dxfattribs={'layer': 'A-LEVEL'})
        msp.add_line((x, y - r*1.4), (x, y + r*1.4), dxfattribs={'layer': 'A-LEVEL'})
        msp.add_line((x, y), (x + 1.1, y), dxfattribs={'layer': 'A-LEVEL'})
        msp.add_text(lv.value, height=0.25, dxfattribs={'layer': 'A-LEVEL', 'style': 'P2C_TEXT'}).set_placement((x + 0.2, y + 0.06))
    if project.dims:
        if 'P2C_DIM' not in doc.dimstyles:
            ds = doc.dimstyles.new('P2C_DIM')
            ds.dxf.dimtxt = 0.18; ds.dxf.dimasz = 0.12; ds.dxf.dimtsz = 0.12; ds.dxf.dimexo = 0.05
            ds.dxf.dimexe = 0.1; ds.dxf.dimgap = 0.05; ds.dxf.dimdec = 2; ds.dxf.dimclrt = 1
            ds.dxf.dimtxsty = 'P2C_TEXT'
        for dm in project.dims:
            try:
                dim = msp.add_aligned_dim(p1=T(*dm.p0), p2=T(*dm.p1), distance=0.0, dimstyle='P2C_DIM',
                                          override={'dimtad': 1}, text=dm.text, dxfattribs={'layer': 'A-DIM'})
                dim.render()
            except Exception:
                msp.add_line(T(*dm.p0), T(*dm.p1), dxfattribs={'layer': 'A-DIM'})
    if include_trace:
        for tr in project.trace:
            if len(tr) > 1:
                pl(tr, 'A-TRACE')
    if include_scan and working_image_file and image_size:
        base = os.path.splitext(os.path.basename(path))[0] + '_scan.png'
        dst = os.path.join(os.path.dirname(os.path.abspath(path)), base)
        if os.path.abspath(working_image_file) != os.path.abspath(dst):
            shutil.copy(working_image_file, dst)
        idef = doc.add_image_def(filename=base, size_in_pixel=image_size)
        msp.add_image(idef, insert=(0, 0), size_in_units=(image_size[0]*s, image_size[1]*s),
                      dxfattribs={'layer': 'A-SCAN'})
    doc.saveas(path)
    return path
