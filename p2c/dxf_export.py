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
        for a in arcs: pl(a, 'A-DOOR-SWING')
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
        msp.add_text(t.text, height=t.h, dxfattribs={'layer': 'A-TEXT', 'style': 'OpenSans'}).set_placement(
            T(t.x, t.y), align=TextEntityAlignment.MIDDLE_CENTER)
    for lv in project.levels:
        x, y = T(lv.x, lv.y); r = 0.15
        msp.add_circle((x, y), r, dxfattribs={'layer': 'A-LEVEL'})
        msp.add_line((x - r*1.4, y), (x + r*1.4, y), dxfattribs={'layer': 'A-LEVEL'})
        msp.add_line((x, y - r*1.4), (x, y + r*1.4), dxfattribs={'layer': 'A-LEVEL'})
        msp.add_line((x, y), (x + 1.1, y), dxfattribs={'layer': 'A-LEVEL'})
        msp.add_text(lv.value, height=0.25, dxfattribs={'layer': 'A-LEVEL'}).set_placement((x + 0.2, y + 0.06))
    if project.dims:
        if 'P2C_DIM' not in doc.dimstyles:
            ds = doc.dimstyles.new('P2C_DIM')
            ds.dxf.dimtxt = 0.18; ds.dxf.dimasz = 0.12; ds.dxf.dimtsz = 0.12; ds.dxf.dimexo = 0.05
            ds.dxf.dimexe = 0.1; ds.dxf.dimgap = 0.05; ds.dxf.dimdec = 2; ds.dxf.dimclrt = 1
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
