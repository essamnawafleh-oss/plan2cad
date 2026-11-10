"""Shared geometry: turns model objects into drawable primitives (image px)."""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from shapely import affinity

from .blocks import block_polylines


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ----------------------------------------------------------------- walls
def wall_polygon(w, scale):
    tp = w.t / scale
    if len(w.pts) < 2:
        return None
    return LineString(w.pts).buffer(tp / 2, cap_style='square', join_style='mitre')


def window_cut(win, scale):
    a, b = np.array(win.p0, float), np.array(win.p1, float)
    u = unit(b - a); n = np.array([-u[1], u[0]]); tp = win.t / scale / 2 + 1.5
    return Polygon([a - n*tp, b - n*tp, b + n*tp, a + n*tp])


def walls_union(project, cut_windows=True):
    polys = [p for p in (wall_polygon(w, project.scale) for w in project.walls) if p is not None and not p.is_empty]
    if not polys:
        return None
    U = unary_union(polys)
    if not U.is_valid:                     # repair self-intersections instead of exporting a broken outline
        U = U.buffer(0)
    if cut_windows and project.windows:
        U = U.difference(unary_union([window_cut(wi, project.scale) for wi in project.windows]))
    return U.simplify(0.15)


def rings(geom):
    """Yield closed rings (Nx2 arrays) of a (multi)polygon."""
    if geom is None or geom.is_empty:
        return
    for g in getattr(geom, 'geoms', [geom]):
        if g.geom_type != 'Polygon':
            continue
        yield np.array(g.exterior.coords)
        for r in g.interiors:
            yield np.array(r.coords)


def column_polygon(c, scale):
    s = c.s / scale
    return affinity.rotate(box(c.x - s/2, c.y - s/2, c.x + s/2, c.y + s/2), c.a, origin=(c.x, c.y))


# ----------------------------------------------------------------- windows
def window_lines(win, scale):
    a, b = np.array(win.p0, float), np.array(win.p1, float)
    u = unit(b - a); n = np.array([-u[1], u[0]]); tp = win.t / scale
    out = [[a + n*o, b + n*o] for o in (-tp/2, -tp/8, tp/8, tp/2)]
    out += [[a - n*tp/2, a + n*tp/2], [b - n*tp/2, b + n*tp/2]]
    return out


# ----------------------------------------------------------------- doors
def door_geometry(door, scale):
    """returns (leaves[list of closed polys], arcs[list of polylines])"""
    p0, p1 = np.array(door.p0, float), np.array(door.p1, float)
    w = np.linalg.norm(p1 - p0)
    if w < 1e-6:
        return [], []
    u = (p1 - p0) / w
    n = np.array([-u[1], u[0]]) * (1 if door.side >= 0 else -1)
    lw = max(0.04 / scale, 1.0)
    leaves, arcs = [], []

    def leaf(h, uu, r):
        a0 = math.atan2(uu[1], uu[0]); a1 = math.atan2(n[1], n[0])
        da = (a1 - a0 + math.pi) % (2*math.pi) - math.pi
        arcs.append(np.array([h + r*np.array([math.cos(t), math.sin(t)]) for t in np.linspace(a0, a0 + da, 24)]))
        leaves.append(np.array([h, h + n*r, h + n*r + uu*lw, h + uu*lw]))

    if door.double:
        leaf(p0, u, w/2); leaf(p1, -u, w/2)
    else:
        leaf(p0, u, w)
    return leaves, arcs


# ----------------------------------------------------------------- stairs
def stair_geometry(st, scale):
    p = np.array(st.p, float); al = np.array(st.along, float); ac = np.array(st.across, float)
    L = np.linalg.norm(al)
    n = max(2, int(round(L * scale / max(st.tread, 0.1))))
    lines = [np.array([p, p + al, p + al + ac, p + ac, p])]
    for i in range(1, n):
        q = p + al * i / n
        lines.append(np.array([q, q + ac]))
    a0 = p + ac/2 + al*0.05; a1 = p + ac/2 + al*0.92
    u = unit(al); v = unit(ac); hs = max(0.15 / scale, 3)
    arrow = [np.array([a0, a1]), np.array([a1 - u*hs + v*hs*0.6, a1, a1 - u*hs - v*hs*0.6])]
    return lines, arrow


# ----------------------------------------------------------------- furniture
def furn_rotation_dxf(f):
    """DXF block rotation (deg, CCW, world y-up) for an image-space facing angle."""
    return 90.0 - f.rot


def furn_polylines(f, scale):
    """Block geometry transformed into image px. list of (Nx2, closed, dashed)."""
    th = math.radians(furn_rotation_dxf(f))
    c, s = math.cos(th), math.sin(th)
    sx = -f.w if f.mirror else f.w
    out = []
    for pts, closed, dashed in block_polylines(f.block):
        lx, ly = pts[:, 0] * sx, pts[:, 1] * f.d
        wx, wy = c*lx - s*ly, s*lx + c*ly
        out.append((np.c_[f.x + wx/scale, f.y - wy/scale], closed, dashed))
    return out


def furn_box(f, scale):
    th = math.radians(furn_rotation_dxf(f))
    c, s = math.cos(th), math.sin(th)
    pts = []
    for lx, ly in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
        lx *= f.w; ly *= f.d
        wx, wy = c*lx - s*ly, s*lx + c*ly
        pts.append((f.x + wx/scale, f.y - wy/scale))
    return Polygon(pts)


# ----------------------------------------------------------------- snapping
def snap_angle(p0, p1, angles, tol_deg=6.0):
    """Snap p1 so that p0->p1 follows one of the dominant directions (or its perpendicular)."""
    d = np.array(p1, float) - np.array(p0, float)
    L = np.linalg.norm(d)
    if L < 1e-6:
        return tuple(p1)
    a = math.degrees(math.atan2(d[1], d[0]))
    cands = set()
    for g in list(angles) + [0.0]:
        for k in range(4):
            cands.add((g + 90*k) % 360)
    best, bd = None, 1e9
    for cnd in cands:
        diff = abs((a - cnd + 180) % 360 - 180)
        if diff < bd:
            best, bd = cnd, diff
    if bd > tol_deg:
        return tuple(p1)
    r = math.radians(best)
    L2 = L * math.cos(math.radians(bd))
    return (p0[0] + L2*math.cos(r), p0[1] + L2*math.sin(r))
