"""Drawing canvas: shows the scan + the CAD model and implements the editing tools."""
from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from PySide6.QtGui import (QPainterPath, QPen, QBrush, QColor, QPainter, QPixmap, QImage,
                           QPainterPathStroker, QFont, QTransform)
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsItem, QInputDialog, QGraphicsEllipseItem)

from ..model import Wall, Column, Door, Window, Stair, Furn, Poly, Text, Level, dist
from .. import geometry as G
from ..blocks import LIB
from . import theme
from .i18n import tr

KIND_LAYER = {'walls': 'A-WALL', 'columns': 'A-COLUMN', 'doors': 'A-DOOR', 'windows': 'A-WINDOW',
              'stairs': 'A-STAIR', 'texts': 'A-TEXT', 'levels': 'A-LEVEL'}


def qpath_from_lines(lines, closed_flags=None):
    path = QPainterPath()
    for k, pts in enumerate(lines):
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        path.moveTo(QPointF(*pts[0]))
        for q in pts[1:]:
            path.lineTo(QPointF(*q))
        if closed_flags and closed_flags[k]:
            path.closeSubpath()
    return path


def qpath_from_polygon(poly):
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for r in G.rings(poly):
        path.moveTo(QPointF(*r[0]))
        for q in r[1:]:
            path.lineTo(QPointF(*q))
        path.closeSubpath()
    return path


def cosmetic(color, w=1.2, style=Qt.PenStyle.SolidLine):
    pen = QPen(color if isinstance(color, QColor) else QColor(color), w, style, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    pen.setCosmetic(True)
    return pen


class ObjItem(QGraphicsPathItem):
    """One model object on the canvas."""

    def __init__(self, kind, idx, path, shape_path=None, color='#000', width=1.2, brush=None, fill_sel=False):
        super().__init__(path)
        self.kind, self.idx = kind, idx
        self._shape = shape_path
        self._color = color
        self._width = width
        self._brush = brush
        self._fill_sel = fill_sel
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self._apply()

    def _apply(self):
        sel = self.isSelected()
        if sel:
            self.setPen(cosmetic(theme.SELECT_COLOR, self._width + 1.0))
            self.setBrush(QBrush(QColor(255, 45, 120, 60)) if self._fill_sel else (self._brush or Qt.BrushStyle.NoBrush))
        else:
            self.setPen(cosmetic(self._color, self._width) if self._color else QPen(Qt.PenStyle.NoPen))
            self.setBrush(self._brush or QBrush(Qt.BrushStyle.NoBrush))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._apply()
        return super().itemChange(change, value)

    def shape(self):
        if self._shape is not None:
            return self._shape
        st = QPainterPathStroker(); st.setWidth(6)
        return st.createStroke(self.path())

    def boundingRect(self):
        r = super().boundingRect()
        if self._shape is not None:
            r = r.united(self._shape.boundingRect())
        return r.adjusted(-3, -3, 3, 3)


class Handle(QGraphicsRectItem):
    def __init__(self, kind, idx, vi, x, y, size):
        super().__init__(-size/2, -size/2, size, size)
        self.kind, self.idx, self.vi = kind, idx, vi
        self.setPos(x, y)
        self.setBrush(QBrush(QColor('#FFFFFF')))
        self.setPen(cosmetic(theme.SELECT_COLOR, 1.5))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(50)
        self.setCursor(Qt.CursorShape.SizeAllCursor)


class PlanView(QGraphicsView):
    status = Signal(str)
    coords = Signal(float, float)
    selectionInfo = Signal(list)          # list of (kind, idx)
    modelChanged = Signal()
    scaleChanged = Signal()
    toolChanged = Signal(str)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.gray = None
        self.image_size = None
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(theme.C['canvas'])))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setMouseTracking(True)
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        self.visible = {k: True for k in ['A-SCAN', 'A-TRACE', 'A-DIM', 'A-WALL', 'A-COLUMN', 'A-WINDOW', 'A-DOOR', 'A-STAIR',
                                          'A-FURN', 'A-SANITARY', 'A-KITCHEN', 'A-COURT', 'A-TEXT', 'A-LEVEL']}
        self.scan_opacity = 0.45
        self.paper = None; self.scan_item = None; self.trace_item = None; self.wall_item = None
        self.obj_items = []; self.handles = []; self.preview = []
        self.undo_stack, self.redo_stack = [], []
        self.dirty = False

        # tool state
        self.tool = 'select'
        self.pts = []
        self.wall_t = 0.20
        self.col_size = 0.35
        self.poly_layer = 'A-FURN'
        self.furn_block = 'SOFA_3'
        self.place_rot = 90.0
        self.place_mirror = False
        self.fit_mode = True
        self._wall_mask = None
        self._pan = None
        self._space = False
        self._mouse = QPointF(0, 0)
        self._press_pos = None

    # ================================================================ image / model
    def set_image(self, gray):
        self.gray = gray
        h, w = gray.shape[:2]
        self.image_size = (w, h)
        img = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self.scene_.clear()
        self.obj_items = []; self.handles = []; self.preview = []
        self.paper = self.scene_.addRect(QRectF(0, 0, w, h), QPen(Qt.PenStyle.NoPen), QBrush(QColor(theme.C['paper'])))
        self.paper.setZValue(-10)
        self.scan_item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        self.scan_item.setZValue(0)
        self.scan_item.setOpacity(self.scan_opacity)
        self.scan_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene_.addItem(self.scan_item)
        self.trace_item = None; self.wall_item = None
        m = max(w, h) * 0.05
        self.scene_.setSceneRect(QRectF(-m, -m, w + 2*m, h + 2*m))
        self.rebuild(full=True)
        QTimer.singleShot(0, self.fit)

    def fit(self):
        if self.image_size:
            w, h = self.image_size
            self.fitInView(QRectF(0, 0, w, h), Qt.AspectRatioMode.KeepAspectRatio)

    def rebuild(self, full=False, keep_selection=None):
        if self.gray is None:
            return
        sc = self.scene_
        for it in self.obj_items + self.handles:
            sc.removeItem(it)
        self.obj_items, self.handles = [], []
        self._wall_mask = None
        p = self.project; s = p.scale
        self.scan_item.setVisible(self.visible['A-SCAN'])
        self.scan_item.setOpacity(self.scan_opacity)
        # raw trace
        if full or self.trace_item is None:
            if self.trace_item is not None:
                sc.removeItem(self.trace_item)
            self.trace_item = QGraphicsPathItem(qpath_from_lines(p.trace))
            self.trace_item.setPen(cosmetic(QColor(224, 123, 57, 150), 0.9))
            self.trace_item.setZValue(1)
            sc.addItem(self.trace_item)
        self.trace_item.setVisible(self.visible['A-TRACE'])
        # wall union (display only)
        if self.wall_item is not None:
            sc.removeItem(self.wall_item); self.wall_item = None
        if self.visible['A-WALL'] and p.walls:
            U = G.walls_union(p)
            self.wall_item = QGraphicsPathItem(qpath_from_polygon(U))
            self.wall_item.setPen(cosmetic('#111111', 1.3))
            self.wall_item.setBrush(QBrush(QColor(60, 60, 64, 150)))
            self.wall_item.setZValue(3)
            sc.addItem(self.wall_item)
        add = self._add
        if self.visible['A-WALL']:
            for i, w in enumerate(p.walls):
                poly = G.wall_polygon(w, s)
                if poly is None:
                    continue
                pp = qpath_from_polygon(poly)
                add(ObjItem('walls', i, pp, pp, color=None, brush=QBrush(QColor(0, 0, 0, 1)), fill_sel=True), 4)
        if self.visible['A-COLUMN']:
            for i, c in enumerate(p.columns):
                pp = qpath_from_polygon(G.column_polygon(c, s))
                add(ObjItem('columns', i, pp, pp, color='#111111', brush=QBrush(QColor('#202020'))), 6)
        if self.visible['A-WINDOW']:
            for i, w in enumerate(p.windows):
                lines = G.window_lines(w, s)
                shp = qpath_from_polygon(G.window_cut(w, s))
                add(ObjItem('windows', i, qpath_from_lines(lines), shp, color=theme.LAYER_COLORS['A-WINDOW'], width=1.1,
                            brush=QBrush(QColor(255, 255, 255, 235))), 5)
        if self.visible['A-DOOR']:
            for i, d in enumerate(p.doors):
                leaves, arcs = G.door_geometry(d, s)
                path = qpath_from_lines(leaves + arcs, [True]*len(leaves) + [False]*len(arcs))
                shp = QPainterPath(); shp.addRect(path.boundingRect())
                add(ObjItem('doors', i, path, shp, color=theme.LAYER_COLORS['A-DOOR'], width=1.2), 7)
        if self.visible['A-STAIR']:
            for i, st in enumerate(p.stairs):
                lines, arrow = G.stair_geometry(st, s)
                shp = qpath_from_lines([lines[0]], [True])
                add(ObjItem('stairs', i, qpath_from_lines(lines + arrow), shp, color=theme.LAYER_COLORS['A-STAIR']), 6)
        for i, f in enumerate(p.furniture):
            lay = f.layer or 'A-FURN'
            if not self.visible.get(lay, True):
                continue
            pls = G.furn_polylines(f, s)
            path = qpath_from_lines([q for q, _, _ in pls], [c for _, c, _ in pls])
            shp = qpath_from_polygon(G.furn_box(f, s))
            add(ObjItem('furniture', i, path, shp, color=theme.LAYER_COLORS.get(lay, '#D9730D'), width=1.1,
                        brush=QBrush(QColor(0, 0, 0, 1))), 8)
        for i, pl in enumerate(p.polys):
            if not self.visible.get(pl.layer, True):
                continue
            add(ObjItem('polys', i, qpath_from_lines([pl.pts], [pl.closed]), None,
                        color=theme.LAYER_COLORS.get(pl.layer, '#555'), width=1.1), 8)
        if self.visible['A-TEXT']:
            for i, t in enumerate(p.texts):
                path = QPainterPath()
                f = QFont(); f.setFamilies(theme.FONT_FAMILIES); f.setPointSizeF(max(t.h / s, 1.0))
                path.addText(0, 0, f, t.text)
                br = path.boundingRect()
                path.translate(t.x - br.center().x(), t.y - br.center().y())
                shp = QPainterPath(); shp.addRect(path.boundingRect())
                it = ObjItem('texts', i, path, shp, color=None, brush=QBrush(QColor(theme.LAYER_COLORS['A-TEXT'])))
                add(it, 9)
        if self.visible['A-LEVEL']:
            for i, lv in enumerate(p.levels):
                r = 0.15 / s
                path = QPainterPath()
                path.addEllipse(QPointF(lv.x, lv.y), r, r)
                path.moveTo(lv.x - r*1.4, lv.y); path.lineTo(lv.x + 1.1/s, lv.y)
                path.moveTo(lv.x, lv.y - r*1.4); path.lineTo(lv.x, lv.y + r*1.4)
                f = QFont(); f.setFamilies(theme.FONT_FAMILIES); f.setPointSizeF(max(0.25 / s, 1.0))
                tp = QPainterPath(); tp.addText(lv.x + 0.2/s, lv.y - 0.06/s, f, lv.value)
                shp = QPainterPath(); shp.addRect(path.boundingRect().united(tp.boundingRect()))
                add(ObjItem('levels', i, path.united(tp) if False else self._merge(path, tp), shp,
                            color=theme.LAYER_COLORS['A-LEVEL'], width=1.0), 9)
        if self.visible.get('A-DIM', True):
            for i, dm in enumerate(p.dims):
                a, b = np.array(dm.p0), np.array(dm.p1)
                u = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-u[1], u[0]])
                tk = 0.12 / s
                path = qpath_from_lines([[a, b], [a - (u + n) * tk / 2, a + (u + n) * tk / 2],
                                         [b - (u + n) * tk / 2, b + (u + n) * tk / 2]])
                f = QFont(); f.setFamilies(theme.FONT_FAMILIES); f.setPointSizeF(max(0.2 / s, 1.0))
                tp = QPainterPath(); tp.addText(0, 0, f, dm.text)
                br = tp.boundingRect(); m_ = (a + b) / 2 - n * (0.18 / s)
                ang = math.degrees(math.atan2(u[1], u[0]))
                if ang > 90 or ang < -90:
                    ang += 180
                tr_ = QTransform(); tr_.translate(m_[0], m_[1]); tr_.rotate(ang); tr_.translate(-br.center().x(), -br.center().y())
                path.addPath(tr_.map(tp))
                shp = QPainterPath(); shp.addRect(path.boundingRect())
                add(ObjItem('dims', i, path, shp, color=theme.LAYER_COLORS['A-DIM'], width=1.0), 9)
        if keep_selection:
            ks = set(keep_selection)
            for it in self.obj_items:
                if (it.kind, it.idx) in ks:
                    it.setSelected(True)
        self._update_handles()

    @staticmethod
    def _merge(a, b):
        a.addPath(b)
        return a

    def _add(self, item, z):
        item.setZValue(z)
        self.scene_.addItem(item)
        self.obj_items.append(item)

    # ================================================================ undo / change
    def push_undo(self):
        self.undo_stack.append(self.project.snapshot())
        if len(self.undo_stack) > 150:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.dirty = True

    def commit(self, keep=None):
        self.rebuild(keep_selection=keep)
        self.modelChanged.emit()
        self._emit_selection()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self.project.snapshot())
        self.project.restore(self.undo_stack.pop())
        self.dirty = True
        self.commit()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self.project.snapshot())
        self.project.restore(self.redo_stack.pop())
        self.dirty = True
        self.commit()

    # ================================================================ selection helpers
    def selected(self):
        return [(it.kind, it.idx) for it in self.scene_.selectedItems() if isinstance(it, ObjItem)]

    def _emit_selection(self):
        self.selectionInfo.emit(self.selected())

    def get(self, kind, idx):
        return getattr(self.project, kind)[idx]

    def delete_selected(self):
        sel = self.selected()
        if not sel:
            return
        self.push_undo()
        by = {}
        for k, i in sel:
            by.setdefault(k, []).append(i)
        for k, idxs in by.items():
            lst = getattr(self.project, k)
            for i in sorted(idxs, reverse=True):
                del lst[i]
        self.commit()

    def transform_selected(self, rot=0.0, mirror=False, flip=False, hinge=False):
        sel = self.selected()
        if not sel:
            return
        self.push_undo()
        for k, i in sel:
            o = self.get(k, i)
            if k == 'furniture':
                o.rot = (o.rot + rot) % 360
                if mirror:
                    o.mirror = not o.mirror
            elif k == 'columns':
                o.a = (o.a + rot) % 360
            elif k == 'doors':
                if flip:
                    o.side = -o.side
                if hinge:
                    o.p0, o.p1 = o.p1, o.p0
                    o.side = -o.side
                if rot:
                    c = ((o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2)
                    o.p0, o.p1 = _rot_pt(o.p0, c, rot), _rot_pt(o.p1, c, rot)
            elif k == 'stairs' and rot:
                o.along = _rot_vec(o.along, rot); o.across = _rot_vec(o.across, rot)
        self.commit(keep=sel)

    def translate(self, kind, idx, dx, dy):
        o = self.get(kind, idx)
        mv = lambda q: (q[0] + dx, q[1] + dy)
        if kind in ('walls', 'polys'):
            o.pts = [mv(q) for q in o.pts]
        elif kind in ('doors', 'windows', 'dims'):
            o.p0, o.p1 = mv(o.p0), mv(o.p1)
        elif kind == 'stairs':
            o.p = mv(o.p)
        else:
            o.x += dx; o.y += dy

    def select_all(self):
        for it in self.obj_items:
            it.setSelected(True)
        self._emit_selection()

    # ================================================================ handles
    def _update_handles(self):
        for h in self.handles:
            self.scene_.removeItem(h)
        self.handles = []
        sel = self.selected()
        if len(sel) != 1 or self.tool != 'select':
            return
        k, i = sel[0]
        o = self.get(k, i)
        pts = []
        if k in ('walls', 'polys'):
            pts = list(o.pts)
        elif k in ('doors', 'windows'):
            pts = [o.p0, o.p1]
        for vi, (x, y) in enumerate(pts):
            h = Handle(k, i, vi, x, y, 9)
            self.scene_.addItem(h)
            self.handles.append(h)

    def _commit_handles(self):
        moved = [h for h in self.handles if True]
        changed = False
        for h in moved:
            o = self.get(h.kind, h.idx)
            pos = (h.pos().x(), h.pos().y())
            if h.kind in ('walls', 'polys'):
                old = o.pts[h.vi]
                if dist(old, pos) > 0.01:
                    if not changed:
                        self.push_undo(); changed = True
                    o = self.get(h.kind, h.idx)
                    ref = o.pts[h.vi - 1] if h.vi > 0 else (o.pts[1] if len(o.pts) > 1 else None)
                    pos = self.snap(pos, ref=ref, exclude=(h.kind, h.idx))
                    o.pts[h.vi] = pos
            else:
                old = o.p0 if h.vi == 0 else o.p1
                if dist(old, pos) > 0.01:
                    if not changed:
                        self.push_undo(); changed = True
                    o = self.get(h.kind, h.idx)
                    pos = self.snap(pos, on_wall=True)
                    if h.vi == 0:
                        o.p0 = pos
                    else:
                        o.p1 = pos
        if changed:
            k, i = self.handles[0].kind, self.handles[0].idx
            self.commit(keep=[(k, i)])

    # ================================================================ snapping
    def _tol(self):
        return 10.0 / max(self.transform().m11(), 1e-6)

    def snap(self, p, ref=None, on_wall=False, exclude=None, shift=None):
        from PySide6.QtWidgets import QApplication
        if shift is None:
            shift = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        if shift:
            return (float(p[0]), float(p[1]))
        tol = self._tol()
        best, bd = None, tol
        pr = self.project
        for wi, w in enumerate(pr.walls):
            if exclude == ('walls', wi):
                continue
            for q in w.pts:
                d = dist(p, q)
                if d < bd:
                    best, bd = q, d
        for d_ in pr.doors:
            for q in (d_.p0, d_.p1):
                if dist(p, q) < bd:
                    best, bd = q, dist(p, q)
        for wi_ in pr.windows:
            for q in (wi_.p0, wi_.p1):
                if dist(p, q) < bd:
                    best, bd = q, dist(p, q)
        for k, pl in enumerate(pr.polys):
            if exclude == ('polys', k):
                continue
            for q in pl.pts:
                if dist(p, q) < bd:
                    best, bd = q, dist(p, q)
        if best is not None:
            return (float(best[0]), float(best[1]))
        if ref is not None:
            return tuple(map(float, G.snap_angle(ref, p, pr.angles)))
        if on_wall:
            proj = self.project_on_wall(p, max_d=tol * 1.5)
            if proj:
                return proj[0]
        return (float(p[0]), float(p[1]))

    def project_on_wall(self, p, max_d=None):
        """Nearest point on a wall centre line -> (point, wall, distance)."""
        best = None
        P = np.array(p, float)
        for w in self.project.walls:
            for a, b in zip(w.pts[:-1], w.pts[1:]):
                a, b = np.array(a, float), np.array(b, float)
                ab = b - a; L2 = ab @ ab
                if L2 < 1e-9:
                    continue
                t = np.clip((P - a) @ ab / L2, 0, 1)
                q = a + ab * t
                d = np.linalg.norm(P - q)
                lim = max_d if max_d is not None else w.t / self.project.scale
                if d <= max(lim, w.t / self.project.scale * 0.75) and (best is None or d < best[2]):
                    best = ((float(q[0]), float(q[1])), w, d)
        return best

    # ================================================================ tools
    def set_tool(self, tool):
        self.cancel_tool()
        self.tool = tool
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag if tool == 'select' else QGraphicsView.DragMode.NoDrag)
        movable = tool == 'select'
        for it in self.obj_items:
            it.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, movable)
            it.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, movable)
        if not movable:
            self.scene_.clearSelection()
        self.setCursor(Qt.CursorShape.ArrowCursor if tool == 'select' else Qt.CursorShape.CrossCursor)
        self._update_handles()
        self.status.emit(tr('h_' + ('furn' if tool == 'furn' else tool)))
        self.toolChanged.emit(tool)
        if tool == 'furn':
            self._update_preview()

    def cancel_tool(self):
        self.pts = []
        self._clear_preview()

    def _clear_preview(self):
        for it in self.preview:
            self.scene_.removeItem(it)
        self.preview = []

    def _prev(self, path, color=theme.PREVIEW_COLOR, w=1.4, dash=False, brush=None):
        it = QGraphicsPathItem(path)
        it.setPen(cosmetic(color, w, Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine))
        if brush:
            it.setBrush(brush)
        it.setZValue(40)
        self.scene_.addItem(it)
        self.preview.append(it)

    def _cursor_pt(self):
        p = (self._mouse.x(), self._mouse.y())
        t = self.tool
        if t in ('wall', 'poly'):
            return self.snap(p, ref=self.pts[-1] if self.pts else None)
        if t in ('door', 'door2', 'window'):
            if len(self.pts) < 2:
                ref = self.pts[0] if self.pts else None
                q = self.snap(p, on_wall=True)
                if ref is not None and q == (float(p[0]), float(p[1])):
                    q = self.snap(p, ref=ref)
                return q
            return p
        if t == 'stair':
            if len(self.pts) == 1:
                return self.snap(p, ref=self.pts[0])
            return p
        if t == 'calib':
            return self.snap(p, ref=self.pts[0] if self.pts else None)
        return self.snap(p) if t in ('column', 'text', 'level') else p

    def _update_preview(self):
        self._clear_preview()
        if self.gray is None:
            return
        t = self.tool; s = self.project.scale
        cp = self._cursor_pt()
        mk = QPainterPath(); r = 4 / max(self.transform().m11(), 1e-6)
        mk.addEllipse(QPointF(*cp), r, r)
        if t != 'select' and t != 'furn':
            self._prev(mk, w=1.5)
        if t in ('wall', 'poly') and self.pts:
            pts = self.pts + [cp]
            if t == 'wall':
                poly = G.wall_polygon(Wall(pts=pts, t=self.wall_t), s)
                self._prev(qpath_from_polygon(poly), brush=QBrush(QColor(61, 155, 255, 50)))
            else:
                self._prev(qpath_from_lines([pts]))
            L = dist(self.pts[-1], cp) * s
            self.status.emit(f"{tr('h_' + t)}   |   {L:.2f} m")
        elif t in ('door', 'door2') and self.pts:
            if len(self.pts) == 1:
                self._prev(qpath_from_lines([[self.pts[0], cp]]), dash=True)
                self.status.emit(f"{tr('h_' + t)}   |   {dist(self.pts[0], cp) * s:.2f} m")
            else:
                d = self._door_from(self.pts[0], self.pts[1], cp, t == 'door2')
                lv, ar = G.door_geometry(d, s)
                self._prev(qpath_from_lines(lv + ar, [True]*len(lv) + [False]*len(ar)))
        elif t == 'window' and self.pts:
            th = self._wall_t_at(self.pts[0])
            self._prev(qpath_from_lines(G.window_lines(Window(self.pts[0], cp, th), s)))
            self.status.emit(f"{tr('h_window')}   |   {dist(self.pts[0], cp) * s:.2f} m")
        elif t == 'stair' and self.pts:
            if len(self.pts) == 1:
                self._prev(qpath_from_lines([[self.pts[0], cp]]), dash=True)
            else:
                st = self._stair_from(self.pts[0], self.pts[1], cp)
                ln, ar = G.stair_geometry(st, s)
                self._prev(qpath_from_lines(ln + ar))
        elif t == 'calib' and self.pts:
            self._prev(qpath_from_lines([[self.pts[0], cp]]), color='#FFB020', w=2)
        elif t == 'column':
            c = Column(cp[0], cp[1], self.col_size, self._angle_near(cp))
            self._prev(qpath_from_polygon(G.column_polygon(c, s)), brush=QBrush(QColor(61, 155, 255, 90)))
        elif t == 'furn':
            p = (self._mouse.x(), self._mouse.y())
            lay, _, w, d, _, _ = LIB[self.furn_block]
            f = Furn(self.furn_block, p[0], p[1], w, d, self.place_rot, self.place_mirror, lay)
            pls = G.furn_polylines(f, s)
            self._prev(qpath_from_lines([q for q, _, _ in pls], [c for _, c, _ in pls]))

    def _angle_near(self, p):
        pr = self.project_on_wall(p, max_d=self._tol() * 3)
        if pr:
            a, b = pr[1].pts[0], pr[1].pts[-1]
            return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        return self.project.angles[0] if self.project.angles else 0.0

    def _wall_t_at(self, p):
        pr = self.project_on_wall(p, max_d=self._tol() * 2)
        return pr[1].t if pr else self.wall_t

    @staticmethod
    def _door_from(a, b, sp, double):
        u = np.array(b, float) - np.array(a, float)
        n = np.array([-u[1], u[0]])
        side = 1 if np.dot(np.array(sp, float) - np.array(a, float), n) >= 0 else -1
        return Door(tuple(a), tuple(b), side, double)

    @staticmethod
    def _stair_from(a, b, c):
        a, b, c = (np.array(v, float) for v in (a, b, c))
        al = b - a
        u = al / max(np.linalg.norm(al), 1e-9); n = np.array([-u[1], u[0]])
        w = (c - b) @ n
        return Stair(tuple(a), tuple(al), tuple(n * w))

    def _click(self, p):
        t = self.tool; pr = self.project; s = pr.scale
        if t in ('wall', 'poly'):
            if self.pts and dist(self.pts[-1], p) < 0.5:
                return
            self.pts.append(p)
        elif t in ('door', 'door2'):
            self.pts.append(p)
            if len(self.pts) == 3:
                d = self._door_from(self.pts[0], self.pts[1], self.pts[2], t == 'door2')
                if dist(d.p0, d.p1) > 2:
                    self.push_undo(); pr.doors.append(d); self.commit()
                self.pts = []
        elif t == 'window':
            self.pts.append(p)
            if len(self.pts) == 2:
                if dist(*self.pts) > 2:
                    self.push_undo(); pr.windows.append(Window(self.pts[0], self.pts[1], self._wall_t_at(self.pts[0]))); self.commit()
                self.pts = []
        elif t == 'stair':
            self.pts.append(p)
            if len(self.pts) == 3:
                st = self._stair_from(*self.pts)
                if np.linalg.norm(st.along) > 3 and np.linalg.norm(st.across) > 3:
                    self.push_undo(); pr.stairs.append(st); self.commit()
                self.pts = []
        elif t == 'column':
            self.push_undo(); pr.columns.append(Column(p[0], p[1], self.col_size, self._angle_near(p))); self.commit()
        elif t == 'furn':
            p = (self._mouse.x(), self._mouse.y())
            lay, _, w, d, _, _ = LIB[self.furn_block]
            f = Furn(self.furn_block, p[0], p[1], w, d, self.place_rot, self.place_mirror, lay)
            if self.fit_mode:
                fitted = self.fit_block_at(p, f)
                if fitted:
                    f = fitted
            self.push_undo()
            pr.furniture.append(f)
            self.commit()
        elif t == 'text':
            txt, ok = QInputDialog.getText(self, tr('t_text'), tr('text_ask'))
            if ok and txt.strip():
                self.push_undo(); pr.texts.append(Text(p[0], p[1], txt.strip(), 0.30)); self.commit()
        elif t == 'level':
            txt, ok = QInputDialog.getText(self, tr('t_level'), tr('level_ask'), text='+0.00')
            if ok and txt.strip():
                self.push_undo(); pr.levels.append(Level(p[0], p[1], txt.strip())); self.commit()
        elif t == 'calib':
            self.pts.append(p)
            if len(self.pts) == 2:
                dpx = dist(*self.pts)
                self.pts = []
                self._clear_preview()
                if dpx < 2:
                    return
                val, ok = QInputDialog.getDouble(self, tr('calib_title'), tr('calib_ask'), round(dpx * s, 3), 0.01, 10000, 3)
                if ok and val > 0:
                    self.push_undo()
                    pr.scale = val / dpx
                    pr.scale_calibrated = True
                    self.commit()
                    self.scaleChanged.emit()
                    self.status.emit(tr('calib_done').format(pr.scale))

    # ---------------------------------------------------------------- fit block to drawing
    def _get_wall_mask(self):
        if self._wall_mask is None:
            import cv2
            m = np.zeros(self.gray.shape, np.uint8)
            U = G.walls_union(self.project, cut_windows=False)
            if U is not None:
                for gp in getattr(U.buffer(2.0), 'geoms', [U.buffer(2.0)]):
                    if gp.geom_type == 'Polygon':
                        cv2.fillPoly(m, [np.round(np.array(gp.exterior.coords)).astype(np.int32)], 255)
                        for r in gp.interiors:
                            cv2.fillPoly(m, [np.round(np.array(r.coords)).astype(np.int32)], 0)
            for c in self.project.columns:
                cv2.circle(m, (int(c.x), int(c.y)), int(c.s / self.project.scale * 0.8), 255, -1)
            self._wall_mask = m
        return self._wall_mask

    def fit_block_at(self, p, f):
        """Measure the drawn shape under the click and size/rotate the block to it."""
        import cv2
        s = self.project.scale
        R = int(max(f.w, f.d) / s * 1.4) + 6
        x0, y0 = int(max(0, p[0] - R)), int(max(0, p[1] - R))
        x1, y1 = int(min(self.gray.shape[1], p[0] + R)), int(min(self.gray.shape[0], p[1] + R))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None
        crop = self.gray[y0:y1, x0:x1]
        for use_mask in (False, True):
            ink = (cv2.adaptiveThreshold(crop, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12)
                   | (crop < 150)).astype(np.uint8)
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            if use_mask:
                ink[self._get_wall_mask()[y0:y1, x0:x1] > 0] = 0
            pick = self._pick_shape(ink, p, x0, y0, f, R)
            if pick is not None:
                break
        if pick is None:
            return None
        ys, xs = np.nonzero(pick)
        (rx, ry), (rw, rh), ang = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        return self._furn_from_rect(f, x0 + rx, y0 + ry, rw, rh, ang)

    def _pick_shape(self, ink, p, x0, y0, f, R):
        import cv2
        s = self.project.scale
        cx, cy = int(round(p[0] - x0)), int(round(p[1] - y0))
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        limit = max(f.w, f.d) / s * 1.9
        pick = None
        if ink[cy, cx] == 0:
            # flood the white area under the click; if it is enclosed, its outline is the piece
            ff = (1 - ink).astype(np.uint8)
            n2, lab2 = cv2.connectedComponents(ff, connectivity=4)
            reg = lab2 == lab2[cy, cx]
            ys_, xs_ = np.nonzero(reg)
            enclosed = not (xs_.min() == 0 or ys_.min() == 0 or xs_.max() == ink.shape[1]-1 or ys_.max() == ink.shape[0]-1)
            if enclosed:
                ring = cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                ids = np.unique(lab[ring & (ink > 0)]); ids = ids[ids > 0]
                if len(ids):
                    big = max(ids, key=lambda i: st[i][4])
                    bx, by, bw, bh, _ = st[big]
                    if max(bw, bh) <= limit:
                        pick = lab == big
                if pick is None and max(np.ptp(xs_), np.ptp(ys_)) <= limit:
                    pick = cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if pick is None:
            best = None
            for i in range(1, n):
                bx, by, bw, bh, area = st[i]
                if max(bw, bh) * s < 0.15 or max(bw, bh) > limit:
                    continue
                dx = max(bx - cx, 0, cx - (bx + bw)); dy = max(by - cy, 0, cy - (by + bh))
                dd = math.hypot(dx, dy)
                if dd < R * 0.35 and (best is None or dd < best[0]):
                    best = (dd, i)
            if best is None:
                return None
            pick = lab == best[1]
        return pick

    def _furn_from_rect(self, f, rx, ry, rw, rh, ang):
        s = self.project.scale
        lo, hi = sorted((rw * s, rh * s))
        if lo < 0.1:
            return None
        long_ang = ang if rw >= rh else ang + 90
        bw_, bd_ = LIB[f.block][2], LIB[f.block][3]
        if bw_ >= bd_:
            w, d = hi, lo; cand = [long_ang + 90, long_ang - 90]
        else:
            w, d = lo, hi; cand = [long_ang, long_ang + 180]
        if abs(bw_ - bd_) < 0.12 * max(bw_, bd_):           # square-ish: keep user's facing, snap to shape
            w = d = (lo + hi) / 2 if 'ROUND' in f.block or 'OFFICE' in f.block else hi
            cand = [long_ang + 90*k for k in range(4)]
        diff = lambda a: abs((a - self.place_rot + 180) % 360 - 180)
        rot = min(cand, key=diff) % 360
        return Furn(f.block, float(rx), float(ry), round(w, 3), round(d, 3), rot, f.mirror, f.layer)

    def finish_polyline(self, close=False):
        t = self.tool
        if t == 'wall' and len(self.pts) >= 2:
            self.push_undo(); self.project.walls.append(Wall(list(self.pts), self.wall_t)); self.pts = []; self.commit()
        elif t == 'poly' and len(self.pts) >= 2:
            self.push_undo(); self.project.polys.append(Poly(list(self.pts), self.poly_layer, close)); self.pts = []; self.commit()
        self._update_preview()

    # ================================================================ events
    def wheelEvent(self, e):
        f = 1.18 if e.angleDelta().y() > 0 else 1 / 1.18
        cur = self.transform().m11()
        if 0.02 < cur * f < 80:
            self.scale(f, f)
        self._update_preview()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton or (self._space and e.button() == Qt.MouseButton.LeftButton):
            self._pan = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self.gray is None:
            return
        sp = self.mapToScene(e.position().toPoint())
        self._mouse = sp
        if self.tool == 'select':
            self._press_pos = sp
            super().mousePressEvent(e)
            return
        if e.button() == Qt.MouseButton.RightButton:
            if self.tool in ('wall', 'poly') and len(self.pts) >= 2:
                self.finish_polyline()
            else:
                self.cancel_tool(); self._update_preview()
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._click(self._cursor_pt() if self.tool != 'furn' else (sp.x(), sp.y()))
            self._update_preview()

    def mouseDoubleClickEvent(self, e):
        if self.tool in ('wall', 'poly'):
            self.finish_polyline()
            return
        super().mouseDoubleClickEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        sp = self.mapToScene(e.position().toPoint())
        self._mouse = sp
        if self.image_size:
            s = self.project.scale
            self.coords.emit(sp.x() * s, (self.image_size[1] - sp.y()) * s)
        if self.tool == 'select':
            super().mouseMoveEvent(e)
        else:
            self._update_preview()

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.setCursor(Qt.CursorShape.ArrowCursor if self.tool == 'select' else Qt.CursorShape.CrossCursor)
            return
        if self.tool != 'select':
            return
        super().mouseReleaseEvent(e)
        # handles moved?
        if any(h.pos() != QPointF(*self._handle_orig(h)) for h in self.handles):
            self._commit_handles()
            return
        moved = [it for it in self.scene_.selectedItems() if isinstance(it, ObjItem) and (abs(it.pos().x()) > 0.01 or abs(it.pos().y()) > 0.01)]
        if moved:
            self.push_undo()
            sel = []
            for it in moved:
                self.translate(it.kind, it.idx, it.pos().x(), it.pos().y())
                sel.append((it.kind, it.idx))
            self.commit(keep=sel)
            return
        self._update_handles()
        self._emit_selection()

    def _handle_orig(self, h):
        o = self.get(h.kind, h.idx)
        if h.kind in ('walls', 'polys'):
            return o.pts[h.vi]
        return o.p0 if h.vi == 0 else o.p1

    def keyPressEvent(self, e):
        k = e.key(); mods = e.modifiers()
        if k == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if k == Qt.Key.Key_Escape:
            if self.pts:
                self.cancel_tool(); self._update_preview()
            else:
                self.set_tool('select')
            return
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finish_polyline(); return
        if self.tool == 'poly' and k == Qt.Key.Key_C and not mods:
            self.finish_polyline(close=True); return
        if self.tool == 'furn':
            if k == Qt.Key.Key_R:
                self.place_rot = (self.place_rot + 90) % 360
            elif k == Qt.Key.Key_E:
                self.place_rot = (self.place_rot + 15) % 360
            elif k == Qt.Key.Key_Q:
                self.place_rot = (self.place_rot - 15) % 360
            elif k == Qt.Key.Key_M:
                self.place_mirror = not self.place_mirror
            self._update_preview()
            return
        if self.tool == 'select' and not (mods & Qt.KeyboardModifier.ControlModifier):
            if k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self.delete_selected(); return
            if k == Qt.Key.Key_R:
                self.transform_selected(rot=90); return
            if k == Qt.Key.Key_E:
                self.transform_selected(rot=15); return
            if k == Qt.Key.Key_Q:
                self.transform_selected(rot=-15); return
            if k == Qt.Key.Key_M:
                self.transform_selected(mirror=True); return
            if k == Qt.Key.Key_F:
                self.transform_selected(flip=True); return
            if k == Qt.Key.Key_H:
                self.transform_selected(hinge=True); return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.setCursor(Qt.CursorShape.ArrowCursor if self.tool == 'select' else Qt.CursorShape.CrossCursor)
            return
        super().keyReleaseEvent(e)


def _rot_pt(p, c, deg):
    a = math.radians(deg); x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x*math.cos(a) - y*math.sin(a), c[1] + x*math.sin(a) + y*math.cos(a))


def _rot_vec(v, deg):
    a = math.radians(deg)
    return (v[0]*math.cos(a) - v[1]*math.sin(a), v[0]*math.sin(a) + v[1]*math.cos(a))
