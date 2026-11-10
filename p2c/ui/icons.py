"""Painted line icons (no external icon font needed)."""
import math
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QPainterPath, QBrush

from ..blocks import block_polylines

FG = QColor('#E8E9EB')


def _make(draw, size=24, color=FG):
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(2, 2)
    pen = QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    draw(p, size)
    p.end()
    pm.setDevicePixelRatio(2)
    return QIcon(pm)


def _arc(p, cx, cy, r, a0, span):
    p.drawArc(QRectF(cx - r, cy - r, 2*r, 2*r), int(a0*16), int(span*16))


def icon(name, color=FG):
    def d(p, s):
        if name == 'select':
            path = QPainterPath(); path.moveTo(6, 4); path.lineTo(6, 19); path.lineTo(10, 15); path.lineTo(13, 21)
            path.lineTo(15.5, 20); path.lineTo(12.5, 14); path.lineTo(18, 14); path.closeSubpath(); p.drawPath(path)
        elif name == 'wall':
            p.drawRect(QRectF(3, 9, 18, 6)); p.drawLine(QPointF(3, 12), QPointF(21, 12)) if False else None
            p.setBrush(QBrush(color)); p.drawRect(QRectF(3, 9, 7, 6)); p.setBrush(Qt.BrushStyle.NoBrush)
        elif name == 'door':
            p.drawLine(QPointF(4, 20), QPointF(20, 20)); p.drawLine(QPointF(5, 20), QPointF(5, 5)); _arc(p, 5, 20, 15, 0, 90)
        elif name == 'door2':
            p.drawLine(QPointF(2, 20), QPointF(22, 20)); p.drawLine(QPointF(3, 20), QPointF(3, 11)); p.drawLine(QPointF(21, 20), QPointF(21, 11))
            _arc(p, 3, 20, 9, 0, 90); _arc(p, 21, 20, 9, 90, 90)
        elif name == 'window':
            p.drawRect(QRectF(3, 9, 18, 6)); p.drawLine(QPointF(3, 11.5), QPointF(21, 11.5)); p.drawLine(QPointF(3, 12.5), QPointF(21, 12.5))
        elif name == 'column':
            p.setBrush(QBrush(color)); p.drawRect(QRectF(7, 7, 10, 10)); p.setBrush(Qt.BrushStyle.NoBrush)
        elif name == 'stair':
            p.drawRect(QRectF(5, 3, 14, 18))
            for y in (6.6, 10.2, 13.8, 17.4): p.drawLine(QPointF(5, y), QPointF(19, y))
            p.drawLine(QPointF(12, 19), QPointF(12, 5))
        elif name == 'poly':
            path = QPainterPath(); path.moveTo(3, 18); path.lineTo(9, 7); path.lineTo(15, 15); path.lineTo(21, 5); p.drawPath(path)
            for x, y in ((3, 18), (9, 7), (15, 15), (21, 5)): p.drawEllipse(QPointF(x, y), 1.5, 1.5)
        elif name == 'text':
            p.drawLine(QPointF(5, 5), QPointF(19, 5)); p.drawLine(QPointF(12, 5), QPointF(12, 20)); p.drawLine(QPointF(9, 20), QPointF(15, 20))
        elif name == 'level':
            p.drawEllipse(QPointF(9, 12), 4.5, 4.5); p.drawLine(QPointF(3, 12), QPointF(21, 12)); p.drawLine(QPointF(9, 6), QPointF(9, 18))
        elif name == 'calib':
            p.drawRect(QRectF(2, 9, 20, 7))
            for i, x in enumerate(range(5, 21, 3)): p.drawLine(QPointF(x, 9), QPointF(x, 12 if i % 2 else 13.5))
        elif name == 'auto':
            p.drawLine(QPointF(5, 19), QPointF(15, 9))
            for cx, cy, r in ((17, 6, 2.2), (20, 12, 1.4), (11, 4, 1.4)):
                p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy)); p.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))
        elif name == 'open':
            path = QPainterPath(); path.moveTo(3, 6); path.lineTo(9, 6); path.lineTo(11, 8); path.lineTo(21, 8); path.lineTo(21, 19)
            path.lineTo(3, 19); path.closeSubpath(); p.drawPath(path)
        elif name == 'image':
            p.drawRect(QRectF(3, 5, 18, 14)); path = QPainterPath(); path.moveTo(4, 17); path.lineTo(10, 11); path.lineTo(14, 15)
            path.lineTo(16, 13); path.lineTo(20, 17); p.drawPath(path); p.drawEllipse(QPointF(16, 9), 1.5, 1.5)
        elif name == 'save':
            p.drawRect(QRectF(4, 4, 16, 16)); p.drawRect(QRectF(8, 4, 8, 5)); p.drawRect(QRectF(7, 13, 10, 7))
        elif name == 'export':
            p.drawLine(QPointF(12, 3), QPointF(12, 15)); p.drawLine(QPointF(8, 7), QPointF(12, 3)); p.drawLine(QPointF(16, 7), QPointF(12, 3))
            path = QPainterPath(); path.moveTo(4, 13); path.lineTo(4, 20); path.lineTo(20, 20); path.lineTo(20, 13); p.drawPath(path)
        elif name == 'undo':
            path = QPainterPath(); path.moveTo(8, 6); path.lineTo(4, 10); path.lineTo(8, 14); p.drawPath(path)
            path2 = QPainterPath(); path2.moveTo(4, 10); path2.cubicTo(18, 10, 21, 13, 17, 19); p.drawPath(path2)
        elif name == 'redo':
            path = QPainterPath(); path.moveTo(16, 6); path.lineTo(20, 10); path.lineTo(16, 14); p.drawPath(path)
            path2 = QPainterPath(); path2.moveTo(20, 10); path2.cubicTo(6, 10, 3, 13, 7, 19); p.drawPath(path2)
        elif name == 'fit':
            for x, y, dx, dy in ((4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)):
                p.drawLine(QPointF(x, y), QPointF(x + 5*dx, y)); p.drawLine(QPointF(x, y), QPointF(x, y + 5*dy))
        elif name == 'eye':
            path = QPainterPath(); path.moveTo(2, 12); path.quadTo(12, 3, 22, 12); path.quadTo(12, 21, 2, 12); p.drawPath(path)
            p.drawEllipse(QPointF(12, 12), 3, 3)
    return _make(d, color=color)


def block_icon(name, size=40, color=QColor('#E8E9EB')):
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(2, 2)
    p.setPen(QPen(color, 1.1))
    m = size * 0.12; k = size - 2*m
    for pts, closed, dashed in block_polylines(name):
        path = QPainterPath()
        for i, (x, y) in enumerate(pts):
            q = QPointF(m + (x + .5) * k, m + (.5 - y) * k)   # front (-y) at the bottom
            path.moveTo(q) if i == 0 else path.lineTo(q)
        if closed:
            path.closeSubpath()
        p.drawPath(path)
    p.end()
    pm.setDevicePixelRatio(2)
    return QIcon(pm)
