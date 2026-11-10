"""Start-up splash screen with the author's name."""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QPainter, QColor, QLinearGradient, QFont, QPen, QGuiApplication
from PySide6.QtWidgets import QSplashScreen

from . import theme

APP_NAME = 'Plan2CAD'
AUTHOR_EN = 'Essam Nawafleh'
AUTHOR_AR = 'عصام النوافلة'
VERSION = '1.0'


def _font(pt, bold=False):
    f = QFont(); f.setFamilies(theme.FONT_FAMILIES); f.setPointSizeF(pt)
    if bold:
        f.setWeight(QFont.Weight.Bold)
    return f


def make_splash():
    W, H = 640, 360
    dpr = QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() else 1.0
    pm = QPixmap(int(W * dpr), int(H * dpr)); pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    g = QLinearGradient(0, 0, W, H); g.setColorAt(0, QColor('#1B2533')); g.setColorAt(1, QColor('#14181E'))
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(g); p.drawRoundedRect(QRectF(0, 0, W, H), 16, 16)
    # faint blueprint grid
    p.setPen(QPen(QColor(61, 155, 255, 22), 1))
    for x in range(0, W, 24):
        p.drawLine(x, 0, x, H)
    for y in range(0, H, 24):
        p.drawLine(0, y, W, y)
    # logo: plan glyph
    p.setPen(QPen(QColor('#3D9BFF'), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(48, 52, 72, 64)); p.drawLine(48, 86, 88, 86); p.drawLine(88, 52, 88, 86)
    p.setPen(QPen(QColor('#3D9BFF'), 2.5)); p.drawArc(QRectF(68, 86, 40, 40), 90 * 16, 90 * 16)
    p.setPen(QColor('#FFFFFF')); p.setFont(_font(30, True)); p.drawText(QRectF(140, 46, 460, 50), Qt.AlignmentFlag.AlignVCenter, APP_NAME)
    p.setPen(QColor('#AEB2B9')); p.setFont(_font(11))
    p.drawText(QRectF(142, 92, 460, 26), Qt.AlignmentFlag.AlignVCenter, 'Scanned plans → layered AutoCAD DXF')
    # author block
    p.setPen(QPen(QColor(255, 255, 255, 40), 1)); p.drawLine(48, 200, W - 48, 200)
    p.setPen(QColor('#8A8F97')); p.setFont(_font(9))
    p.drawText(QRectF(48, 214, W - 96, 20), Qt.AlignmentFlag.AlignLeft, 'Developed by')
    p.setPen(QColor('#FFFFFF')); p.setFont(_font(20, True))
    p.drawText(QRectF(48, 236, W - 96, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, AUTHOR_EN)
    p.setFont(_font(20, True))
    p.drawText(QRectF(48, 236, W - 96, 40), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, AUTHOR_AR)
    p.setPen(QColor('#3D9BFF')); p.setFont(_font(9))
    p.drawText(QRectF(48, H - 44, W - 96, 20), Qt.AlignmentFlag.AlignLeft, f'Version {VERSION}')
    p.end()
    sp = QSplashScreen(pm, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
    sp.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    return sp
