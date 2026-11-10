"""Side panels: layers, block library, properties."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPixmap, QIcon, QPainter
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QTreeWidget, QTreeWidgetItem, QLineEdit, QSlider, QFormLayout, QDoubleSpinBox,
                               QCheckBox, QComboBox, QPushButton, QSizePolicy, QFrame, QLineEdit as _LE)

from ..blocks import LIB, CATEGORIES
from . import theme
from .i18n import tr, layer_name, LANG
from .icons import block_icon

SP = theme.SP


def _swatch(color):
    pm = QPixmap(14, 14); pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color)); p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(1, 1, 12, 12, 3, 3); p.end()
    return QIcon(pm)


class LayersPanel(QWidget):
    visibilityChanged = Signal(str, bool)
    opacityChanged = Signal(float)

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.setObjectName('panel')
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['m'], SP['m'], SP['m'], SP['m']); lay.setSpacing(SP['s'])
        self.list = QListWidget()
        self.list.setSpacing(1)
        for code in ['A-SCAN', 'A-TRACE', 'A-DIM', 'A-WALL', 'A-COLUMN', 'A-WINDOW', 'A-DOOR', 'A-STAIR', 'A-FURN',
                     'A-SANITARY', 'A-KITCHEN', 'A-COURT', 'A-TEXT', 'A-LEVEL']:
            it = QListWidgetItem(_swatch(theme.LAYER_COLORS[code]), f"{layer_name(code)}")
            it.setToolTip(code)
            it.setData(Qt.ItemDataRole.UserRole, code)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if view.visible.get(code, True) else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.list.itemChanged.connect(lambda it: self.visibilityChanged.emit(
            it.data(Qt.ItemDataRole.UserRole), it.checkState() == Qt.CheckState.Checked))
        lay.addWidget(self.list, 1)
        cap = QLabel('شفافية صورة السكان' if LANG['cur'] == 'ar' else 'Scan opacity')
        cap.setObjectName('caption')
        lay.addWidget(cap)
        self.op = QSlider(Qt.Orientation.Horizontal); self.op.setRange(0, 100); self.op.setValue(int(view.scan_opacity * 100))
        self.op.valueChanged.connect(lambda v: self.opacityChanged.emit(v / 100))
        lay.addWidget(self.op)


class LibraryPanel(QWidget):
    blockChosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('panel')
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['m'], SP['m'], SP['m'], SP['m']); lay.setSpacing(SP['s'])
        self.search = QLineEdit(); self.search.setPlaceholderText(tr('search')); self.search.setClearButtonEnabled(True)
        lay.addWidget(self.search)
        self.tree = QTreeWidget(); self.tree.setHeaderHidden(True); self.tree.setIconSize(QSize(34, 34))
        self.tree.setIndentation(10)
        ar = LANG['cur'] == 'ar'
        self.items = []
        for cat, a, e in CATEGORIES:
            top = QTreeWidgetItem([a if ar else e])
            f = top.font(0); f.setBold(True); top.setFont(0, f)
            top.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(top)
            for name, (layer, c, w, d, an, en) in LIB.items():
                if c != cat:
                    continue
                it = QTreeWidgetItem([f"{an if ar else en}\n\u2066{w:.2f} × {d:.2f} m\u2069"])
                it.setIcon(0, block_icon(name))
                it.setData(0, Qt.ItemDataRole.UserRole, name)
                it.setToolTip(0, name)
                top.addChild(it)
                self.items.append((it, f"{an} {en} {name}".lower()))
            top.setExpanded(True)
        self.tree.itemClicked.connect(self._clicked)
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.tree, 1)

    def _clicked(self, it, col):
        name = it.data(0, Qt.ItemDataRole.UserRole)
        if name:
            self.blockChosen.emit(name)

    def _filter(self, txt):
        txt = txt.lower().strip()
        for it, key in self.items:
            it.setHidden(bool(txt) and txt not in key)


class PropertiesPanel(QWidget):
    """Edits the selected object(s)."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.setObjectName('panel')
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(SP['m'], SP['m'], SP['m'], SP['m']); self.lay.setSpacing(SP['s'])
        self.lay.addStretch(1)
        self.body = None
        self.show_for([])

    def _clear(self):
        if self.body is not None:
            self.body.setParent(None); self.body.deleteLater()
        self.body = QWidget(); self.body.setObjectName('panel')
        self.form = QFormLayout(self.body); self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(SP['s']); self.form.setLabelAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.lay.insertWidget(0, self.body)

    def _spin(self, val, lo, hi, step=0.01, dec=3):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val)
        s.setKeyboardTracking(False)
        return s

    def show_for(self, sel):
        self._clear()
        v = self.view; p = v.project
        if not sel:
            lab = QLabel(tr('nothing_sel')); lab.setObjectName('caption')
            self.form.addRow(lab)
            return
        kinds = {k for k, _ in sel}
        head = QLabel(f"{tr('n_selected')}: {len(sel)}"); head.setObjectName('section')
        self.form.addRow(head)
        if kinds == {'walls'}:
            t = self._spin(p.walls[sel[0][1]].t, 0.03, 2.0)
            t.valueChanged.connect(lambda val: self._set(sel, 't', val))
            self.form.addRow(tr('thickness'), t)
        elif kinds == {'furniture'}:
            f = p.furniture[sel[0][1]]
            cb = QComboBox()
            for name, row in LIB.items():
                cb.addItem(f"{row[4] if LANG['cur'] == 'ar' else row[5]}  ({name})", name)
            cb.setCurrentIndex(max(0, cb.findData(f.block)))
            cb.currentIndexChanged.connect(lambda i: self._set_block(sel, cb.itemData(i)))
            self.form.addRow(tr('block'), cb)
            w = self._spin(f.w, 0.05, 20); w.valueChanged.connect(lambda val: self._set(sel, 'w', val))
            d = self._spin(f.d, 0.05, 20); d.valueChanged.connect(lambda val: self._set(sel, 'd', val))
            r = self._spin(f.rot, -360, 360, 1, 1); r.valueChanged.connect(lambda val: self._set(sel, 'rot', val % 360))
            self.form.addRow(tr('width'), w); self.form.addRow(tr('depth'), d); self.form.addRow(tr('rotation'), r)
            m = QCheckBox(tr('mirror')); m.setChecked(f.mirror); m.toggled.connect(lambda val: self._set(sel, 'mirror', val))
            self.form.addRow('', m)
            lc = QComboBox()
            for code in ['A-FURN', 'A-SANITARY', 'A-KITCHEN', 'A-COURT']:
                lc.addItem(layer_name(code), code)
            lc.setCurrentIndex(max(0, lc.findData(f.layer)))
            lc.currentIndexChanged.connect(lambda i: self._set(sel, 'layer', lc.itemData(i)))
            self.form.addRow(tr('layer'), lc)
        elif kinds == {'doors'}:
            d = p.doors[sel[0][1]]
            w = QLabel(f"{((d.p1[0]-d.p0[0])**2 + (d.p1[1]-d.p0[1])**2) ** .5 * p.scale:.2f} m")
            self.form.addRow(tr('width'), w)
            cb = QCheckBox(tr('double')); cb.setChecked(d.double); cb.toggled.connect(lambda val: self._set(sel, 'double', val))
            self.form.addRow('', cb)
            row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(SP['s'])
            b1 = QPushButton(tr('flip')); b1.clicked.connect(lambda: v.transform_selected(flip=True))
            b2 = QPushButton(tr('hinge')); b2.clicked.connect(lambda: v.transform_selected(hinge=True))
            hl.addWidget(b1); hl.addWidget(b2)
            self.form.addRow(row)
        elif kinds == {'windows'}:
            t = self._spin(p.windows[sel[0][1]].t, 0.03, 2.0)
            t.valueChanged.connect(lambda val: self._set(sel, 't', val))
            self.form.addRow(tr('depth'), t)
        elif kinds == {'columns'}:
            c = p.columns[sel[0][1]]
            s = self._spin(c.s, 0.05, 3.0); s.valueChanged.connect(lambda val: self._set(sel, 's', val))
            a = self._spin(c.a, -360, 360, 1, 1); a.valueChanged.connect(lambda val: self._set(sel, 'a', val))
            self.form.addRow(tr('size'), s); self.form.addRow(tr('rotation'), a)
        elif kinds == {'stairs'}:
            t = self._spin(p.stairs[sel[0][1]].tread, 0.15, 0.6)
            t.valueChanged.connect(lambda val: self._set(sel, 'tread', val))
            self.form.addRow(tr('tread'), t)
        elif kinds == {'texts'}:
            o = p.texts[sel[0][1]]
            e = QLineEdit(o.text); e.editingFinished.connect(lambda: self._set(sel, 'text', e.text()))
            h = self._spin(o.h, 0.05, 5); h.valueChanged.connect(lambda val: self._set(sel, 'h', val))
            self.form.addRow(tr('text'), e); self.form.addRow(tr('height'), h)
        elif kinds == {'levels'}:
            o = p.levels[sel[0][1]]
            e = QLineEdit(o.value); e.editingFinished.connect(lambda: self._set(sel, 'value', e.text()))
            self.form.addRow(tr('value'), e)
        elif kinds == {'polys'}:
            o = p.polys[sel[0][1]]
            lc = QComboBox()
            for code in ['A-FURN', 'A-SANITARY', 'A-KITCHEN', 'A-COURT', 'A-STAIR', 'A-WALL']:
                lc.addItem(layer_name(code), code)
            lc.setCurrentIndex(max(0, lc.findData(o.layer)))
            lc.currentIndexChanged.connect(lambda i: self._set(sel, 'layer', lc.itemData(i)))
            self.form.addRow(tr('layer'), lc)
            c = QCheckBox(tr('closed')); c.setChecked(o.closed); c.toggled.connect(lambda val: self._set(sel, 'closed', val))
            self.form.addRow('', c)

    def _set(self, sel, attr, val):
        v = self.view
        v.push_undo()
        for k, i in sel:
            o = v.get(k, i)
            if hasattr(o, attr):
                setattr(o, attr, val)
        v.commit(keep=sel)

    def _set_block(self, sel, name):
        v = self.view
        v.push_undo()
        for k, i in sel:
            o = v.get(k, i)
            if o.block != name:
                o.block = name
                o.layer = LIB[name][0]
        v.commit(keep=sel)
