"""Dialogs: import/prepare image, auto-detect settings, export options."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox, QCheckBox,
                               QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QGridLayout, QDialogButtonBox,
                               QProgressBar, QWidget, QSlider)

from . import theme
from .i18n import tr

SP = theme.SP


def _buttons(dlg, ok_text=None):
    box = QDialogButtonBox()
    ok = box.addButton(ok_text or tr('ok'), QDialogButtonBox.ButtonRole.AcceptRole)
    ok.setObjectName('primary')
    cancel = box.addButton(tr('cancel'), QDialogButtonBox.ButtonRole.RejectRole)
    box.accepted.connect(dlg.accept); box.rejected.connect(dlg.reject)
    return box


def _title(text):
    l = QLabel(text); l.setObjectName('h1')
    return l


class ImportDialog(QDialog):
    def __init__(self, scale, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('import_title'))
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['xl'], SP['xl'], SP['xl'], SP['xl']); lay.setSpacing(SP['l'])
        lay.addWidget(_title(tr('import_title')))
        form = QFormLayout(); form.setSpacing(SP['m'])
        self.rot = QComboBox()
        for a in (0, 90, 180, 270):
            self.rot.addItem(f"{a}°", a)
        form.addRow(tr('rotate'), self.rot)
        self.persp = QComboBox()
        self.persp.addItem(tr('persp_auto'), 'auto'); self.persp.addItem(tr('persp_off'), 'off')
        form.addRow(tr('perspective'), self.persp)
        self.orient = QCheckBox(tr('auto_orient')); self.orient.setChecked(False)
        self.deskew = QCheckBox(tr('deskew')); self.deskew.setChecked(True)
        self.clean = QCheckBox(tr('clean_bg')); self.clean.setChecked(True)
        self.denoise = QCheckBox(tr('denoise_auto')); self.denoise.setChecked(True)
        for c in (self.orient, self.deskew, self.clean, self.denoise):
            form.addRow('', c)
        self.known = QCheckBox(tr('scale_known')); self.known.setChecked(False)
        self.scale = QDoubleSpinBox(); self.scale.setDecimals(5); self.scale.setRange(0.00001, 10); self.scale.setValue(scale)
        self.scale.setSingleStep(0.001); self.scale.setEnabled(False)
        self.known.toggled.connect(self.scale.setEnabled)
        form.addRow('', self.known)
        form.addRow(tr('scale_m_px'), self.scale)
        lay.addLayout(form)
        note = QLabel(tr('scale_note')); note.setObjectName('hint'); note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(_buttons(self))

    def values(self):
        return dict(rotate=self.rot.currentData(), deskew=self.deskew.isChecked(), clean=self.clean.isChecked(),
                    perspective=self.persp.currentData(), auto_orient=self.orient.isChecked(),
                    denoise='auto' if self.denoise.isChecked() else 'off',
                    scale=self.scale.value(), scale_known=self.known.isChecked())


class CompareDialog(QDialog):
    """Original input next to the processed working image, with the list of processing steps."""

    def __init__(self, original, processed, steps, parent=None):
        super().__init__(parent)
        from PySide6.QtGui import QImage, QPixmap
        import cv2
        self.setWindowTitle(tr('compare_title')); self.resize(1100, 700)
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['l'], SP['l'], SP['l'], SP['l'])
        row = QHBoxLayout(); lay.addLayout(row, 1)

        def pix(img):
            if img is None:
                return QPixmap()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            f = 1000.0 / max(img.shape[:2])
            if f < 1:
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qi.copy())
        for title, img in ((tr('original'), original), (tr('processed'), processed)):
            col = QVBoxLayout(); cap = QLabel(title); cap.setObjectName('section'); col.addWidget(cap)
            lab = QLabel(); lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setPixmap(pix(img).scaled(520, 600, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
            col.addWidget(lab, 1); row.addLayout(col, 1)
        lines = []
        for st in steps:
            mark = '✔' if st.get('applied') else '–'
            lines.append(f"{mark} {st.get('step')}: {st.get('reason', '')}")
        info = QLabel(tr('pp_steps') + '\n' + '\n'.join(lines)); info.setObjectName('hint'); info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(info)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close); box.rejected.connect(self.reject)
        lay.addWidget(box)


class ValidationDialog(QDialog):
    """Validation report: severity, message, 'show' and 'fix' per issue."""
    SEV_COL = {'critical': '#E5484D', 'major': '#F5A524', 'minor': '#3E9BFF', 'info': '#8B8D98'}

    def __init__(self, issues, on_show, on_fix, parent=None, extra=''):
        super().__init__(parent)
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
        from .i18n import LANG
        self.setWindowTitle(tr('validation')); self.resize(720, 560)
        self.on_show, self.on_fix = on_show, on_fix
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['l'], SP['l'], SP['l'], SP['l'])
        cnt = {k: 0 for k in self.SEV_COL}
        for d in issues:
            cnt[d['severity']] += 1
        head = QLabel('   '.join(f"{tr('sev_' + k)}: {v}" for k, v in cnt.items()) + (('\n' + extra) if extra else ''))
        head.setObjectName('section'); head.setWordWrap(True); lay.addWidget(head)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels([tr('severity'), tr('issue')]); self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 110)
        self.issues = issues
        ar = LANG['cur'] == 'ar'
        for i, d in enumerate(issues):
            it = QTreeWidgetItem([tr('sev_' + d['severity']), d['msg_ar'] if ar else d['msg_en']])
            from PySide6.QtGui import QColor
            it.setForeground(0, QColor(self.SEV_COL[d['severity']]))
            it.setData(0, Qt.ItemDataRole.UserRole, i)
            if d.get('fix'):
                it.setText(1, it.text(1) + '   ⚙')
            self.tree.addTopLevelItem(it)
        self.tree.itemDoubleClicked.connect(lambda it, c: self._show(it))
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        b_show = QPushButton(tr('show')); b_show.clicked.connect(lambda: self._show(self.tree.currentItem()))
        b_fix = QPushButton(tr('fix')); b_fix.clicked.connect(self._fix)
        b_close = QPushButton(tr('close')); b_close.clicked.connect(self.accept)
        note = QLabel(tr('fix_note')); note.setObjectName('hint'); note.setWordWrap(True)
        row.addWidget(b_show); row.addWidget(b_fix); row.addWidget(note, 1); row.addWidget(b_close)
        lay.addLayout(row)

    def _show(self, it):
        if it is not None:
            self.on_show(self.issues[it.data(0, Qt.ItemDataRole.UserRole)])

    def _fix(self):
        it = self.tree.currentItem()
        if it is None:
            return
        d = self.issues[it.data(0, Qt.ItemDataRole.UserRole)]
        if d.get('fix') and self.on_fix(d):
            self.accept()


class DetectDialog(QDialog):
    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('detect_title'))
        self.setMinimumWidth(520)
        p = params
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['xl'], SP['xl'], SP['xl'], SP['xl']); lay.setSpacing(SP['l'])
        lay.addWidget(_title(tr('detect_title')))
        intro = QLabel(tr('detect_intro')); intro.setObjectName('caption'); intro.setWordWrap(True)
        lay.addWidget(intro)

        what = QGroupBox(tr('what')); g = QGridLayout(what); g.setSpacing(SP['s'])
        self.chk = {}
        for i, (key, lab) in enumerate([('do_walls', 'w_walls'), ('do_columns', 'w_columns'), ('do_doors', 'w_doors'),
                                         ('do_windows', 'w_windows'), ('do_furniture', 'w_furn')]):
            c = QCheckBox(tr(lab)); c.setChecked(p.get(key, True)); self.chk[key] = c
            g.addWidget(c, i // 2, i % 2)
        lay.addWidget(what)

        adv = QGroupBox(tr('w_walls')); f = QFormLayout(adv); f.setSpacing(SP['m'])
        self.thr = QSpinBox(); self.thr.setRange(80, 245); self.thr.setValue(p['threshold'])
        self.dark = QSpinBox(); self.dark.setRange(40, 240); self.dark.setValue(p['wall_dark'])
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(SP['s'])
        self.tmin = QDoubleSpinBox(); self.tmin.setRange(0.02, 1); self.tmin.setDecimals(2); self.tmin.setSingleStep(0.01); self.tmin.setValue(p['wall_t_min'])
        self.tmax = QDoubleSpinBox(); self.tmax.setRange(0.05, 2); self.tmax.setDecimals(2); self.tmax.setSingleStep(0.01); self.tmax.setValue(p['wall_t_max'])
        hl.addWidget(QLabel(tr('min'))); hl.addWidget(self.tmin); hl.addWidget(QLabel(tr('max'))); hl.addWidget(self.tmax)
        self.minlen = QDoubleSpinBox(); self.minlen.setRange(0.1, 10); self.minlen.setDecimals(2); self.minlen.setSingleStep(0.05); self.minlen.setValue(p['wall_min_len'])
        f.addRow(tr('threshold'), self.thr)
        f.addRow(tr('wall_dark'), self.dark)
        f.addRow(tr('wall_t'), row)
        f.addRow(tr('wall_len'), self.minlen)
        lay.addWidget(adv)
        self.read_dims = QCheckBox(tr('read_dims')); self.read_dims.setChecked(p.get('read_dims', True))
        lay.addWidget(self.read_dims)
        self.fit = QCheckBox(tr('fit_dims')); self.fit.setChecked(p.get('fit_dims', True))
        lay.addWidget(self.fit)
        self.chainw = QCheckBox(tr('chain_windows')); self.chainw.setChecked(p.get('chain_windows', True))
        lay.addWidget(self.chainw)
        self.replace = QCheckBox(tr('replace')); self.replace.setChecked(True)
        lay.addWidget(self.replace)
        lay.addWidget(_buttons(self, tr('run')))

    def values(self):
        d = {k: c.isChecked() for k, c in self.chk.items()}
        d.update(threshold=self.thr.value(), wall_dark=self.dark.value(), wall_t_min=self.tmin.value(),
                 wall_t_max=self.tmax.value(), wall_min_len=self.minlen.value(), read_dims=self.read_dims.isChecked(), fit_dims=self.fit.isChecked(), chain_windows=self.chainw.isChecked())
        return d, self.replace.isChecked()


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('export_title'))
        self.setMinimumWidth(440)
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['xl'], SP['xl'], SP['xl'], SP['xl']); lay.setSpacing(SP['l'])
        lay.addWidget(_title(tr('export_title')))
        self.scan = QCheckBox(tr('inc_scan')); self.scan.setChecked(True)
        self.trace = QCheckBox(tr('inc_trace')); self.trace.setChecked(True)
        lay.addWidget(self.scan); lay.addWidget(self.trace)
        lay.addWidget(_buttons(self, tr('export')))


class DetectWorker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    cancelled = Signal()

    def __init__(self, project, gray, params):
        super().__init__()
        self.project, self.gray, self.params = project, gray, params
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        from ..engine import pipeline
        try:
            pipeline.run(self.project, self.gray, self.params, progress=lambda f, m='': self.progress.emit(f, m),
                         cancel=lambda: self._cancel)
            self.done.emit(self.project)
        except pipeline.Cancelled:
            self.cancelled.emit()
        except MemoryError:
            self.failed.emit(tr('err_memory'))
        except Exception as e:  # pragma: no cover
            import traceback
            self.failed.emit(f"{tr('err_detect')}\n\n{type(e).__name__}: {e}\n\n" + traceback.format_exc()[-1500:])


class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('working'))
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['xl'], SP['xl'], SP['xl'], SP['xl']); lay.setSpacing(SP['m'])
        lay.addWidget(_title(tr('working')))
        self.lab = QLabel(''); self.lab.setObjectName('caption')
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False)
        lay.addWidget(self.lab); lay.addWidget(self.bar)
        self.cancel_btn = QPushButton(tr('cancel'))
        self.cancel_btn.clicked.connect(self._on_cancel)
        lay.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignRight)
        self.on_cancel = None

    def _on_cancel(self):
        self.cancel_btn.setEnabled(False); self.lab.setText(tr('cancelling'))
        if self.on_cancel:
            self.on_cancel()

    def reject(self):                       # Esc = cancel request, the worker closes the dialog
        if self.on_cancel and self.cancel_btn.isEnabled():
            self._on_cancel()
        elif not self.on_cancel:
            super().reject()

    def set(self, f, msg):
        self.bar.setValue(int(f * 100))
        self.lab.setText(tr('p_' + msg) if msg else '')
