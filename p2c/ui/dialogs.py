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
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['xl'], SP['xl'], SP['xl'], SP['xl']); lay.setSpacing(SP['l'])
        lay.addWidget(_title(tr('import_title')))
        form = QFormLayout(); form.setSpacing(SP['m'])
        self.rot = QComboBox()
        for a in (0, 90, 180, 270):
            self.rot.addItem(f"{a}°", a)
        form.addRow(tr('rotate'), self.rot)
        self.deskew = QCheckBox(tr('deskew')); self.deskew.setChecked(True)
        self.clean = QCheckBox(tr('clean_bg')); self.clean.setChecked(True)
        form.addRow('', self.deskew); form.addRow('', self.clean)
        self.scale = QDoubleSpinBox(); self.scale.setDecimals(5); self.scale.setRange(0.00001, 10); self.scale.setValue(scale)
        self.scale.setSingleStep(0.001)
        form.addRow(tr('scale_m_px'), self.scale)
        lay.addLayout(form)
        note = QLabel(tr('scale_note')); note.setObjectName('hint'); note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(_buttons(self))

    def values(self):
        return dict(rotate=self.rot.currentData(), deskew=self.deskew.isChecked(), clean=self.clean.isChecked(),
                    scale=self.scale.value())


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

    def __init__(self, project, gray, params):
        super().__init__()
        self.project, self.gray, self.params = project, gray, params

    def run(self):
        try:
            from ..engine import pipeline
            pipeline.run(self.project, self.gray, self.params, progress=lambda f, m='': self.progress.emit(f, m))
            self.done.emit(self.project)
        except Exception as e:  # pragma: no cover
            import traceback
            self.failed.emit(traceback.format_exc())


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

    def set(self, f, msg):
        self.bar.setValue(int(f * 100))
        self.lab.setText(tr('p_' + msg) if msg else '')


class AIDialog(QDialog):
    def __init__(self, n_tiles, est, key, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QLineEdit
        self.setWindowTitle(tr('ai_title'))
        self.setMinimumWidth(540)
        self.n, self.est = n_tiles, est
        lay = QVBoxLayout(self); lay.setContentsMargins(SP['xl'], SP['xl'], SP['xl'], SP['xl']); lay.setSpacing(SP['l'])
        lay.addWidget(_title(tr('ai_title')))
        intro = QLabel(tr('ai_intro')); intro.setObjectName('caption'); intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout(); form.setSpacing(SP['m'])
        self.key = QLineEdit(key or ''); self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText('sk-ant-...')
        form.addRow(tr('ai_key'), self.key)
        help_ = QLabel(tr('ai_key_help')); help_.setObjectName('caption'); help_.setWordWrap(True)
        form.addRow('', help_)
        self.q = QComboBox(); self.q.addItem(tr('ai_q_high'), 'high'); self.q.addItem(tr('ai_q_med'), 'medium')
        self.q.currentIndexChanged.connect(self._upd)
        form.addRow(tr('ai_quality'), self.q)
        lay.addLayout(form)
        self.cost = QLabel(); self.cost.setObjectName('hint'); self.cost.setWordWrap(True)
        lay.addWidget(self.cost); self._upd()
        lay.addWidget(_buttons(self, tr('ai_run')))

    def _upd(self):
        f = 1.0 if self.q.currentData() == 'high' else 0.6
        self.cost.setText(tr('ai_est').format(self.n, self.est * f))

    def values(self):
        return self.key.text().strip(), self.q.currentData()


class AIWorker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, project, gray, key, effort, need_auto):
        super().__init__()
        self.project, self.gray, self.key, self.effort, self.need_auto = project, gray, key, effort, need_auto

    def run(self):
        try:
            from ..engine import pipeline, ai_trace
            if self.need_auto:
                pipeline.run(self.project, self.gray, {}, progress=lambda f, m='': self.progress.emit(0.25 * f, m))
            ai_trace.run_ai(self.project, self.gray, api_key=self.key, effort=self.effort,
                            progress=lambda f, m='': self.progress.emit(0.25 + 0.75 * f if self.need_auto else f, m))
            self.done.emit(self.project)
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())
