"""Main application window."""
from __future__ import annotations

import os
import tempfile
import numpy as np
from PySide6.QtCore import Qt, QSettings, QSize, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QIcon, QColor
from PySide6.QtWidgets import (QMainWindow, QToolBar, QDockWidget, QLabel, QFileDialog, QMessageBox, QComboBox,
                               QWidget, QSizePolicy, QStackedWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QToolButton, QApplication, QInputDialog, QFrame)

from ..model import Project, LAYERS
from ..engine import imageio, pipeline
from ..dxf_export import export_dxf
from .canvas import PlanView
from .panels import LayersPanel, LibraryPanel, PropertiesPanel
from .dialogs import ImportDialog, DetectDialog, ExportDialog, DetectWorker, ProgressDialog
from .icons import icon
from . import theme
from .i18n import tr, LANG, layer_name, is_rtl

SP = theme.SP
TOOLS = [('select', 'select', 'V'), ('wall', 'wall', 'W'), ('door', 'door', 'D'), ('door2', 'door2', 'B'),
         ('window', 'window', 'N'), ('column', 'column', 'C'), ('stair', 'stair', 'S'), ('poly', 'poly', 'P'),
         ('text', 'text', 'T'), ('level', 'level', 'L'), ('calib', 'calib', 'K')]


class Welcome(QWidget):
    def __init__(self, win):
        super().__init__()
        self.setObjectName('panel')
        outer = QVBoxLayout(self); outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card = QFrame(); card.setObjectName('card')
        card.setStyleSheet(f"QFrame#card {{ background: {theme.C['card']}; border: 1px solid {theme.C['border']}; border-radius: 12px; }}")
        card.setFixedWidth(560)
        lay = QVBoxLayout(card); lay.setContentsMargins(SP['xxl'], SP['xxl'], SP['xxl'], SP['xxl']); lay.setSpacing(SP['l'])
        t = QLabel('Plan2CAD'); t.setStyleSheet('font-size: 22pt; font-weight: 700;')
        by = QLabel('Essam Nawafleh  —  عصام النوافلة'); by.setStyleSheet(f"color: {theme.C['accent']}; font-weight: 600;")
        sub = QLabel(tr('app').split('—')[-1].strip()); sub.setObjectName('caption')
        lay.addWidget(t); lay.addWidget(by); lay.addWidget(sub)
        steps_ar = ['١. افتح صورة المخطط (سكان أو صورة جوال)', '٢. عاير المقياس على بُعد معروف (أداة المسطرة)',
                    '٣. شغّل الكشف الأوتوماتيكي', '٤. صحّح بالأدوات وضيف الأثاث من المكتبة', '٥. صدّر DXF وافتحه بأوتوكاد']
        steps_en = ['1. Open the plan image (scan or phone photo)', '2. Calibrate the scale on a known dimension (ruler tool)',
                    '3. Run auto detection', '4. Correct with the tools and add furniture from the library',
                    '5. Export DXF and open it in AutoCAD']
        for s in (steps_ar if LANG['cur'] == 'ar' else steps_en):
            l = QLabel(s); l.setStyleSheet(f"color: {theme.C['text2']}; padding: 2px 0;")
            lay.addWidget(l)
        row = QHBoxLayout(); row.setSpacing(SP['m'])
        b1 = QPushButton(icon('image', QColor('#FFFFFF')), '  ' + tr('open_image'))
        b1.setObjectName('primary'); b1.setIconSize(QSize(18, 18)); b1.clicked.connect(win.open_image)
        b2 = QPushButton(icon('open'), '  ' + tr('open_project')); b2.setIconSize(QSize(18, 18)); b2.clicked.connect(win.open_project)
        row.addWidget(b1); row.addWidget(b2); row.addStretch(1)
        lay.addSpacing(SP['s']); lay.addLayout(row)
        outer.addWidget(card)


class MainWindow(QMainWindow):
    def __init__(self, project=None, gray=None, state=None):
        super().__init__()
        self.settings = QSettings('Plan2CAD', 'Plan2CAD')
        self.project = project or Project()
        self.project_path = (state or {}).get('project_path')
        self.work_image = (state or {}).get('work_image')
        self.setWindowTitle(tr('app'))
        self.resize(1440, 900)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl() else Qt.LayoutDirection.LeftToRight)

        self.view = PlanView(self.project)
        self.stack = QStackedWidget()
        self.welcome = Welcome(self)
        self.stack.addWidget(self.welcome); self.stack.addWidget(self.view)
        self.setCentralWidget(self.stack)

        self._build_actions()
        self._build_toolbars()
        self._build_docks()
        self._build_menus()
        self._build_status()

        v = self.view
        v.status.connect(self.hint.setText)
        v.coords.connect(lambda x, y: self.coord_lbl.setText(tr('coords').format(x, y)))
        v.selectionInfo.connect(self.props.show_for)
        v.modelChanged.connect(self._model_changed)
        v.scaleChanged.connect(self._update_scale_lbl)
        v.toolChanged.connect(self._tool_changed)
        if state:
            v.undo_stack = state.get('undo', []); v.redo_stack = state.get('redo', [])
            v.dirty = state.get('dirty', False)
            v.visible.update(state.get('visible', {}))
            v.scan_opacity = state.get('opacity', v.scan_opacity)
        if gray is not None:
            v.set_image(gray)
            self.stack.setCurrentIndex(1)
            if state and state.get('transform') is not None:
                QTimer.singleShot(10, lambda: (v.setTransform(state['transform']),
                                               v.horizontalScrollBar().setValue(state['hs']),
                                               v.verticalScrollBar().setValue(state['vs'])))
        self._update_scale_lbl()
        self._model_changed()
        v.set_tool('select')
        if gray is None:
            self.hint.setText(tr('h_start'))

    # ------------------------------------------------------------------ UI build
    def _act(self, key, ic, slot, shortcut=None, checkable=False):
        a = QAction(icon(ic) if ic else QIcon(), tr(key), self)
        a.setToolTip(tr(key) + (f"  ({shortcut})" if shortcut else ''))
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.triggered.connect(slot)
        return a

    def _build_actions(self):
        self.a_img = self._act('open_image', 'image', self.open_image, 'Ctrl+I')
        self.a_open = self._act('open_project', 'open', self.open_project, 'Ctrl+O')
        self.a_save = self._act('save', 'save', self.save_project, 'Ctrl+S')
        self.a_saveas = self._act('save_as', None, lambda: self.save_project(as_new=True), 'Ctrl+Shift+S')
        self.a_export = self._act('export', 'export', self.export, 'Ctrl+E')
        self.a_auto = self._act('auto', 'auto', self.auto_detect, 'Ctrl+D')
        self.a_undo = self._act('undo', 'undo', self.view.undo, 'Ctrl+Z')
        self.a_redo = self._act('redo', 'redo', self.view.redo, 'Ctrl+Y')
        self.a_fit = self._act('fit', 'fit', self.view.fit, 'Ctrl+0')
        self.a_del = self._act('delete', None, self.view.delete_selected)
        self.a_all = self._act('select_all', None, self.view.select_all, 'Ctrl+A')
        self.a_clear = self._act('clear_layer', None, self.clear_layer)
        self.a_fit_dims = self._act('fit_dims', 'calib', self.fit_dims, 'Ctrl+M')
        self.a_dim_rep = self._act('dim_report', None, self.dim_report)
        self.tool_group = QActionGroup(self); self.tool_group.setExclusive(True)
        self.tool_actions = {}
        for key, ic, sc in TOOLS:
            a = QAction(icon(ic), tr('t_' + key), self)
            a.setToolTip(f"{tr('t_' + key)}  ({sc})")
            a.setShortcut(QKeySequence(sc)); a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            a.setCheckable(True)
            a.triggered.connect(lambda _=False, k=key: self.view.set_tool(k))
            self.tool_group.addAction(a); self.tool_actions[key] = a

    def _build_toolbars(self):
        tb = QToolBar('main'); tb.setMovable(False); tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        for a in (self.a_img, self.a_open, self.a_save, self.a_export):
            tb.addAction(a)
        tb.addSeparator()
        auto_btn = QToolButton(); auto_btn.setDefaultAction(self.a_auto); auto_btn.setObjectName('primary')
        auto_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon); auto_btn.setIconSize(QSize(18, 18))
        tb.addWidget(auto_btn)
        tb.addAction(self.a_fit_dims)
        tb.addSeparator()
        tb.addAction(self.a_undo); tb.addAction(self.a_redo); tb.addAction(self.a_fit)
        tb.addSeparator()
        lab = QLabel(' ' + tr('cur_thick') + ' '); lab.setObjectName('caption'); tb.addWidget(lab)
        self.thick_cb = QComboBox(); self.thick_cb.setEditable(True); self.thick_cb.setMinimumWidth(84)
        for t in ('0.10', '0.12', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40'):
            self.thick_cb.addItem(t)
        self.thick_cb.setCurrentText('0.20')
        self.thick_cb.currentTextChanged.connect(self._thick_changed)
        tb.addWidget(self.thick_cb)
        lab2 = QLabel('   ' + tr('cur_layer') + ' '); lab2.setObjectName('caption'); tb.addWidget(lab2)
        self.layer_cb = QComboBox(); self.layer_cb.setMinimumWidth(140)
        for code in ['A-FURN', 'A-SANITARY', 'A-KITCHEN', 'A-COURT', 'A-STAIR']:
            self.layer_cb.addItem(layer_name(code), code)
        self.layer_cb.currentIndexChanged.connect(lambda i: setattr(self.view, 'poly_layer', self.layer_cb.itemData(i)))
        tb.addWidget(self.layer_cb)
        from PySide6.QtWidgets import QCheckBox
        self.fit_chk = QCheckBox(tr('fit_mode')); self.fit_chk.setChecked(True); self.fit_chk.setToolTip(tr('fit_tip'))
        self.fit_chk.toggled.connect(lambda v: setattr(self.view, 'fit_mode', v))
        tb.addSeparator(); tb.addWidget(self.fit_chk)
        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        lang = QToolButton(); lang.setText(tr('lang_btn')); lang.setObjectName('lang'); lang.clicked.connect(self.toggle_language)
        tb.addWidget(lang)

        tools = QToolBar('tools'); tools.setMovable(False); tools.setIconSize(QSize(22, 22))
        tools.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tools)
        for i, (key, _, _) in enumerate(TOOLS):
            if key in ('door', 'poly', 'calib'):
                tools.addSeparator()
            tools.addAction(self.tool_actions[key])

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _build_docks(self):
        self.layers = LayersPanel(self.view)
        self.layers.visibilityChanged.connect(self._layer_vis)
        self.layers.opacityChanged.connect(self._opacity)
        self.library = LibraryPanel()
        self.library.blockChosen.connect(self._choose_block)
        self.props = PropertiesPanel(self.view)
        right = Qt.DockWidgetArea.RightDockWidgetArea
        self.d_props = self._dock(tr('properties'), self.props, right)
        self.d_layers = self._dock(tr('layers'), self.layers, right)
        self.d_lib = self._dock(tr('library'), self.library, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.resizeDocks([self.d_props, self.d_layers], [260, 420], Qt.Orientation.Vertical)
        self.resizeDocks([self.d_lib], [250], Qt.Orientation.Horizontal)
        self.resizeDocks([self.d_props], [280], Qt.Orientation.Horizontal)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu(tr('file'))
        for a in (self.a_img, self.a_open, None, self.a_save, self.a_saveas, None, self.a_export):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu(tr('edit'))
        for a in (self.a_undo, self.a_redo, None, self.a_all, self.a_del, self.a_clear):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu(tr('tools'))
        m.addAction(self.a_auto); m.addAction(self.a_fit_dims); m.addAction(self.a_dim_rep); m.addSeparator()
        for key, _, _ in TOOLS:
            m.addAction(self.tool_actions[key])
        m = mb.addMenu(tr('view'))
        m.addAction(self.a_fit)
        for d in (self.d_lib, self.d_layers, self.d_props):
            m.addAction(d.toggleViewAction())
        m = mb.addMenu(tr('help'))
        m.addAction(tr('shortcuts'), lambda: QMessageBox.information(self, tr('shortcuts'), tr('shortcuts_text')))
        m.addAction(tr('about'), lambda: QMessageBox.about(self, tr('about'), tr('about_text')))

    def _build_status(self):
        sb = self.statusBar()
        self.hint = QLabel(''); self.hint.setMinimumWidth(200)
        self.scale_lbl = QLabel(''); self.coord_lbl = QLabel('')
        self.coord_lbl.setMinimumWidth(170)
        sb.addWidget(self.hint, 1)
        sb.addPermanentWidget(self.scale_lbl); sb.addPermanentWidget(self.coord_lbl)

    # ------------------------------------------------------------------ slots
    def _tool_changed(self, tool):
        a = self.tool_actions.get(tool)
        if a:
            a.setChecked(True)
        elif tool == 'furn':
            for a in self.tool_actions.values():
                a.setChecked(False)

    def _thick_changed(self, txt):
        try:
            self.view.wall_t = max(0.02, float(txt.replace(',', '.')))
        except ValueError:
            pass

    def _layer_vis(self, code, on):
        self.view.visible[code] = on
        self.view.rebuild()

    def _opacity(self, v):
        self.view.scan_opacity = v
        if self.view.scan_item:
            self.view.scan_item.setOpacity(v)

    def _choose_block(self, name):
        if self.view.gray is None:
            return
        self.view.furn_block = name
        self.view.set_tool('furn')
        self.view.setFocus()

    def _model_changed(self):
        self.a_undo.setEnabled(bool(self.view.undo_stack))
        self.a_redo.setEnabled(bool(self.view.redo_stack))
        title = tr('app')
        if self.project_path:
            title = os.path.basename(self.project_path) + ' — ' + title
        self.setWindowTitle(('● ' if self.view.dirty else '') + title)

    def _update_scale_lbl(self):
        p = self.project
        self.scale_lbl.setText(tr('scale_lbl').format(p.scale) + ('' if p.scale_calibrated else tr('not_calib')))

    # ------------------------------------------------------------------ file ops
    def _confirm_discard(self):
        if not self.view.dirty:
            return True
        r = QMessageBox.question(self, 'Plan2CAD', tr('unsaved'),
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save_project()
        return r == QMessageBox.StandardButton.Discard

    def open_image(self):
        if not self._confirm_discard():
            return
        last = self.settings.value('last_dir', '')
        path, _ = QFileDialog.getOpenFileName(self, tr('open_image'), last,
                                              'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)')
        if not path:
            return
        self.settings.setValue('last_dir', os.path.dirname(path))
        self.load_image(path)

    def load_image(self, path, ask=True, opts=None):
        gray = imageio.imread_gray(path)
        try:
            wv = imageio.wall_view(imageio.imread_color(path))
        except Exception:
            wv = None
        if ask:
            dlg = ImportDialog(self.project.scale, self)
            if dlg.exec() != ImportDialog.DialogCode.Accepted:
                return
            opts = dlg.values()
        opts = opts or dict(rotate=0, deskew=True, clean=True, scale=self.project.scale)
        if opts['rotate']:
            gray = imageio.rotate_bound(gray, -opts['rotate'])  # clockwise
            if wv is not None:
                wv = imageio.rotate_bound(wv, -opts['rotate'])
        if opts['deskew']:
            sk = imageio.estimate_skew(gray)
            if sk:
                gray = imageio.rotate_bound(gray, sk)
                if wv is not None:
                    wv = imageio.rotate_bound(wv, sk)
        if opts['clean']:
            gray = imageio.clean_background(gray)
        work = os.path.join(tempfile.gettempdir(), f"plan2cad_work_{os.getpid()}.png")
        imageio.imwrite(work, gray)
        wpath = imageio.wall_view_path(work)
        if wv is not None and wv.shape == gray.shape:
            imageio.imwrite(wpath, wv)
        elif os.path.exists(wpath):
            os.remove(wpath)
        self.work_image = work
        self.project = Project(image_path=work, scale=opts['scale'])
        self.project_path = None
        self.view.project = self.project
        self.props.view = self.view
        self.view.undo_stack.clear(); self.view.redo_stack.clear(); self.view.dirty = True
        self.view.set_image(gray)
        self.stack.setCurrentIndex(1)
        self._update_scale_lbl(); self._model_changed()
        self.view.set_tool('select')
        self.hint.setText(tr('h_calib'))

    def open_project(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, tr('open_project'), self.settings.value('last_dir', ''),
                                              'Plan2CAD (*.p2c)')
        if path:
            self.load_project(path)

    def load_project(self, path):
        pr = Project.load(path)
        img = pr.image_path
        if not os.path.isabs(img):
            img = os.path.join(os.path.dirname(path), img)
        gray = imageio.imread_gray(img)
        pr.image_path = img
        self.project = pr; self.project_path = path; self.work_image = img
        self.view.project = pr
        self.view.undo_stack.clear(); self.view.redo_stack.clear(); self.view.dirty = False
        self.view.set_image(gray)
        self.stack.setCurrentIndex(1)
        self._update_scale_lbl(); self._model_changed()
        self.settings.setValue('last_dir', os.path.dirname(path))

    def save_project(self, as_new=False):
        if self.view.gray is None:
            return False
        path = self.project_path
        if as_new or not path:
            path, _ = QFileDialog.getSaveFileName(self, tr('save'), os.path.join(self.settings.value('last_dir', ''), 'plan.p2c'),
                                                  'Plan2CAD (*.p2c)')
            if not path:
                return False
            if not path.lower().endswith('.p2c'):
                path += '.p2c'
        base = os.path.splitext(os.path.basename(path))[0] + '_image.png'
        img_dst = os.path.join(os.path.dirname(path), base)
        imageio.imwrite(img_dst, self.view.gray)
        old = self.project.image_path
        self.project.image_path = base
        self.project.save(path)
        self.project.image_path = img_dst
        self.work_image = img_dst
        self.project_path = path
        self.view.dirty = False
        self._model_changed()
        self.hint.setText(tr('saved').format(path))
        return True

    def export(self):
        if self.view.gray is None:
            QMessageBox.information(self, 'Plan2CAD', tr('no_image')); return
        dlg = ExportDialog(self)
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        start = os.path.splitext(self.project_path)[0] + '.dxf' if self.project_path else os.path.join(self.settings.value('last_dir', ''), 'plan.dxf')
        path, _ = QFileDialog.getSaveFileName(self, tr('export'), start, 'DXF (*.dxf)')
        if not path:
            return
        if not path.lower().endswith('.dxf'):
            path += '.dxf'
        export_dxf(self.project, path, self.view.image_size, self.work_image,
                   include_scan=dlg.scan.isChecked(), include_trace=dlg.trace.isChecked())
        self.hint.setText(tr('exported').format(path))

    def clear_layer(self):
        kinds = [('walls', 'A-WALL'), ('columns', 'A-COLUMN'), ('windows', 'A-WINDOW'), ('doors', 'A-DOOR'),
                 ('stairs', 'A-STAIR'), ('furniture', 'A-FURN'), ('polys', 'A-KITCHEN'), ('texts', 'A-TEXT'),
                 ('levels', 'A-LEVEL'), ('trace', 'A-TRACE')]
        names = [layer_name(c) for _, c in kinds]
        name, ok = QInputDialog.getItem(self, tr('clear_layer'), tr('clear_ask'), names, 0, False)
        if not ok:
            return
        kind = kinds[names.index(name)][0]
        self.view.push_undo()
        setattr(self.project, kind, [])
        self.view.rebuild(full=(kind == 'trace'))
        self.view.commit()

    # ------------------------------------------------------------------ detection
    def auto_detect(self):
        if self.view.gray is None:
            QMessageBox.information(self, 'Plan2CAD', tr('no_image')); return
        from ..engine import dims as _DM
        if not self.project.scale_calibrated and not _DM.ocr_available():
            r = QMessageBox.question(self, 'Plan2CAD', tr('scale_warn'))
            if r != QMessageBox.StandardButton.Yes:
                self.view.set_tool('calib'); return
        params = dict(pipeline.DEFAULTS)
        saved = self.settings.value('detect_params')
        if isinstance(saved, dict):
            params.update(saved)
        dlg = DetectDialog(params, self)
        if dlg.exec() != DetectDialog.DialogCode.Accepted:
            return
        vals, replace = dlg.values()
        self.settings.setValue('detect_params', vals)
        params.update(vals)
        work = Project(image_path=self.project.image_path, scale=self.project.scale, scale_calibrated=self.project.scale_calibrated)
        self._replace = replace
        self.prog = ProgressDialog(self)
        self.worker = DetectWorker(work, self.view.gray, params)
        self.worker.progress.connect(self.prog.set)
        self.worker.done.connect(self._detect_done)
        self.worker.failed.connect(self._detect_failed)
        self.worker.start()
        self.prog.exec()

    def _detect_done(self, res):
        self.prog.accept()
        v = self.view; p = self.project
        v.push_undo()
        p.angles = res.angles
        p.trace = res.trace
        p.dims = res.dims; p.texts_raw = res.texts_raw; p.dims_info = res.dims_info; p.chains = res.chains
        if res.scale_source == 'dims' and not p.scale_calibrated:
            p.scale = res.scale; p.scale_calibrated = True; p.scale_source = 'dims'
            self._update_scale_lbl()
        elif res.scale_source == 'doors' and not p.scale_calibrated:
            p.scale = res.scale; p.scale_source = 'doors'
            self._update_scale_lbl()
        for k in ('walls', 'columns', 'doors', 'windows', 'furniture'):
            if self._replace:
                setattr(p, k, getattr(res, k))
            else:
                getattr(p, k).extend(getattr(res, k))
        v.rebuild(full=True)
        v.commit()
        msg = tr('detected').format(len(res.walls), len(res.columns), len(res.doors), len(res.windows), len(res.furniture))
        di = res.dims_info or {}
        if di.get('scale'):
            msg = tr('dims_found').format(di.get('n_numbers', 0), di.get('n_dims', 0), di['scale']) + '   |   ' + msg
            if getattr(res, 'completed', 0):
                msg += '   |   ' + tr('completed').format(res.completed)
            if getattr(res, 'fit_info', {}).get('used'):
                msg += '   |   ' + self._fit_msg(res.fit_info)
            ci = getattr(res, 'chain_info', {})
            if ci.get('windows'):
                msg += '   |   ' + tr('chains_done').format(ci['windows'], ci['chains'])
        if res.scale_source == 'doors' and not p.scale_calibrated:
            msg = tr('scale_doors').format(res.scale) + '   |   ' + msg
        self.hint.setText(msg)

    def fit_dims(self):
        from ..engine.fit import fit_to_dimensions
        self.view.push_undo()
        from ..engine.chains import windows_from_chains
        ci = windows_from_chains(self.project, self.view.gray) if self.project.dims else {}
        info = fit_to_dimensions(self.project)
        if not info.get('used') and not ci.get('windows'):
            self.view.undo_stack.pop()
            self.hint.setText(tr('fit_none')); return
        self.view.commit()
        msg = self._fit_msg(info) if info.get('used') else ''
        if ci.get('windows'):
            msg += ('   |   ' if msg else '') + tr('chains_done').format(ci['windows'], ci['chains'])
        self.hint.setText(msg)

    def _fit_msg(self, info):
        return tr('fit_done').format(info['walls_moved'], info['used'], info['err_before'] * 100,
                                     info['err_after'] * 100, info['max_after'] * 100)

    def dim_report(self):
        from ..engine.fit import dimension_report
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel
        rep = dimension_report(self.project)
        dlg = QDialog(self); dlg.setWindowTitle(tr('dim_report')); dlg.resize(460, 560)
        lay = QVBoxLayout(dlg); lay.setContentsMargins(SP['l'], SP['l'], SP['l'], SP['l'])
        ok = sum(1 for r in rep if r['diff'] is not None and abs(r['diff']) < 0.02)
        off = sum(1 for r in rep if r['diff'] is not None and abs(r['diff']) >= 0.02)
        lab = QLabel(tr('rep_summary').format(ok, off, len(rep) - ok - off)); lab.setObjectName('caption')
        lay.addWidget(lab)
        tab = QTableWidget(len(rep), 3); tab.setHorizontalHeaderLabels(tr('rep_cols').split('|'))
        tab.setAlternatingRowColors(True); tab.verticalHeader().setVisible(False)
        rep.sort(key=lambda r: (r['diff'] is None, -abs(r['diff'] or 0)))
        for i, r in enumerate(rep):
            vals = [r['text'], '—' if r['drawn'] is None else f"{r['drawn']:.3f}",
                    '—' if r['diff'] is None else f"{r['diff'] * 100:+.1f}"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 2 and r['diff'] is not None:
                    it.setForeground(QColor('#3FB950') if abs(r['diff']) < 0.02 else QColor('#F85149'))
                tab.setItem(i, j, it)
        tab.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(tab)
        dlg.exec()

    def _detect_failed(self, msg):
        self.prog.reject()
        QMessageBox.critical(self, 'Plan2CAD', msg[-2000:])

    # ------------------------------------------------------------------ language
    def toggle_language(self):
        LANG['cur'] = 'en' if LANG['cur'] == 'ar' else 'ar'
        self.settings.setValue('lang', LANG['cur'])
        v = self.view
        state = dict(project_path=self.project_path, work_image=self.work_image, undo=v.undo_stack,
                     redo=v.redo_stack, dirty=v.dirty, visible=dict(v.visible), opacity=v.scan_opacity,
                     transform=v.transform(), hs=v.horizontalScrollBar().value(), vs=v.verticalScrollBar().value())
        QApplication.instance().setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl() else Qt.LayoutDirection.LeftToRight)
        w = MainWindow(self.project, v.gray, state)
        w.setGeometry(self.geometry())
        if self.isMaximized():
            w.showMaximized()
        else:
            w.show()
        QApplication.instance()._main = w
        self.view.dirty = False
        self.close()

    def closeEvent(self, e):
        if self.view.dirty and not self._confirm_discard():
            e.ignore(); return
        e.accept()
