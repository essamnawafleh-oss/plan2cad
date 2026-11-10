"""Offscreen UI workflow: import -> detect (and cancel) -> validation report -> fix + undo -> export + verify."""
import os
import time
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QtW = pytest.importorskip('PySide6.QtWidgets')


@pytest.fixture(scope='module')
def app():
    a = QtW.QApplication.instance() or QtW.QApplication([])
    yield a


def _wait(app, cond, timeout=600):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents(); time.sleep(0.02)
    return cond()


def _image(tmp_path, case='clean_filled'):
    import cv2
    from bench import synth
    img, pl, Hc, kind = synth.make_case(case, 1)
    f = str(tmp_path / f'{case}.png'); cv2.imwrite(f, img)
    return f, pl


def _run(w, app, cancel_after=None):
    from p2c.ui.dialogs import DetectWorker
    from p2c.engine import pipeline
    from p2c.model import Project
    params = dict(pipeline.DEFAULTS); params['read_dims'] = False
    work = Project(image_path=w.project.image_path, scale=w.project.scale, scale_calibrated=w.project.scale_calibrated)
    wk = DetectWorker(work, w.view.gray, params)
    got = {}
    wk.done.connect(lambda r: got.setdefault('done', r))
    wk.failed.connect(lambda e: got.setdefault('failed', e))
    wk.cancelled.connect(lambda: got.setdefault('cancelled', True))
    if cancel_after is not None:
        wk.progress.connect(lambda f, m: wk.request_cancel() if f >= cancel_after else None)
    wk.start()
    assert _wait(app, lambda: got and wk.isFinished())
    return got


def test_full_workflow(app, tmp_path):
    from p2c.ui.main_window import MainWindow
    from p2c.engine import validate as VAL
    from p2c.dxf_export import verify_dxf, export_dxf
    f, pl = _image(tmp_path)
    w = MainWindow(); w.show(); app.processEvents()
    w.load_image(f, ask=False, opts=dict(scale=pl.scale, scale_known=True))
    assert w.project.scale_calibrated and w.project.transform.get('steps') is not None
    got = _run(w, app)
    assert 'done' in got, got.get('failed')
    w._replace = True
    w.prog = type('P', (), {'accept': lambda self: None})()
    w._detect_done(got['done'])
    p = w.project
    assert len(p.walls) >= 8 and all(getattr(x, 'origin', '') in ('auto', 'fixed', 'uncertain') for x in p.walls)
    # validation + a reversible fix
    from p2c.model import Wall
    w.view.push_undo(); p.walls.append(Wall(list(p.walls[0].pts), p.walls[0].t)); w.view.commit()
    dup = [d for d in VAL.validate(p) if d['code'] == 'duplicate_wall']
    assert dup
    n = len(p.walls)
    w.view.push_undo(); VAL.apply_fix(p, dup[0]); w.view.commit()
    assert len(w.project.walls) == n - 1
    w.view.undo()
    assert len(w.project.walls) == n
    # origin overlay renders
    w.a_origin.setChecked(True); w.toggle_origin(); app.processEvents()
    # manual edit marks the object
    w.view.commit(keep=[('walls', 0)])
    assert w.project.walls[0].origin == 'manual'
    # export + verify
    out = str(tmp_path / 'o.dxf')
    export_dxf(w.project, out, w.view.image_size, w.work_image, include_scan=True)
    rep = verify_dxf(out, w.project)
    assert rep['ok'], rep['problems']
    # save / load keeps origin, validation and corrections
    pp = str(tmp_path / 'o.p2c'); w.project.save(pp)
    from p2c.model import Project
    q = Project.load(pp)
    assert q.walls[0].origin == 'manual' and isinstance(q.corrections, list) and q.transform


def test_detection_can_be_cancelled(app, tmp_path):
    from p2c.ui.main_window import MainWindow
    f, pl = _image(tmp_path)
    w = MainWindow(); app.processEvents()
    w.load_image(f, ask=False, opts=dict(scale=pl.scale, scale_known=True))
    got = _run(w, app, cancel_after=0.3)
    assert got.get('cancelled') and 'done' not in got
    assert w.project.walls == []                               # nothing applied


def test_phone_photo_import_rectifies(app, tmp_path):
    from p2c.ui.main_window import MainWindow
    f, pl = _image(tmp_path, 'photo_persp')
    w = MainWindow(); app.processEvents()
    w.load_image(f, ask=False, opts=dict(scale=0.03))
    st = {s['step']: s for s in w.project.transform['steps']}
    assert st['perspective']['applied'] and not w.project.scale_calibrated
    from p2c.ui.dialogs import CompareDialog
    CompareDialog(w._import.original, w._import.processed_color, w.project.transform['steps'], w).show()
    app.processEvents()
