"""Regression tests for confirmed bugs (see AUDIT.md).  Each test failed on the baseline."""
import json, os, subprocess, sys, tempfile
import numpy as np
import cv2
import pytest


def test_B1_cli_does_not_claim_calibration_without_scale(tmp_path):
    """cli.py marked every project as calibrated with the 0.03 default, so the scale read from the
    dimensions (or estimated) was never used."""
    import importlib.util, inspect
    code = os.environ.get('P2C_CODE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(code, 'cli.py'), encoding='utf-8').read()
    assert 'scale_calibrated=True' not in src.replace(' ', '') or 'a.scale is not None' in src


def test_B2_progress_labels_translated():
    from p2c.ui.i18n import S
    from p2c.engine import pipeline
    import re
    keys = set(re.findall(r"say\([^)]*?'([a-z_]+)'\)", open(pipeline.__file__, encoding='utf-8').read()))
    keys |= set(re.findall(r"say\([^,]+, '([a-z_]+)'", open(pipeline.__file__, encoding='utf-8').read()))
    missing = [k for k in keys if 'p_' + k not in S]
    assert not missing, missing


def test_B3_project_roundtrip_keeps_results(tmp_path):
    """fit_info / chain_info / completed / chains were silently dropped on save."""
    from p2c.model import Project, Wall
    p = Project(scale=0.02, walls=[Wall([(0, 0), (100, 0)], 0.2)])
    p.fit_info = dict(used=3, err_before=0.1); p.completed = 2; p.chain_info = dict(windows=1)
    f = tmp_path / 'a.p2c'; p.save(str(f))
    q = Project.load(str(f))
    assert q.fit_info.get('used') == 3 and q.completed == 2 and q.chain_info.get('windows') == 1


def test_B4_project_load_tolerates_unknown_fields(tmp_path):
    """A project written by a newer version (extra keys) crashed from_dict with TypeError."""
    from p2c.model import Project
    d = Project().to_dict()
    d['walls'] = [dict(pts=[[0, 0], [10, 0]], t=0.2, confidence=0.9, origin='auto', future_key=1)]
    d['brand_new_section'] = {'x': 1}
    q = Project.from_dict(d)
    assert len(q.walls) == 1 and abs(q.walls[0].t - 0.2) < 1e-9


def test_B5_nn_input_size_is_bounded():
    """neural.segment resized by TGT_T/(0.18/scale) with no limit: a wrong scale (e.g. 0.2 m/px on a
    2000 px image) asked for a ~15600 px square float32 x4 buffer (several GB)."""
    from p2c.engine import neural as NN
    if not NN.available():
        pytest.skip('model not available')
    g = np.full((2000, 2000), 255, np.uint8)
    f = NN.input_factor(0.2, g.shape) if hasattr(NN, 'input_factor') else NN.TGT_T / (0.18 / 0.2)
    assert max(g.shape) * f <= 6000, f


def test_B6_dxf_export_without_image_size_not_mirrored(tmp_path):
    """export_dxf(image_size=None) used H=0, so every y became negative and the plan was mirrored."""
    import ezdxf
    from p2c.model import Project, Wall
    from p2c.dxf_export import export_dxf
    p = Project(scale=0.01, walls=[Wall([(0, 0), (100, 0)], 0.2), Wall([(0, 0), (0, 300)], 0.2)])
    f = str(tmp_path / 'a.dxf')
    export_dxf(p, f, None, None, include_scan=False)
    d = ezdxf.readfile(f)
    ys = [v[1] for e in d.modelspace().query('LWPOLYLINE') for v in e.get_points('xy')]
    assert min(ys) >= -0.2, min(ys)                    # world y must stay positive (y up, origin bottom)
