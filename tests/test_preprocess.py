"""Preprocessing: every geometric step is recorded and maps ground truth correctly."""
import math
import numpy as np
import cv2
import pytest

from bench import synth
from p2c.engine import preprocess as PP


def _corner_angles(q):
    out = []
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        u, v = a - b, c - b
        out.append(math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    return out


def _outer_corners(pl):
    xs = [p[0] for a, b, t in pl.walls for p in (a, b)]; ys = [p[1] for a, b, t in pl.walls for p in (a, b)]
    return np.array([[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]], float)


def test_clean_scan_untouched():
    img, pl, Hc, _ = synth.make_case('clean_filled', 1)
    r = PP.preprocess(img)
    assert np.allclose(r.H, np.eye(3))
    assert not any(s['applied'] for s in r.steps if s['step'] in ('perspective', 'deskew', 'orientation'))


@pytest.mark.parametrize('case', ['scan_rot3', 'photo_shadow'])
def test_rotated_scan_is_straightened(case):
    img, pl, Hc, _ = synth.make_case(case, 1)
    r = PP.preprocess(img)
    q = PP.apply_h(r.H @ Hc, _outer_corners(pl))
    e = q[1] - q[0]
    assert abs(math.degrees(math.atan2(e[1], e[0]))) < 0.4          # top wall horizontal again
    # the mapped corner lands on wall ink
    x, y = np.round(q[0]).astype(int)
    assert r.gray[y - 2:y + 3, x - 2:x + 3].min() < 100


def test_perspective_photo_is_rectified():
    img, pl, Hc, kind = synth.make_case('photo_persp', 1)
    before = _corner_angles(PP.apply_h(Hc, _outer_corners(pl)))
    r = PP.preprocess(img)
    st = {s['step']: s for s in r.steps}
    assert st['perspective']['applied']
    after = _corner_angles(PP.apply_h(r.H @ Hc, _outer_corners(pl)))
    assert max(abs(a - 90) for a in after) < 1.0 < max(abs(a - 90) for a in before)
    # aspect ratio of the building preserved (within 2 %)
    q = PP.apply_h(r.H @ Hc, _outer_corners(pl)); c = _outer_corners(pl)
    ar_true = (c[1, 0] - c[0, 0]) / (c[3, 1] - c[0, 1])
    ar = np.linalg.norm(q[1] - q[0]) / np.linalg.norm(q[3] - q[0])
    assert abs(ar / ar_true - 1) < 0.02


def test_perspective_not_applied_without_page_edge():
    img, pl, Hc, _ = synth.make_case('scan_noisy', 1)
    r = PP.preprocess(img)
    assert not {s['step']: s for s in r.steps}['perspective']['applied']


def test_round_trip_mapping():
    img, pl, Hc, _ = synth.make_case('photo_persp', 2)
    r = PP.preprocess(img)
    pts = np.array([[10.0, 20.0], [400.0, 300.0], [700.0, 50.0]])
    assert np.allclose(r.to_source(r.to_work(pts)), pts, atol=1e-6)


def test_uneven_lighting_flattened():
    img, pl, Hc, _ = synth.make_case('clean_filled', 1)
    dark, _ = synth.deg_lighting(img, 0.55, seed=1)
    g0 = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
    r = PP.preprocess(dark)
    paper = cv2.erode((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) == 255).astype(np.uint8), np.ones((9, 9))) > 0
    assert {s['step']: s for s in r.steps}['lighting']['applied']
    assert np.percentile(r.gray[paper], 5) > 200 > np.percentile(g0[paper], 5)
    walls = synth.wall_mask(pl, include_windows=False) > 0
    walls = cv2.erode(walls.astype(np.uint8), np.ones((3, 3))) > 0
    assert np.median(r.gray[walls]) < 80                             # thin and thick strokes survive


def test_noise_detected_and_reduced():
    img, pl, Hc, _ = synth.make_case('clean_filled', 1)
    noisy, _ = synth.deg_noise(img, 25, seed=3)
    g = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    s0 = PP.noise_sigma(g)
    r = PP.preprocess(noisy)
    assert {s['step']: s for s in r.steps}['denoise']['applied']
    assert PP.noise_sigma(r.gray) < 0.5 * s0
    walls = cv2.erode((synth.wall_mask(pl, include_windows=False) > 0).astype(np.uint8), np.ones((3, 3))) > 0
    assert np.median(r.gray[walls]) < 80


def test_noise_free_image_not_denoised():
    img, pl, Hc, _ = synth.make_case('clean_hatched', 1)
    r = PP.preprocess(img)
    assert not {s['step']: s for s in r.steps}['denoise']['applied']


def test_low_resolution_ok():
    img, pl, Hc, _ = synth.make_case('scan_lowres', 1)
    r = PP.preprocess(img)
    assert r.gray.shape[:2] == img.shape[:2]


def test_manual_rotation_matrix_exact():
    img = np.full((100, 200, 3), 255, np.uint8); img[10, 20] = 0
    for k in (1, 2, 3):
        r = PP.preprocess(img, PP.PreprocessOptions(rotate=90 * k, deskew=False, lighting='off', denoise='off',
                                                    perspective='off'))
        x, y = np.round(r.to_work([[20, 10]])[0]).astype(int)
        assert r.gray[y, x] == 0


def test_size_limit_recorded():
    img = np.full((300, 1200, 3), 255, np.uint8)
    r = PP.preprocess(img, PP.PreprocessOptions(max_side=600))
    assert max(r.gray.shape) == 600 and abs(r.H[0, 0] - 0.5) < 1e-9


def test_exif_orientation_applied(tmp_path):
    PIL = pytest.importorskip('PIL.Image')
    a = np.zeros((100, 200, 3), np.uint8); a[:, :50] = 255
    im = PIL.fromarray(a); ex = im.getexif(); ex[0x0112] = 6
    f = tmp_path / 'e.jpg'; im.save(str(f), exif=ex)
    from p2c.engine import imageio
    assert imageio.imread_color(str(f)).shape[:2] == (200, 100)
