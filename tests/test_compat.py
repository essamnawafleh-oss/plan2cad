"""Saved-project compatibility in both directions."""
import json
from p2c.model import Project, Wall, Door, Dim

LEGACY = {   # a project as written by v1.0.x (no conf/origin/validation/... keys)
    "image_path": "plan.png", "scale": 0.025, "scale_calibrated": True, "angles": [0.0, 90.0],
    "dims_info": {"scale": 0.025}, "scale_source": "dims", "texts_raw": [["3.45", [[0, 0], [1, 0], [1, 1], [0, 1]]]],
    "walls": [{"pts": [[0, 0], [100, 0]], "t": 0.2}], "columns": [{"x": 5, "y": 5, "s": 0.3, "a": 0}],
    "doors": [{"p0": [10, 0], "p1": [50, 0], "side": 1, "double": False}],
    "windows": [{"p0": [60, 0], "p1": [90, 0], "t": 0.2}], "stairs": [], "furniture": [], "polys": [],
    "texts": [], "levels": [], "dims": [{"p0": [0, -10], "p1": [100, -10], "text": "2.50", "value": 2.5}],
    "trace": [[[0, 0], [1, 1]]]}


def test_legacy_project_loads():
    p = Project.from_dict(json.loads(json.dumps(LEGACY)))
    assert len(p.walls) == 1 and p.walls[0].origin == '' and p.walls[0].conf == 1.0
    assert p.dims[0].unit == 'm' and p.validation == [] and p.transform == {}


def test_files_without_new_features_stay_legacy_shaped():
    """When nothing new is used, element records carry only the old keys, so v1.0.x can still open them
    (its loader passes every element key to the dataclass constructor)."""
    p = Project.from_dict(LEGACY)
    d = p.to_dict()
    for kind in ('walls', 'doors', 'windows', 'columns', 'dims'):
        for rec in d[kind]:
            assert not ({'conf', 'origin', 'unit', 'status'} & set(rec)), (kind, rec)


def test_new_fields_roundtrip(tmp_path):
    p = Project(scale=0.02, walls=[Wall([(0, 0), (10, 0)], 0.2, conf=0.42, origin='uncertain')],
                doors=[Door((0, 0), (40, 0), conf=0.7, origin='auto')],
                dims=[Dim((0, 0), (10, 0), '345', 3.45, conf=0.8, unit='cm', status='used')])
    p.transform = {'H': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    f = tmp_path / 'a.p2c'; p.save(str(f))
    q = Project.load(str(f))
    assert q.walls[0].origin == 'uncertain' and abs(q.walls[0].conf - 0.42) < 1e-9
    assert q.dims[0].unit == 'cm' and q.dims[0].status == 'used' and q.transform['H'][0][0] == 1
