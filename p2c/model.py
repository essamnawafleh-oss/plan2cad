"""Project data model.

All geometry is stored in *image pixel* coordinates of the working image
(x to the right, y downward).  Real-world sizes (wall thickness, furniture
size, ...) are stored in metres.  ``Project.scale`` converts pixels to metres.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

Pt = Tuple[float, float]

# Layer names used everywhere (canvas + DXF)
LAYERS = [
    "A-SCAN", "A-TRACE", "A-DIM", "A-WALL", "A-COLUMN", "A-WINDOW", "A-DOOR", "A-STAIR",
    "A-FURN", "A-SANITARY", "A-KITCHEN", "A-COURT", "A-TEXT", "A-LEVEL",
]


@dataclass
class Wall:
    pts: List[Pt]
    t: float = 0.20            # thickness (m)


@dataclass
class Column:
    x: float
    y: float
    s: float = 0.35            # size (m)
    a: float = 0.0             # rotation (deg, image space)


@dataclass
class Door:
    p0: Pt                     # jamb (hinge side for single doors)
    p1: Pt                     # other jamb
    side: int = 1              # +1 / -1 : swing to the left/right of p0->p1
    double: bool = False


@dataclass
class Window:
    p0: Pt
    p1: Pt
    t: float = 0.25            # depth (m), normally = wall thickness


@dataclass
class Stair:
    p: Pt                      # start corner
    along: Pt                  # vector along the flight (px)
    across: Pt                 # vector across the flight (px)
    tread: float = 0.30        # tread depth (m)


@dataclass
class Furn:
    block: str
    x: float                   # centre (px)
    y: float
    w: float                   # width (m)  (local x)
    d: float                   # depth (m)  (local y, front = -y)
    rot: float = 0.0           # facing angle in image space (deg); 90 = faces down
    mirror: bool = False
    layer: str = "A-FURN"


@dataclass
class Poly:
    pts: List[Pt]
    layer: str = "A-FURN"
    closed: bool = False


@dataclass
class Text:
    x: float
    y: float
    text: str
    h: float = 0.30            # height (m)


@dataclass
class Level:
    x: float
    y: float
    value: str


@dataclass
class Dim:
    p0: Pt
    p1: Pt
    text: str
    value: float


@dataclass
class Project:
    image_path: str = ""
    scale: float = 0.03                    # metres per pixel
    scale_calibrated: bool = False
    angles: List[float] = field(default_factory=lambda: [0.0, 90.0])   # dominant directions (deg, image space)
    walls: List[Wall] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    stairs: List[Stair] = field(default_factory=list)
    furniture: List[Furn] = field(default_factory=list)
    polys: List[Poly] = field(default_factory=list)
    texts: List[Text] = field(default_factory=list)
    levels: List[Level] = field(default_factory=list)
    trace: List[List[Pt]] = field(default_factory=list)   # raw vectorised scan lines
    dims: List[Dim] = field(default_factory=list)          # dimensions read from the drawing
    texts_raw: list = field(default_factory=list)          # OCR texts (text, box)
    dims_info: dict = field(default_factory=dict)
    scale_source: str = ''
    completed: int = 0
    fit_info: dict = field(default_factory=dict)
    chain_info: dict = field(default_factory=dict)
    chains: list = field(default_factory=list)
    ai_info: dict = field(default_factory=dict)

    KINDS = {
        "walls": Wall, "columns": Column, "doors": Door, "windows": Window, "stairs": Stair,
        "furniture": Furn, "polys": Poly, "texts": Text, "levels": Level, "dims": Dim,
    }

    # ------------------------------------------------------------ helpers
    def px(self, metres: float) -> float:
        return metres / self.scale

    def m(self, pixels: float) -> float:
        return pixels * self.scale

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self, with_trace: bool = True) -> dict:
        d = {"image_path": self.image_path, "scale": self.scale,
             "scale_calibrated": self.scale_calibrated, "angles": list(self.angles),
             "dims_info": dict(self.dims_info), "scale_source": self.scale_source,
             "texts_raw": [[t, [list(q) for q in b]] for t, b in self.texts_raw]}
        for k in self.KINDS:
            d[k] = [asdict(o) for o in getattr(self, k)]
        if with_trace:
            d["trace"] = [[list(p) for p in pl] for pl in self.trace]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls()
        p.image_path = d.get("image_path", "")
        p.scale = float(d.get("scale", 0.03))
        p.scale_calibrated = bool(d.get("scale_calibrated", False))
        p.angles = list(d.get("angles", [0.0, 90.0]))
        p.dims_info = dict(d.get("dims_info", {}))
        p.scale_source = d.get("scale_source", "")
        p.texts_raw = [(t, [tuple(q) for q in b]) for t, b in d.get("texts_raw", [])]
        for k, C in cls.KINDS.items():
            items = []
            for o in d.get(k, []):
                o = dict(o)
                for key, val in list(o.items()):
                    if isinstance(val, list) and key in ("pts",):
                        o[key] = [tuple(v) for v in val]
                    elif isinstance(val, list) and key in ("p0", "p1", "p", "along", "across"):
                        o[key] = tuple(val)
                items.append(C(**o))
            setattr(p, k, items)
        if "trace" in d:
            p.trace = [[tuple(pt) for pt in pl] for pl in d["trace"]]
        return p

    def snapshot(self) -> dict:
        """Undo snapshot (trace excluded – it is never edited)."""
        return copy.deepcopy(self.to_dict(with_trace=False))

    def restore(self, snap: dict):
        trace = self.trace
        new = Project.from_dict(snap)
        new.trace = trace
        self.__dict__.update(new.__dict__)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def clear_detected(self):
        for k in self.KINDS:
            setattr(self, k, [])
        self.trace = []


def dist(a: Pt, b: Pt) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
