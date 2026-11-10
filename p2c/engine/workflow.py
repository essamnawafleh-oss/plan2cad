"""One import path for the UI, the command line and the benchmark: file -> working image."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import imageio
from .preprocess import PreprocessOptions, preprocess


@dataclass
class ImportOptions(PreprocessOptions):
    clean: bool = True              # legacy option: background cleaning (now = lighting 'auto')


@dataclass
class ImportResult:
    gray: np.ndarray
    wall_view: np.ndarray | None
    H_in_to_work: np.ndarray
    transform: dict = field(default_factory=dict)
    original: np.ndarray | None = None     # colour input (for the original/processed comparison)
    processed_color: np.ndarray | None = None


def import_image(path, opts: ImportOptions | None = None, progress=None) -> ImportResult:
    o = opts or ImportOptions()
    if not o.clean:
        o.lighting = 'off'
    color = imageio.imread_color(path)          # OpenCV applies EXIF orientation while decoding
    r = preprocess(color, o, progress=progress)
    try:
        wv = imageio.wall_view(r.color)
    except Exception:
        wv = None
    tr = r.record(); tr['source'] = path
    return ImportResult(gray=r.gray, wall_view=wv, H_in_to_work=r.H, transform=tr, original=color,
                        processed_color=r.color)
