"""AI mode: trace the plan the way a person does, with Claude's vision.

The plan is cut into overlapping tiles.  Each tile is enlarged and gets a labelled
coordinate grid (original image pixels), exactly like tracing by hand.  Claude reads
the tile together with what the automatic engine already found there and the
dimensions read from the drawing, and returns every wall, door, window, column,
stair, furniture block and room label with absolute coordinates.  The tiles are
merged, snapped to the drawing directions and fitted to the written dimensions.
"""
from __future__ import annotations

import base64
import json
import math
import concurrent.futures as cf

import numpy as np
import cv2

from ..model import Wall, Column, Door, Window, Stair, Furn, Text
from ..blocks import LIB

MODEL = "claude-opus-5-5"
TILE = 440          # tile size, original px
OVERLAP = 70
ZOOM = 2.4
EST_COST_PER_TILE = 0.20   # USD, rough (image + context + reasoning + output)

BLOCKS = sorted(LIB)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["walls", "doors", "windows", "columns", "stairs", "furniture", "labels"],
    "properties": {
        "walls": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["x1", "y1", "x2", "y2", "t"],
            "properties": {"x1": {"type": "number"}, "y1": {"type": "number"},
                           "x2": {"type": "number"}, "y2": {"type": "number"},
                           "t": {"type": "number", "description": "thickness in metres"}}}},
        "doors": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["hinge_x", "hinge_y", "other_x", "other_y", "swing_x", "swing_y", "double"],
            "properties": {"hinge_x": {"type": "number"}, "hinge_y": {"type": "number"},
                           "other_x": {"type": "number"}, "other_y": {"type": "number"},
                           "swing_x": {"type": "number"}, "swing_y": {"type": "number"},
                           "double": {"type": "boolean"}}}},
        "windows": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["x1", "y1", "x2", "y2"],
            "properties": {"x1": {"type": "number"}, "y1": {"type": "number"},
                           "x2": {"type": "number"}, "y2": {"type": "number"}}}},
        "columns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["x", "y", "size"],
            "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                           "size": {"type": "number", "description": "metres"}}}},
        "stairs": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["start_x", "start_y", "end_x", "end_y", "width_x", "width_y"],
            "properties": {"start_x": {"type": "number"}, "start_y": {"type": "number"},
                           "end_x": {"type": "number"}, "end_y": {"type": "number"},
                           "width_x": {"type": "number"}, "width_y": {"type": "number"}}}},
        "furniture": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["block", "x", "y", "w", "d", "facing_deg"],
            "properties": {"block": {"type": "string", "enum": BLOCKS},
                           "x": {"type": "number"}, "y": {"type": "number"},
                           "w": {"type": "number", "description": "width in metres"},
                           "d": {"type": "number", "description": "depth in metres"},
                           "facing_deg": {"type": "number"}}}},
        "labels": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["x", "y", "text"],
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "text": {"type": "string"}}}},
    },
}

SYSTEM = """You are an expert architectural draftsman converting a scanned floor plan into clean CAD geometry, tracing it by hand piece by piece.

You receive one enlarged tile of the scan with a coordinate grid drawn on it. Grid labels are ORIGINAL image pixel coordinates (x to the right, y downward); thin lines every 10 px, labelled lines every 50 px. Every coordinate you return must be in these original pixel coordinates, read off the grid as precisely as you can (aim for 1-2 px).

Return everything whose centre lies inside the tile's core area (stated in the request); the tile shows a margin around it for context.

- walls: straight centre-line segments, one per straight run, with real thickness in metres (typical: exterior 0.25-0.35, interior 0.20, partitions 0.10-0.12, parapets 0.15). Stop a wall at each door opening (leave the opening empty). Windows sit inside walls, so walls run continuously under windows. Make walls meet: at corners and T-junctions extend the centre lines so they touch. Hatched or solid-filled bands are walls too. Dashed lines (axes, beams, roof edges, dimension lines) are not walls.
- doors: hinge jamb point, the other jamb point (both on the wall centre line at the opening edges), and any point on the side the leaf swings into. double=true for two leaves.
- windows: the two ends of the glazed part, on the wall centre line.
- columns: solid filled squares, centre + size in metres.
- stairs: start of the flight, end of the flight (the up direction), and a point on the far side giving the flight width.
- furniture and fixtures: pick the closest block from the enum; centre, width and depth in metres, and facing_deg = direction the front faces in image coordinates (0 = right, 90 = down, 180 = left, 270 = up). Beds: the front is the foot end. Chairs and sofas: the front is where you sit facing.
- labels: room names written on the plan.

Be complete: trace every wall, including thin partitions inside bathrooms and kitchens. The automatic detection given in the request is a hint and may be wrong or incomplete: correct it from what you see. Written dimensions are authoritative for lengths."""


def tiles_for(shape):
    H, W = shape[:2]
    step = TILE - OVERLAP
    xs = list(range(0, max(1, W - OVERLAP), step)) or [0]
    ys = list(range(0, max(1, H - OVERLAP), step)) or [0]
    out = []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(W, x0 + TILE), min(H, y0 + TILE)
            out.append((x0, y0, x1, y1))
    return out


def _ink_fraction(gray, box):
    x0, y0, x1, y1 = box
    c = gray[y0:y1, x0:x1]
    return float((c < 160).mean()) if c.size else 0.0


def useful_tiles(gray):
    return [b for b in tiles_for(gray.shape) if _ink_fraction(gray, b) > 0.004]


def estimate_cost(gray):
    n = len(useful_tiles(gray))
    return n, n * EST_COST_PER_TILE


def render_tile(gray, box):
    x0, y0, x1, y1 = box
    c = gray[y0:y1, x0:x1]
    img = cv2.resize(c, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = (img * 0.85 + 255 * 0.15).astype(np.uint8)
    for x in range((x0 // 10 + 1) * 10, x1, 10):
        X = int((x - x0) * ZOOM)
        major = x % 50 == 0
        cv2.line(img, (X, 0), (X, img.shape[0]), (0, 0, 255) if major else (255, 205, 205), 1)
        if major:
            cv2.putText(img, str(x), (X + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(x), (X + 2, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    for y in range((y0 // 10 + 1) * 10, y1, 10):
        Y = int((y - y0) * ZOOM)
        major = y % 50 == 0
        cv2.line(img, (0, Y), (img.shape[1], Y), (0, 0, 255) if major else (255, 205, 205), 1)
        if major:
            cv2.putText(img, str(y), (2, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(y), (img.shape[1] - 40, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return base64.standard_b64encode(buf.tobytes()).decode('ascii')


def _inside(p, box, pad=0):
    return box[0] - pad <= p[0] <= box[2] + pad and box[1] - pad <= p[1] <= box[3] + pad


def tile_context(project, box):
    """Hints for one tile: auto-detected walls/doors and the dimensions read there."""
    s = project.scale
    walls = [dict(x1=round(w.pts[0][0], 1), y1=round(w.pts[0][1], 1), x2=round(w.pts[-1][0], 1),
                  y2=round(w.pts[-1][1], 1), t=w.t)
             for w in project.walls if any(_inside(q, box, 10) for q in w.pts)]
    doors = [dict(p0=[round(v, 1) for v in d.p0], p1=[round(v, 1) for v in d.p1], double=d.double)
             for d in project.doors if _inside(d.p0, box)]
    dims = [dict(text=d.text, p0=[round(v, 1) for v in d.p0], p1=[round(v, 1) for v in d.p1])
            for d in project.dims if _inside(d.p0, box, 30) or _inside(d.p1, box, 30)]
    return dict(walls=walls[:120], doors=doors[:60], dims=dims[:60])


def ask_tile(client, gray, project, box, core, effort="high"):
    ctx = tile_context(project, box)
    req = (f"Tile area (original px): x {box[0]}..{box[2]}, y {box[1]}..{box[3]}.\n"
           f"Core area (return items whose centre is inside it): x {core[0]}..{core[2]}, y {core[1]}..{core[3]}.\n"
           f"Scale: {project.scale:.5f} metres per original pixel (1 m = {1 / project.scale:.1f} px).\n"
           f"Main drawing directions (deg, image coords): {', '.join(f'{a:.2f}' for a in project.angles)} "
           f"(and their perpendiculars). Keep walls exactly on these directions.\n"
           f"Automatic detection in this tile (hint, may be wrong or incomplete):\n{json.dumps(ctx, separators=(',', ':'))}\n"
           f"Trace this tile completely.")
    import anthropic
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": render_tile(gray, box)}},
        {"type": "text", "text": req},
    ]
    common = dict(model=MODEL, max_tokens=48000, system=SYSTEM,
                  output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                  messages=[{"role": "user", "content": content}])
    try:
        # refusal fallback on the server side (another model finishes the tile if this one declines)
        with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                         **common) as stream:
            msg = stream.get_final_message()
    except anthropic.BadRequestError:
        # an optional parameter was rejected: retry with the plain request
        with client.messages.stream(**common) as stream:
            msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("refused")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("output too long")
    text = next((b.text for b in msg.content if b.type == "text"), "")
    data = json.loads(text)
    usage = msg.usage
    return data, (getattr(usage, 'input_tokens', 0) or 0, getattr(usage, 'output_tokens', 0) or 0)


def _core(box, shape):
    H, W = shape[:2]
    h = OVERLAP / 2
    x0, y0, x1, y1 = box
    return (x0 + (h if x0 > 0 else 0), y0 + (h if y0 > 0 else 0),
            x1 - (h if x1 < W else 0), y1 - (h if y1 < H else 0))


def run_ai(project, gray, api_key=None, workers=4, effort="high", progress=None, ask=None):
    """Trace the whole plan with Claude. `ask` can replace the API call (testing)."""
    say = progress or (lambda f, m='': None)
    if ask is None:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        ask = lambda box, core: ask_tile(client, gray, project, box, core, effort)
    boxes = useful_tiles(gray)
    results, errors, tokens = [], [], [0, 0]
    say(0.0, 'ai')
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ask, b, _core(b, gray.shape)): b for b in boxes}
        for k, f in enumerate(cf.as_completed(futs)):
            b = futs[f]
            try:
                data, (ti, to) = f.result()
                results.append((b, _core(b, gray.shape), data))
                tokens[0] += ti; tokens[1] += to
            except Exception as e:  # keep going, report at the end
                errors.append(f"{b}: {e}")
            say((k + 1) / len(boxes) * 0.9, 'ai')
    merge_results(project, results)
    say(0.95, 'fit')
    if project.dims:
        from .fit import fit_to_dimensions
        fit_to_dimensions(project)
    cost = tokens[0] * 4e-6 + tokens[1] * 20e-6
    project.ai_info = dict(tiles=len(boxes), ok=len(results), errors=errors[:10], cost=round(cost, 2),
                           walls=len(project.walls), doors=len(project.doors), windows=len(project.windows),
                           furniture=len(project.furniture))
    say(1.0, 'done')
    return project.ai_info


def _in_core(p, core):
    return core[0] <= p[0] < core[2] and core[1] <= p[1] < core[3]


def merge_results(project, results):
    from . import detect as D
    s = project.scale
    walls, doors, windows, cols, stairs, furn, labels = [], [], [], [], [], [], []
    for box, core, data in results:
        for w in data.get('walls', []):
            a, b = np.array([w['x1'], w['y1']]), np.array([w['x2'], w['y2']])
            if np.linalg.norm(b - a) * s < 0.15 or not _in_core((a + b) / 2, core) and not (
                    _inside(a, core) or _inside(b, core)):
                continue
            a2, b2 = D._snap_dir(a, b, project.angles, tol=4.0)
            walls.append(Wall(pts=[tuple(map(float, a2)), tuple(map(float, b2))], t=float(min(max(w['t'], 0.05), 0.6))))
        for d in data.get('doors', []):
            h, o = (d['hinge_x'], d['hinge_y']), (d['other_x'], d['other_y'])
            if not _in_core(((h[0] + o[0]) / 2, (h[1] + o[1]) / 2), core):
                continue
            u = np.array(o) - np.array(h); n = np.array([-u[1], u[0]])
            side = 1 if np.dot(np.array([d['swing_x'], d['swing_y']]) - np.array(h), n) >= 0 else -1
            doors.append(Door(tuple(map(float, h)), tuple(map(float, o)), side, bool(d['double'])))
        for w in data.get('windows', []):
            a, b = (w['x1'], w['y1']), (w['x2'], w['y2'])
            if _in_core(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), core):
                windows.append(Window(tuple(map(float, a)), tuple(map(float, b)), 0.25))
        for c in data.get('columns', []):
            if _in_core((c['x'], c['y']), core):
                cols.append(Column(float(c['x']), float(c['y']), float(c['size'])))
        for st in data.get('stairs', []):
            if not _in_core(((st['start_x'] + st['end_x']) / 2, (st['start_y'] + st['end_y']) / 2), core):
                continue
            from ..ui.canvas import PlanView  # geometry helper only
            stairs.append(PlanView._stair_from((st['start_x'], st['start_y']), (st['end_x'], st['end_y']),
                                               (st['width_x'], st['width_y'])))
        for f in data.get('furniture', []):
            if f['block'] in LIB and _in_core((f['x'], f['y']), core):
                furn.append(Furn(f['block'], float(f['x']), float(f['y']), float(f['w']), float(f['d']),
                                 float(f['facing_deg']) % 360, False, LIB[f['block']][0]))
        for t in data.get('labels', []):
            if _in_core((t['x'], t['y']), core) and t['text'].strip():
                labels.append(Text(float(t['x']), float(t['y']), t['text'].strip().upper(), 0.30))

    def dedupe(items, key, r):
        out = []
        for it in items:
            p = np.array(key(it))
            if all(np.linalg.norm(p - np.array(key(o))) > r for o in out):
                out.append(it)
        return out
    r = 0.25 / s
    doors = dedupe(doors, lambda d: ((d.p0[0] + d.p1[0]) / 2, (d.p0[1] + d.p1[1]) / 2), r)
    windows = dedupe(windows, lambda w: ((w.p0[0] + w.p1[0]) / 2, (w.p0[1] + w.p1[1]) / 2), r)
    cols = dedupe(cols, lambda c: (c.x, c.y), r)
    furn = dedupe(furn, lambda f: (f.x, f.y), 0.15 / s)
    labels = dedupe(labels, lambda t: (t.x, t.y), 0.8 / s)
    # walls: quantise thickness, merge collinear pieces across tiles, close junctions
    TQ = np.array([0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    for w in walls:
        w.t = float(TQ[np.argmin(abs(TQ - w.t))])
    walls = D._merge_wall_list(walls, s)
    walls = D.join_walls(walls, s, 0.35)
    # window depth = wall it sits in
    from ..geometry import wall_polygon
    from shapely.geometry import Point
    polys = [wall_polygon(w, s) for w in walls]
    for wi in windows:
        m = Point((wi.p0[0] + wi.p1[0]) / 2, (wi.p0[1] + wi.p1[1]) / 2)
        best = min(((pg.distance(m), w.t) for pg, w in zip(polys, walls)), default=(1e9, 0.25))
        if best[0] < 0.3 / s:
            wi.t = best[1]
    project.walls, project.doors, project.windows = walls, doors, windows
    project.columns, project.stairs, project.furniture = cols, stairs, furn
    project.texts = labels
