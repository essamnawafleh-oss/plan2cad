"""Markdown comparison of benchmark result files.

    python bench/report.py bench_results/baseline.json bench_results/v2.json
"""
import json
import os
import sys

import numpy as np

KEYS = [('wall_f1', 'Wall F1', '{:.3f}'), ('wall_precision', 'Precision', '{:.3f}'), ('wall_recall', 'Recall', '{:.3f}'),
        ('wall_iou', 'IoU', '{:.3f}'), ('endpoint_err_m', 'Endpoint err (m)', '{:.3f}'),
        ('thick_err_m', 'Thickness err (m)', '{:.3f}'), ('connectivity', 'Junctions connected', '{:.2f}'),
        ('missing_walls', 'Missing walls', '{:.2f}'), ('false_walls', 'False walls', '{:.2f}'),
        ('door_f1', 'Door F1', '{:.3f}'), ('window_f1', 'Window F1', '{:.3f}'),
        ('t_run', 'Pipeline time (s)', '{:.1f}'), ('t_nn', 'NN time (s)', '{:.1f}'), ('t_import', 'Import time (s)', '{:.2f}'),
        ('t_dxf', 'DXF time (s)', '{:.2f}'), ('peak_mb', 'Peak memory (MB)', '{:.0f}')]


def load(f):
    return [r for r in json.load(open(f))]


def mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None and not r.get('error')]
    return float(np.mean(v)) if v else None


def group(rows):
    g = {}
    for r in rows:
        if r.get('case', '').startswith('none'):
            continue
        kind = r.get('kind') or ('real' if str(r.get('case', '')).startswith('real') else '?')
        key = (kind, 'calibrated' if r.get('calibrated') else 'uncalibrated')
        g.setdefault(key, []).append(r)
    return g


def main(files):
    data = {os.path.splitext(os.path.basename(f))[0]: load(f) for f in files}
    names = list(data)
    print('# Benchmark comparison\n')
    print('Files: ' + ', '.join(f'`{f}`' for f in files) + '\n')
    errs = {n: sum(1 for r in rows if r.get('error')) for n, rows in data.items()}
    print('Runs with errors: ' + ', '.join(f'{n}: {e}' for n, e in errs.items()) + '\n')
    keys = sorted({k for rows in data.values() for k in group(rows)})
    for key in keys:
        print(f'## {key[0]} drawings, {key[1]}\n')
        print('| Metric | ' + ' | '.join(names) + ' |')
        print('|---|' + '---|' * len(names))
        n_runs = [len(group(data[n]).get(key, [])) for n in names]
        print('| runs | ' + ' | '.join(str(x) for x in n_runs) + ' |')
        for k, label, fmt in KEYS:
            vals = [mean(group(data[n]).get(key, []), k) for n in names]
            if all(v is None for v in vals):
                continue
            print(f'| {label} | ' + ' | '.join('-' if v is None else fmt.format(v) for v in vals) + ' |')
        if key[1] == 'uncalibrated':
            vals = [mean(group(data[n]).get(key, []), 'scale_err') for n in names]
            print('| Scale error (rel.) | ' + ' | '.join('-' if v is None else f'{v:.3f}' for v in vals) + ' |')
        print()
    print('## Per case (calibrated, wall F1 / IoU)\n')
    cases = sorted({(r['case'], r.get('seed')) for rows in data.values() for r in rows if r.get('calibrated')})
    print('| Case | seed | ' + ' | '.join(names) + ' |')
    print('|---|---|' + '---|' * len(names))
    for c, sd in cases:
        cells = []
        for n in names:
            r = [x for x in data[n] if x['case'] == c and x.get('seed') == sd and x.get('calibrated')]
            if not r or r[0].get('error'):
                cells.append('error' if r else '-')
            else:
                cells.append(f"{r[0].get('wall_f1', float('nan')):.3f} / {r[0].get('wall_iou', float('nan')):.3f}")
        print(f'| {c} | {sd} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(sys.argv[1:])
