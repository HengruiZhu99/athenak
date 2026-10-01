#!/usr/bin/env python3
"""Compare overlapping gamma-five runs, retaining actual sample time offsets."""
import argparse
import json
from pathlib import Path
from analyze import analyze, numeric_rows
from diagnostics import slice_peaks


def parameters(run):
    blocks, name = {}, None
    for line in (run / 'input.athinput').read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<') and line.endswith('>'):
            name = line[1:-1]
            blocks.setdefault(name, {})
        elif '=' in line and name:
            key, value = line.split('=', 1)
            blocks[name][key.strip()] = value.strip()
    return blocks


def compare(coarse, fine):
    decks = [parameters(run) for run in (coarse, fine)]
    required = {'problem': ['punc_ADM_mass', 'punc_velocity_x1', 'pgen_name'],
                'z4c': ['telegraph_lapse', 'telegraph_tau', 'telegraph_kappa',
                        'damp_kappa1', 'damp_kappa2'],
                'time': ['integrator', 'cfl_number'],
                'mesh': ['nx1', 'nx2', 'nx3', 'x1min', 'x1max', 'x2min',
                         'x2max', 'x3min', 'x3max']}
    matched = {}
    for block, keys in required.items():
        for key in keys:
            a, b = (deck[block][key] for deck in decks)
            if a != b:
                raise ValueError('Different physical setting: ' + block + '/' + key)
            matched[block + '/' + key] = a
    spacings = []
    for deck in decks:
        mesh = deck['mesh']
        spacings.append((float(mesh['x1max']) - float(mesh['x1min'])) /
                        (int(mesh['nx1']) * 2**int(deck['z4c']['co_0_reflevel'])))
    if abs(spacings[0] / spacings[1] - 2) > 1e-12:
        raise ValueError('Expected factor-two finest-grid resolution difference')
    histories = [numeric_rows(next(run.glob('*.hst'))) for run in (coarse, fine)]
    overlap_end = min(rows[-1][0] for rows in histories)
    norms = []
    for t in [0, .1, 1, 2, 3, 4, 4.5, 5, 10, 15, 20]:
        if t > overlap_end:
            continue
        a, b = (min(rows, key=lambda r: abs(r[0] - t)) for rows in histories)
        if max(abs(a[0] - t), abs(b[0] - t)) > .06:
            continue
        norms.append(dict(target_time=t, coarse_time=a[0], fine_time=b[0],
                          fine_minus_coarse_time=b[0] - a[0],
                          coarse_H_norm2=a[3], fine_H_norm2=b[3],
                          fine_over_coarse_H_norm2=b[3] / a[3] if a[3] else None))
    peaks = [slice_peaks(run) for run in (coarse, fine)]
    slices = []
    for t in range(int(overlap_end) + 1):
        a, b = (min(rows, key=lambda r: abs(r['time'] - t)) for rows in peaks)
        if max(abs(a['time'] - t), abs(b['time'] - t)) > .01:
            continue
        slices.append(dict(target_time=t, coarse_peak=a, fine_peak=b,
                           coarse_over_fine_abs_H=a['max_abs_H'] / b['max_abs_H']))
    return dict(coarse=analyze(coarse), fine=analyze(fine), finest_dx=spacings,
                matching_physical_parameters=matched, overlap_end=overlap_end,
                integrated_norm_comparison=norms, x_axis_peak_comparison=slices,
                note='Partial comparison when either run is unfinished. Chi>=0.0625 '
                     'is a coordinate mask, not horizon excision. Adding an AMR level '
                     'does not uniformly refine the entire domain. The slice maxima '
                     'can occur at different locations; no convergence order is fitted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coarse', type=Path)
    parser.add_argument('fine', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.coarse, args.fine), indent=2, allow_nan=False))
