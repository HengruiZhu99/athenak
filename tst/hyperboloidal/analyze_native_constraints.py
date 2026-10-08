#!/usr/bin/env python3
"""Radial constraint budgets from masked native hyperboloidal binary output.

Uses every active cell and the same unweighted conformal norms as native history.
This diagnoses error location; it does not certify stability or convergence.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('hyp_binary', ROOT / 'vis/python/bin_convert.py')
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)


def analyze(path, edges):
    data = READER.read_binary(str(path))
    if data['n_mbs'] != 1 or data['mb_logical'][0, 3] != 0:
        raise ValueError('requires one uniform native prototype block')
    mask_values = np.asarray(data['mb_data']['z4c_active'])[0]
    if not np.isin(mask_values, [0, 1]).all():
        raise ValueError('invalid active mask')
    mask = mask_values.astype(bool)
    coordinates = []
    for axis in range(3):
        lower, upper = data['mb_geometry'][0, 2*axis:2*axis+2]
        step = (upper-lower)/data[f'nx{axis+1}_mb']
        start = data['mb_index'][0, 2*axis]
        first = lower+(start+0.5)*step
        coordinates.append(first+np.arange(mask.shape[2-axis])*step)
    z, y, x = np.meshgrid(*coordinates[::-1], indexing='ij')
    radius2 = x*x+y*y+z*z
    if not np.array_equal(mask, radius2 < 1):
        raise ValueError('mask does not match the unit spherical domain')
    radius = np.sqrt(radius2[mask])
    xyz = np.column_stack([a[mask] for a in (x, y, z)])
    edges = np.asarray(edges, dtype=float)
    if (len(edges) < 2 or not np.isfinite(edges).all() or (np.diff(edges) <= 0).any()
            or not len(radius) or edges[0] > radius.min() or edges[-1] <= radius.max()):
        raise ValueError('radial bins must cover all active nodes')
    squares = {}
    for label, key in [('H', 'con_H'), ('M', 'con_M'), ('Z', 'con_Z')]:
        value = np.asarray(data['mb_data'][key], dtype=float)[0][mask]
        if not np.isfinite(value).all() or (label != 'H' and (value < 0).any()):
            raise ValueError(f'invalid active {label} constraints')
        squares[label] = value*value if label == 'H' else value
    result = {'path': str(path), 'time': data['time'], 'cycle': data['cycle'],
              'active_cells': len(radius), 'global': {}, 'radial_bins': []}
    for label, value in squares.items():
        peak = int(np.argmax(value))
        result['global'][label] = {'rms': float(np.sqrt(value.mean())),
                                   'max': float(np.sqrt(value[peak])),
                                   'max_radius': float(radius[peak]),
                                   'max_xyz': xyz[peak].tolist()}
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (radius >= lower) & (radius < upper)
        row = {'r_min': float(lower), 'r_max': float(upper),
               'cells': int(selected.sum())}
        for label, value in squares.items():
            total = float(value.sum())
            row[label] = {'rms': float(np.sqrt(value[selected].mean()))
                          if selected.any() else None,
                          'squared_norm_fraction': float(value[selected].sum()) / total
                          if total else 0.0}
        result['radial_bins'].append(row)
    assert sum(row['cells'] for row in result['radial_bins']) == len(radius)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--edges', nargs='+', type=float,
                        default=[0, .25, .5, .75, .85, .9, .95, 1])
    args = parser.parse_args()
    print(json.dumps([analyze(path, args.edges) for path in args.files], indent=2,
                     allow_nan=False))


if __name__ == '__main__':
    main()
