#!/usr/bin/env python3
"""Compare point-matched native scri geometry and constraint diagnostics at common endpoints."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tst/z4c_hyperboloidal'))
sys.path.insert(0, str(root / 'vis/python'))
from scri_reader import read
from scri_geometry import endpoint_jets, spacetime_metric
from nonradiative_native_analysis import metric_rate
from nonradiative_tensor_attribution import tensor


def diagnostics(directory, scri_only=False, output_number=3):
    anchor = sorted(directory.glob(f'*.o{output_number}.*.r00000000.scri'))[-1]
    series = []
    for path in sorted(directory.glob(anchor.name.replace('r00000000', 'r????????'))):
        record = read(path)
        epoch = read(path.parent / record['epoch'])
        ids = epoch['surface_ids'][:, [0, 2, 3]]
        x = epoch['surface_geometry'][:, :3]
        jets = endpoint_jets(record, epoch)
        native = spacetime_metric(jets, x)
        g = native[..., 0]
        dg = np.empty((len(x), 4, 4, 4))
        dg[:, 0] = metric_rate(jets, x, record['projected_metric_rates'])
        dg[:, 1:] = np.moveaxis(native[..., 1:4], -1, -3)
        tracefree = tensor(g, dg, x)
        # The tangential trace of T_ab divided by n^t is the difference
        # between the angular expansion and the four-dimensional clock source.
        inverse = np.linalg.inv(g[:, 1:, 1:])
        radial = np.einsum('qij,qj->qi', inverse, x)
        norm = np.einsum('qi,qi->q', radial, x)
        tangent = inverse - radial[:, :, None]*radial[:, None, :]/norm[:, None, None]
        nt = -np.einsum('qi,qi->q', np.linalg.inv(g)[:, 0, 1:], x)
        mismatch = np.einsum('qij,qij->q', tangent, tracefree[:, 1:, 1:])/nt
        series.append((ids, tracefree, mismatch, record['time'][0]))
    ids = np.concatenate([row[0] for row in series])
    order = np.lexsort(ids.T[::-1])
    time = series[0][3]
    if any(row[3] != time for row in series):
        raise ValueError('ranks wrote different scri endpoints')
    values = dict(regularity_tensor=np.concatenate([row[1] for row in series])[order],
                  clock_mismatch=np.concatenate([row[2] for row in series])[order])
    if not scri_only:
        history = np.atleast_2d(np.loadtxt(next(directory.glob('*.z4c_hyperboloidal.hst'))))[-1]
        if time != history[0]:
            raise ValueError('scri and constraint diagnostics have different endpoints')
        for i, name in enumerate(['hamiltonian_rms', 'momentum_rms', 'theta_rms', 'z_rms']):
            values[name] = history[i + 3]
    return ids[order], time, values


def compare(base, scri_only=False):
    ids, time, reference = diagnostics(base / 'uniform-32',scri_only)
    differences = []
    for steps in (2, 4, 8):
        other_ids, other_time, state = diagnostics(base / f'mixed-{steps}',scri_only)
        if not np.array_equal(ids, other_ids) or time != other_time:
            raise ValueError('native diagnostic points or times do not match')
        differences.append({key: float(np.sqrt(np.mean((value-reference[key])**2)))
                            for key, value in state.items()})
    orders = {key: [float(np.log2(differences[i][key]/differences[i+1][key]))
                    if differences[i][key] and differences[i+1][key] else None for i in (0, 1)]
              for key in reference}
    return dict(time=float(time), points=len(ids), rms_differences=differences, orders=orders,
                reference_max_abs={key: float(np.max(abs(value))) for key,value in reference.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--scri-only', action='store_true')
    args = parser.parse_args()
    result = compare(args.study,args.scri_only)
    (args.study/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
