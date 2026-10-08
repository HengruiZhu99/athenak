#!/usr/bin/env python3
"""Single-core evolution checks; extended runs audit convergence and scri closure.

Only the standard library is required. Output is retained if --output is supplied.
This tests a spherical Minkowski gauge pulse, not a puncture or 3D mesh boundary.
"""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import tempfile


def read(path):
    with open(path) as stream:
        rows = [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    assert rows and all(math.isfinite(v) for row in rows for v in row.values())
    return rows


def run(executable, directory, name, n, end, *options):
    prefix = directory / name
    with open(str(prefix) + '.log', 'w') as log:
        subprocess.run([str(executable), '--n', str(n), '--t', str(end),
                        '--output', str(prefix), *options], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    rows = read(str(prefix) + '-diagnostics.csv')
    fields = read(str(prefix) + '-fields.csv')
    assert abs(rows[-1]['t'] - end) < 1e-12
    assert all(row['min_chi'] > 0 and row['min_alpha'] > 0 for row in rows)
    print(name, {k: rows[-1][k] for k in ('H_L2', 'M_L2', 'max_deviation')},
          flush=True)
    return rows, fields


def difference(coarse, fine, field):
    # Fourth-order interpolation to identical radii; exclude only the two
    # endpoint samples requiring ghosts. Constraint norms include every cell.
    errors = []
    for i in range(1, len(coarse) - 1):
        j = 2*i
        value = (-fine[j-1][field] + 9*fine[j][field]
                 + 9*fine[j+1][field] - fine[j+2][field])/16
        errors.append((coarse[i][field] - value)**2)
    return math.sqrt(sum(errors)/len(errors))


def check(executable, directory, extended):
    reference, fields = run(executable, directory, 'reference', 16, .2,
                            '--amplitude', '0')
    assert all(row['max_deviation'] == 0 for row in reference)
    assert all(row['scri_pole_max'] == 0 and row['scri_lapse_pole'] == 0
               for row in reference)
    assert max(abs(row['H']) for row in fields) < 1e-12
    pulse, _ = run(executable, directory, 'pulse', 64, 3)
    assert max(row['max_deviation'] for row in pulse) < .1
    assert pulse[-1]['max_deviation'] < pulse[0]['max_deviation']
    assert pulse[-1]['H_L2'] < 5e-4 and pulse[-1]['M_L2'] < 5e-4
    for option in ('--amplitude', '--t', '--slicing'):
        invalid = subprocess.run([str(executable), option, 'nan'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert invalid.returncode != 0
    # Initial-data audit; the separate check_puncture.py runs the live-gauge study.
    trumpet = [run(executable, directory, f'trumpet-id-{n}', n, 0,
                   '--mass', '0.05', '--amplitude', '0')[0][0]
               for n in (128, 256, 512)]
    for key in ('H_L2', 'M_L2'):
        assert all(b[key] < a[key]/3 for a, b in zip(trumpet, trumpet[1:]))
    for row in trumpet:
        assert abs(row['mass_near_half'] - .05) < 1e-6
        assert abs(row['horizon_areal_radius'] - .1) < .001
    equilibrium, _ = run(executable, directory, 'trumpet-equilibrium', 64, 1,
                         '--mass', '.05', '--amplitude', '0',
                         '--analytic-trumpet', '--fixed-lapse', '--fixed-shift')
    assert max(row['H_L2'] for row in equilibrium) < 1e-8
    assert max(row['M_L2'] for row in equilibrium) < 1e-8
    assert max(abs(row['mass_near_half'] - .05) for row in equilibrium) < 1e-9
    # Plain differences still see the puncture profile's truncation error.
    assert equilibrium[0]['H_raw_L2'] > 1
    assert abs(equilibrium[-1]['H_raw_L2'] / equilibrium[0]['H_raw_L2'] - 1) < 1e-8
    if not extended:
        return
    resolutions = [64, 128, 256, 512]
    results = [run(executable, directory, f'gauge-{n}', n, 1) for n in resolutions]
    for key in ('H_L2', 'M_L2'):
        norms = [result[0][-1][key] for result in results]
        assert all(b < a/1.5 for a, b in zip(norms, norms[1:])), (key, norms)
        print(key, 'orders', [math.log2(a/b) for a, b in zip(norms, norms[1:])])
    for field in list(results[0][1][0])[1:9]:
        errors = [difference(a[1], b[1], field) for a, b in zip(results, results[1:])]
        assert all(b < a/1.5 for a, b in zip(errors, errors[1:])), (field, errors)
        print(field, 'self-convergence orders',
              [math.log2(a/b) for a, b in zip(errors, errors[1:])])
    long, _ = run(executable, directory, 'long', 128, 10)
    assert long[-1]['H_L2'] < 1e-9 and long[-1]['M_L2'] < 1e-9
    assert long[-1]['max_deviation'] < 1e-6
    for degree in (3, 5):
        boundary, _ = run(executable, directory, f'boundary-{degree}', 128, 3,
                          '--extrapolation', str(degree))
        assert boundary[-1]['H_L2'] < 2e-4 and boundary[-1]['M_L2'] < 2e-4
    print('PASS: bounded pulse, decreasing errors, long decay, boundary alternatives')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    executable = args.executable.resolve()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        check(executable, args.output.resolve(), args.extended)
    else:
        with tempfile.TemporaryDirectory(prefix='hyperboloidal-') as temporary:
            check(executable, Path(temporary), args.extended)
