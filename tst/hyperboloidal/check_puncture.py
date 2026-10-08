#!/usr/bin/env python3
"""Finite-duration live-gauge puncture study, retaining whole-domain diagnostics.

By default run 128/256/512 cells to t=5 on one host thread, sequentially.
--prefixes audits three existing histories/snapshots instead of rerunning them.
This is a convergence experiment, not an assertion of asymptotic stability.
"""
import argparse
import math
from pathlib import Path

from check_evolution import difference, read, run


def audit(results):
    for rows, fields in results:
        assert len(fields) in (128, 256, 512)
        assert abs(rows[-1]['t'] - 5) < 1e-12
        assert all(row['min_chi'] > 0 and row['min_alpha'] > 0 for row in rows)
        assert max(abs(row['mass_near_half'] - .05) for row in rows) < .0025
        assert max(abs(row['horizon_areal_radius'] - .1) for row in rows) < .005
        assert rows[0]['H_L2'] < 1e-8 and rows[0]['M_L2'] < 1e-8
        assert rows[0]['H_raw_L2'] > .1
    assert [len(fields) for _, fields in results] == [128, 256, 512]
    for key in ('H_L2', 'M_L2', 'H_raw_L2', 'M_raw_L2', 'Z_L2', 'Theta_L2'):
        values = [rows[-1][key] for rows, _ in results]
        orders = [math.log2(a/b) for a, b in zip(values, values[1:])]
        print(key, values, 'orders', orders)
        assert all(order > 0 for order in orders), key
    for key in ('H_L2', 'M_L2', 'Z_L2', 'Theta_L2'):
        peaks = [max(row[key] for row in rows) for rows, _ in results]
        print(key, 'sampled time maxima', peaks)
        assert all(b < a for a, b in zip(peaks, peaks[1:])), key
    for key, expected in [('mass_near_half', .05), ('horizon_areal_radius', .1)]:
        errors = [abs(rows[-1][key] - expected) for rows, _ in results]
        print(key, 'errors', errors)
        assert all(b < a for a, b in zip(errors, errors[1:])), key
    # Publish all component self-differences. A component may be too small or
    # underresolved for asymptotic order; do not replace this with a norm that
    # drops troublesome fields or interior samples.
    for field in list(results[0][1][0])[1:9]:
        a = difference(results[0][1], results[1][1], field)
        b = difference(results[1][1], results[2][1], field)
        print(field, 'self-differences', a, b, 'order', math.log2(a/b))
        assert b < a, field
    for rows, _ in results:
        if 'scri_pole_max' in rows[-1]:
            keys = ('scri_null_residual', 'scri_pole_max', 'scri_lapse_pole')
            print('scri endpoint', {k: rows[-1][k] for k in keys})
    poles = [rows[-1]['scri_pole_max'] for rows, _ in results
             if 'scri_pole_max' in rows[-1]]
    assert len(poles) >= 2 and all(p < 1e-5 for p in poles)
    assert poles[-1] < poles[0]
    print('PASS: finite-duration puncture evolution and decreasing errors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--output', type=Path, default=Path('puncture-results'))
    parser.add_argument('--prefixes', nargs=3, type=Path)
    args = parser.parse_args()
    if args.prefixes:
        results = [(read(str(p) + '-diagnostics.csv'), read(str(p) + '-fields.csv'))
                   for p in args.prefixes]
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        results = [run(args.executable.resolve(), args.output.resolve(),
                       f'live-{n}', n, 5,
                       '--mass', '.05', '--amplitude', '0', '--slicing', '2',
                       '--shift-driver', '.1', '--puncture-gauge', '--one-plus-log',
                       '--analytic-trumpet', '--lapse-scaled-damping', '--kappa1', '5')
                   for n in (128, 256, 512)]
    audit(results)
