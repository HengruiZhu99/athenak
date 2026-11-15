#!/usr/bin/env python3
"""Compare final native constraint norms at three resolutions and half dt.

These are norm decay rates toward zero, not Richardson orders of a solution
field. Horizon masks and discrete volume quadratures can differ between grids.
Use common-point field self-convergence and timestep/sampling checks separately.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scri_run_summary import history
from scri_radial_partition import partition

N = np.array([12., 16., 20.])
KEYS = ['H_M0_squared', 'M_M0_squared', 'Theta_M0', 'Z_M0']
UNITS = np.array([1/9, 1/9, 1/3, 1/3])


def decay(values, halfdt=None):
    values = np.asarray(values, float)
    assert values.shape == (3,) and np.isfinite(values).all() and np.all(values >= 0)
    # This floor suppresses interpretations of roundoff-sized norms; no value
    # is replaced by it in the data or plots.
    orders = [float(np.log(values[i]/values[i+1])/np.log(N[i+1]/N[i]))
              if min(values[i:i+2]) > 1e-14 else None for i in range(2)]
    result = dict(values=values.tolist(), pairwise_decay_orders=orders,
                  strictly_decreasing=bool(np.all(np.diff(values) < 0)),
                  meaning='Decay of measured constraint RMS toward zero, not solution-field self-convergence.')
    if halfdt is not None:
        assert np.isfinite(halfdt) and halfdt >= 0
        change = abs(float(values[0]-halfdt))
        spatial = abs(float(values[0]-values[1]))
        result.update(N12_half_dt=float(halfdt), N12_dt_change=change,
                      N12_dt_change_relative_to_norm=change/values[0] if values[0] else None,
                      N12_dt_change_over_N12_N16_norm_change=change/spatial if spatial else None)
    return result


def read(root):
    root = Path(root)
    receipt = json.loads((root/'final-checkpoint.json').read_text())
    identity = receipt['identity']
    match = re.search(r';n=(\d+),(\d+),(\d+);', identity)
    assert match and len(set(match.groups())) == 1
    n = int(match.group(1))
    common = re.sub(r';n=\d+,\d+,\d+|;p.requested_dt=[^;]*', '', identity)
    paths = [next(root.glob('*.z4c_hyperboloidal.hst'))]
    paths += [next(root.glob(f'*.scri-shell-{i}.hst')) for i in range(4)]
    metrics, hashes = {}, {}
    for i, path in enumerate(paths):
        a, _ = history(path)
        assert abs(3*a[-1,0]-receipt['time_M0']) < 1e-9, ('unmatched final history', path)
        columns = a[-1,8:12] if i == 0 else a[-1,3:7]
        metrics['exterior' if i == 0 else 'shell_'+str(i-1)] = dict(zip(KEYS, columns*UNITS))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    radial = partition(root)
    assert abs(radial['time_M0'][-1]-receipt['time_M0']) < 1e-9
    metrics['inner_exterior'] = dict(zip(KEYS,radial['rms_by_time_region_constraint'][-1][0]))
    return dict(n=n, identity=common, time_M0=receipt['time_M0'], dt_native=receipt['dt_native'],
                metrics=metrics, histories=hashes, directory=str(root.resolve()),
                radial_partition={k:radial[k] for k in ['maximum_horizon_radius_bound',
                    'final_subtraction_conditioning','source_hashes']},
                final_squared_integral_fractions=radial['squared_integral_fractions'][-1])


def compare(roots, halfdt=None):
    runs = [read(root) for root in roots]
    assert [r['n'] for r in runs] == [12,16,20]
    assert len(set(r['identity'] for r in runs)) == 1, 'different formulation/settings'
    times = [r['time_M0'] for r in runs]
    assert max(times)-min(times) < 1e-9, ('unmatched physical times', times)
    # Main sequence must follow the documented dt proportional to 1/N.
    np.testing.assert_allclose([r['dt_native']*r['n'] for r in runs],
                               runs[0]['dt_native']*12, rtol=1e-14, atol=0)
    half = read(halfdt) if halfdt else None
    if half:
        assert half['n'] == 12 and half['identity'] == runs[0]['identity']
        assert abs(half['time_M0']-times[0]) < 1e-9
        np.testing.assert_allclose(2*half['dt_native'], runs[0]['dt_native'], rtol=1e-14, atol=0)
    result = dict(time_M0=times, resolutions=[12,16,20], regions={},
                  weights='Compact-coordinate volume. Fixed shells have matched bounds; exterior masks follow each grid horizon.',
                  limitation='Native norms use different grid quadratures. N12 half-dt is a sensitivity measure, not a rigorous temporal error bound.',
                  sources=runs, halfdt_source=half)
    for region in runs[0]['metrics']:
        result['regions'][region] = {name:decay([r['metrics'][region][name] for r in runs],
            half['metrics'][region][name] if half else None) for name in KEYS}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs=3, type=Path, help='N12, N16 and N20 run/merged directories')
    parser.add_argument('--halfdt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.runs, args.halfdt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['regions'], indent=2))
