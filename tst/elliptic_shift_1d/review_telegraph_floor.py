#!/usr/bin/env python3
"""Separate initial lapse-floor changes from recorded total-RK clipping.

Reads a frozen case and a captured collector manifest; never evolves a state.
The native absolute correction counter belongs to store_alpha, whereas the
initial floor_lapse call records radius, logarithmic change, and node count.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_telegraph_time import numerical_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON file; a companion NPZ retains the arrays')
    args = parser.parse_args()
    paths = [args.case/'initial-data.npz', args.case/'profile-0001.npz',
             args.export/'manifest-long.json',
             args.case/'frozen-source/telegraph_integrator.cpp']
    manifest = json.loads(paths[2].read_bytes())
    numerical_binding(args.case, manifest)
    assert manifest['arguments']['lapse_update'] == 'total'
    with np.load(paths[0], allow_pickle=False) as before, np.load(paths[1], allow_pickle=False) as after:
        r = before['r']; np.testing.assert_array_equal(r, after['r'])
        assert float(after['t']) == 0 and manifest['arguments']['n'] == len(r)
        other = [0, 1, 2, 3, 4, 5, 7]
        np.testing.assert_array_equal(before['q'][:, other], after['q'][:, other])
        np.testing.assert_array_equal(before['q'][[0, -1], 6], after['q'][[0, -1], 6])
        mask = (r > 0) & (r < 1); rr = r[mask]
        omega = (1-rr*rr)/2; reference = (1+rr*rr)/2*rr*rr/(rr*rr+omega**4)
        initial = reference*np.exp(before['q'][mask, 6])
        floored = reference*np.exp(after['q'][mask, 6])
        np.testing.assert_allclose(floored, after['diagnostic'][mask, 7], rtol=2e-15, atol=0)
        np.testing.assert_allclose(floored, np.maximum(initial, manifest['arguments']['lapse_floor']*omega), rtol=4e-14, atol=0)
        log_change = after['q'][mask, 6]-before['q'][mask, 6]
        changed = log_change != 0
    report = dict(
        scope='Initial analytic-to-evolved lapse floor, distinct from later total-lapse RK clipping. The AF endpoint and scri are excluded and verified unchanged; all other initial fields are unchanged.',
        resolution=len(r), physical_lapse_floor=manifest['arguments']['lapse_floor'],
        changed_nodes=int(changed.sum()),
        max_initial_conformal_lapse_correction=float(np.max(floored-initial)),
        max_initial_physical_lapse_correction=float(np.max((floored-initial)/omega)),
        max_initial_floor_radius=float(rr[changed].max()) if changed.any() else 0.,
        max_initial_log_correction=float(np.max(log_change)),
        min_initial_physical_lapse_before=float(np.min(initial/omega)),
        min_initial_physical_lapse_after=float(np.min(floored/omega)),
        later_total_update_counter=dict(
            through=manifest['time'],
            max_conformal_correction=manifest['all_RK_stages']['max_lapse_floor_conformal_correction'],
            max_nonpositive_clamps_per_call=manifest['all_RK_stages']['max_nonpositive_lapse_clamps_per_call'],
            scope='store_alpha counter: total-RK stages and accepted updates; excludes the initial logarithmic floor_lapse call'),
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    with args.output.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, r=rr, omega=omega, initial_alpha=initial,
                            floored_alpha=floored, log_change=log_change)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
