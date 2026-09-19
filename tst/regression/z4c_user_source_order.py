#!/usr/bin/env python3
"""Check the experimental post-boundary user source in actual RK stage arrays.

Requires the z4c_tov_ks problem generator and NumPy. The pulse touches the outer
boundary, so homogeneous closure changes it. This checks additive damping and
ordering, not the long-time stability of the experimental boundary datum.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def read_snapshot(run, operation, stage):
    path = run / f'z4c_snapshot_{operation}_rank0_cycle0_stage{stage}.json'
    meta = json.loads(path.read_text())
    assert meta['scalar_bytes'] == 8 and meta['byte_order'] == 'little', meta
    values = np.fromfile(path.with_suffix('.bin'), dtype='<f8').reshape(meta['shape'])
    return meta, values


def active(meta, values):
    i, j, k = meta['active_start']
    ni, nj, nk = meta['active_count']
    return values[:, :, k:k+nk, j:j+nj, i:i+ni]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    baseline = (repo / 'inputs/tests/z4c_characteristic_cpbc_plane_pulse.athinput').read_text()
    baseline = baseline.replace('nx1 = 128', 'nx1 = 32').replace('nx1 = 64', 'nx1 = 16')
    baseline = baseline.replace('nlim = -1', 'nlim = 1')
    baseline = baseline.replace('characteristic_test_center = 3.0',
                                'characteristic_test_center = 7.0')
    baseline = baseline.replace('characteristic_test_width = 0.5',
                                'characteristic_test_width = 2.0')
    baseline = baseline.replace('<z4c>', '<z4c>\ndebug_balance = true\n'
        'debug_snapshot_operations = pre_boundary_rhs,post_boundary_rhs,'
        'pre_excision_rhs,post_excision_rhs,post_excision_state')
    baseline = baseline.replace('<problem>', '<problem>\n'
        'excision_freeze_radius = 0.0\nexcision_ramp_radius = 0.0\n'
        'outer_sponge_geometry = radial\nouter_sponge_start_radius = 0.1\n'
        'outer_sponge_ramp_width = 0.1\nouter_sponge_damping_time = 5.0')
    cases = [('default', None, True), ('before', False, True), ('after', True, True),
             ('no_source_before', False, False), ('no_source_after', True, False)]
    stages = {}; results = {}
    for name, after, sponge in cases:
        run = args.output.resolve() / name
        run.mkdir(parents=True, exist_ok=False)
        content = baseline
        if after is not None:
            content = content.replace('<z4c>', '<z4c>\nuser_rhs_after_boundary = '
                                      + str(after).lower())
        if sponge:
            content = content.replace('outer_sponge_enabled = false',
                                      'outer_sponge_enabled = true')
        (run / 'input.athinput').write_text(content)
        with (run / 'run.log').open('w') as log:
            subprocess.run([str(args.exe.resolve()), '-i', 'input.athinput'], cwd=run,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '1'})
        stages[name] = []
        source_error = 0.; closure_change = 0.; boundary_source = 0.
        for stage in range(1, 4):
            meta, pre = read_snapshot(run, 'pre_excision_rhs', stage)
            _, post = read_snapshot(run, 'post_excision_rhs', stage)
            _, state = read_snapshot(run, 'post_excision_state', stage)
            _, pre_bc = read_snapshot(run, 'pre_boundary_rhs', stage)
            _, post_bc = read_snapshot(run, 'post_boundary_rhs', stage)
            for values in [pre, post, state, pre_bc, post_bc]:
                assert np.isfinite(active(meta, values)).all(), (name, stage)
            # All active cell centers are outside r=0.2 here: sigma=1/tau=0.2.
            expected = active(meta, pre) - (0.2 if sponge else 0.) * active(meta, state)
            error = np.max(abs(active(meta, post) - expected))
            scale = max(np.max(abs(expected)), 1e-30)
            assert error <= 8*np.finfo(float).eps*scale, (name, stage, error, scale)
            source_error = max(source_error, float(error))
            if after:
                assert np.array_equal(active(meta, pre), active(meta, post_bc))
                final = post
            else:
                assert np.array_equal(active(meta, pre_bc), active(meta, post))
                final = post_bc
            stages[name].append(active(meta, final).copy())
            i0, j0, k0 = meta['active_start']
            ni, nj, nk = meta['active_count']
            for m, block in enumerate(meta['blocks']):
                if block['xmax'][0] != 8.:
                    continue
                boundary = (m, slice(None), slice(k0,k0+nk), slice(j0,j0+nj), i0+ni-1)
                closure_change = max(closure_change,
                                     float(np.max(abs(post_bc[boundary]-pre_bc[boundary]))))
                boundary_source = max(boundary_source,
                                      float(np.max(abs(post[boundary]-pre[boundary]))))
        results[name] = dict(stages=3, additive_source_max_error=source_error,
                             boundary_closure_max_change=closure_change,
                             boundary_source_max_change=boundary_source)
        assert closure_change > 1e-10, (name, closure_change)
        if sponge:
            assert boundary_source > 1e-9, (name, boundary_source)
        print(name, 'PASS', flush=True)
    for stage in range(3):
        assert np.array_equal(stages['default'][stage], stages['before'][stage])
        assert np.array_equal(stages['no_source_before'][stage],
                              stages['no_source_after'][stage])
    # Same initial state, genuinely different first-stage boundary datum.
    difference = np.max(abs(stages['after'][0] - stages['before'][0]))
    assert difference > 1e-9, difference
    results['first_stage_ordering_difference'] = float(difference)
    (args.output / 'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
