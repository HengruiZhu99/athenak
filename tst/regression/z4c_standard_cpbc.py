#!/usr/bin/env python3
"""Check finite-state standard-gauge boundary rows independently (requires NumPy).

The finite lapse/shift pulse reaches the boundary initially. Verify all scalar
incoming zero-rate equations using the full-state principal symbol. Also verify
that the adapted-only lapse multiplier cannot affect standard subtraction, and
that MPI repartitioning preserves the actual boundary updates bitwise.
"""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpiexec')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('characteristics', repo /
        'analysis/z4c_characteristic/check_residual_characteristics_numeric.py')
    algebra = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(algebra)
    baseline = (repo/'inputs/tests/z4c_characteristic_cpbc_plane_pulse.athinput').read_text()
    baseline = baseline.replace('nx1 = 64', 'nx1 = 32').replace('nlim = -1', 'nlim = 3')
    baseline = baseline.replace('residual_gauge = background_adapted',
                                'residual_gauge = standard_subtract')
    baseline = baseline.replace('characteristic_test_amplitude = 1.0e-6',
                                'characteristic_test_amplitude = 0.01')
    baseline = baseline.replace('characteristic_test_center = 3.0',
                                'characteristic_test_center = 7.0')
    baseline = baseline.replace('<z4c>', '<z4c>\ndebug_balance = true\n'
        'debug_reduction_stride = 1\ndebug_snapshot_operations = '
        'rhs_full_vs_bg,pre_boundary_rhs,post_boundary_rhs')
    results = {}; reference = None
    for ranks in [1, 4]:
        for multiplier in [1.0, 0.37]:
            run = args.output.resolve()/f'r{ranks}_f{multiplier}'
            run.mkdir(parents=True, exist_ok=False)
            (run/'input.athinput').write_text(baseline.replace(
                'residual_lapse_f = 1.0', f'residual_lapse_f = {multiplier}'))
            with (run/'run.log').open('w') as log:
                subprocess.run([args.launcher, '-n', str(ranks),
                                str(args.exe.resolve()), '-i', 'input.athinput'],
                               cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
            samples = {}; max_error = 0.; sensitivity = 0.; changed = 0.
            for meta_path in run.glob('z4c_snapshot_rhs_full_vs_bg_*.json'):
                meta = json.loads(meta_path.read_text())
                full = np.fromfile(meta_path.with_suffix('.bin'), dtype='<f8').reshape(meta['shape'])
                rhs_path = Path(str(meta_path.with_suffix('.bin')).replace(
                    'rhs_full_vs_bg', 'post_boundary_rhs'))
                rhs = np.fromfile(rhs_path, dtype='<f8').reshape(meta['shape'])
                pre = np.fromfile(Path(str(rhs_path).replace('post_boundary_rhs',
                                  'pre_boundary_rhs')), dtype='<f8').reshape(meta['shape'])
                assert np.isfinite(full).all() and np.isfinite(rhs).all()
                for m, block in enumerate(meta['blocks']):
                    if block['xmax'][0] != 8.0:
                        continue
                    k = meta['ng'] + meta['active_count'][2]//2
                    j = meta['ng'] + meta['active_count'][1]//2
                    i = meta['ng'] + meta['active_count'][0] - 1
                    v = full[m, :, k, j, i]; q = rhs[m, :, k, j, i]
                    # This plane pulse has diagonal conformal metric.
                    assert max(abs(v[[2, 3, 5]])) < 1e-14
                    g = v[[1, 4, 6]]; inv = 1/g
                    normal_u = np.sqrt(inv[0]); normal_d = 1/normal_u
                    dq = normal_u*(3*q-4*rhs[m,:,k,j,i-1]+rhs[m,:,k,j,i-2])/(2*block['dx'][0])
                    p = np.array([q[7], q[17], q[8]*inv[0]-np.dot(q[[8,11,13]],inv)/3,
                                  normal_d*q[14]])
                    d = np.array([dq[0], dq[1]*inv[0]-np.dot(dq[[1,4,6]],inv)/3,
                                  dq[18], normal_d*dq[19]])
                    beta = normal_d*v[19]
                    left, _ = algebra.finite_scalar_left(v[18], v[0], 2*v[18], 1., beta, beta, 1.)
                    error = np.max(abs(left @ np.concatenate((p,d)))/np.linalg.norm(left,axis=1))
                    max_error = max(max_error, float(error))
                    # A background-only lapse coefficient must fail this check.
                    sensitivity = max(sensitivity, abs(-np.sqrt(2/v[0])*p[0]+d[2]))
                    changed = max(changed, float(np.max(abs(q-pre[m,:,k,j,i]))))
                    samples[(meta['cycle'],meta['stage'])] = q.copy()
            assert len(samples)==9 and max_error<1e-12, (len(samples),max_error)
            assert sensitivity>1e-8 and changed>1e-6, (sensitivity,changed)
            if reference is None:
                reference = samples
            else:
                for key in reference:
                    assert reference[key].tobytes() == samples[key].tobytes(), key
            results[run.name] = {'stages':len(samples), 'scalar_rate_error':max_error,
                'wrong_background_lapse_rate':sensitivity, 'boundary_change':changed}
            print(run.name, 'PASS', flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
