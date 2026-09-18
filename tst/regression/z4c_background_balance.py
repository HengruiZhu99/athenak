#!/usr/bin/env python3
"""Sixth-order Kerr-Schild residual regression, using a supplied MPI executable.

Example:
  python3 tst/regression/z4c_background_balance.py --exe build/src/athena \
      --output /tmp/ks-balance --ranks 1 4

Exact vacuum stage preservation and physical response are separate gates.
This short regression does not establish long-time perturbation stability.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(run, ranks, equilibrium, refined):
    state_files = sorted(run.glob('z4c_balance_rank*.csv'))
    geometry_files = sorted(run.glob('z4c_geometry_rank*.csv'))
    algebraic_files = sorted(run.glob('z4c_algebraic_rank*.csv'))
    assert len(state_files) == len(geometry_files) == len(algebraic_files) == ranks
    required = {'init_state', 'init_reconstructed', 'init_projected', 'init_recast',
                'rhs_full_vs_bg', 'volume_rhs', 'post_ko_rhs', 'pre_excision_rhs',
                'post_excision_rhs', 'pre_boundary_rhs', 'post_boundary_rhs',
                'post_rk', 'post_exchange', 'pre_physical_bc', 'post_physical_bc',
                'pre_projection', 'post_projection', 'post_recast'}
    if refined:
        required |= {'post_restrict', 'post_prolong'}
    response = 0.0
    max_det = max_trace = 0.0
    for file in state_files:
        rows = read_csv(file)
        assert required <= {row['operation'] for row in rows}
        assert {'1', '2', '3'} <= {row['stage'] for row in rows}
        if refined:
            assert any(row['region'] == 'refinement_block' for row in rows)
        for row in rows:
            value = float(row['max_abs'])
            assert math.isfinite(value) and int(row['nonfinite']) == 0
            if equilibrium:
                assert value == 0 and int(row['nonzero']) == 0
                assert int(row['bit_mismatch']) == 0
            if (row['field'] == 'z4c_Theta' and row['operation'] == 'post_recast'
                    and row['region'] != 'ghost'):
                response = max(response, value)
    for file in geometry_files:
        for row in read_csv(file):
            assert math.isfinite(float(row['max_abs'])) and int(row['nonfinite']) == 0
            if equilibrium:
                assert float(row['max_abs']) == 0 and int(row['bit_mismatch']) == 0
    for file in algebraic_files:
        for row in read_csv(file):
            if row['operation'] not in {'init_projected', 'post_projection'}:
                continue
            det, trace = float(row['det_error']), float(row['trace_A'])
            assert int(row['nonfinite']) == 0
            assert math.isfinite(det) and math.isfinite(trace)
            # Projection is an algebraic constraint to floating-point precision;
            # exact residual preservation does not mean an exactly unit determinant.
            assert det < 2e-14 and trace < 2e-14
            max_det, max_trace = max(max_det, det), max(max_trace, trace)
    if not equilibrium:
        assert response > 0, 'A physical Theta perturbation was erased'
    log = (run / 'run.log').read_text(errors='replace')
    assert 'Terminating on cycle limit' in log and '### FATAL ERROR' not in log
    return {'exact_equilibrium': equilibrium, 'ranks': ranks, 'refined': refined,
            'theta_response': response, 'max_det_error': max_det, 'max_trace_A': max_trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--launcher', default='mpiexec')
    args = parser.parse_args()
    exe = args.exe.resolve()
    source = Path(__file__).resolve().parents[1] / 'inputs/z4c_ks_background.athinput'
    baseline = source.read_text()
    results = {}
    for ranks in args.ranks:
        for case in ['equilibrium', 'refined_equilibrium', 'pulse', 'pulse_double',
                     'pulse_dipole', 'pulse_dipole_double',
                     'refined_dipole', 'refined_dipole_double']:
            run = args.output.resolve() / f'{case}_r{ranks}'
            run.mkdir(parents=True, exist_ok=False)
            text = baseline
            refined = case.startswith('refined_')
            perturbed = 'pulse' in case or 'dipole' in case
            if refined:
                # Same dx=0.5 coarse spacing, with genuine refinement interfaces.
                for axis in range(1, 4):
                    head, rest = text.split('<meshblock>', 1)
                    head = re.sub(rf'^nx{axis}\s*=.*', f'nx{axis} = 32', head, flags=re.M)
                    head = re.sub(rf'^x{axis}min\s*=.*', f'x{axis}min = -8', head, flags=re.M)
                    head = re.sub(rf'^x{axis}max\s*=.*', f'x{axis}max = 8', head, flags=re.M)
                    text = head + '<meshblock>' + rest
                text = text.replace('refinement = none', 'refinement = static')
                text = text.replace('max_nmb_per_rank = 8', 'max_nmb_per_rank = 128')
                text += ('\n<refined_region0>\nlevel = 1\n'
                         'x1min = -1\nx1max = 1\nx2min = -1\nx2max = 1\nx3min = -1\nx3max = 1\n')
            if perturbed:
                amp = 2e-8 if case.endswith('double') else 1e-8
                text = text.replace('<problem>', '<problem>\n'
                    f'outer_sponge_test_theta_pulse_amplitude = {amp}\n'
                    f'outer_sponge_test_theta_pulse_radius = {4 if refined else 3}\n'
                    'outer_sponge_test_theta_pulse_width = 0.3')
            if 'dipole' in case:
                text = text.replace('<problem>', '<problem>\n'
                                    'outer_sponge_test_theta_pulse_dipole_axis = 1')
                text = text.replace('debug_balance_profiles = false',
                                    'debug_balance_profiles = true')
            (run / 'input.athinput').write_text(text)
            with (run / 'run.log').open('w') as log:
                subprocess.run([args.launcher, '-n', str(ranks), str(exe), '-i', 'input.athinput'],
                               cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
            results[run.name] = verify(run, ranks, not perturbed, refined)
            if 'dipole' in case:
                profile = {tuple(float(row[key]) for key in ('x', 'y', 'z')):
                           float(row['theta'])
                           for file in run.glob('z4c_theta_rank*_cycle0_stage1.csv')
                           for row in read_csv(file)}
                assert profile, 'Missing dipole spatial profile'
                magnitude = max(abs(value) for value in profile.values())
                parity_error = max(abs(value + profile[(-x, y, z)])
                                   for (x, y, z), value in profile.items()) / magnitude
                assert parity_error < 1e-5, 'The seeded Theta pulse is not odd in x'
                results[run.name]['dipole_parity_error'] = parity_error
        refined_ratio = (results[f'refined_dipole_double_r{ranks}']['theta_response'] /
                         results[f'refined_dipole_r{ranks}']['theta_response'])
        assert abs(refined_ratio - 2) < 1e-4
        dipole_ratio = (results[f'pulse_dipole_double_r{ranks}']['theta_response'] /
                        results[f'pulse_dipole_r{ranks}']['theta_response'])
        assert abs(dipole_ratio - 2) < 1e-4
        small = results[f'pulse_r{ranks}']['theta_response']
        large = results[f'pulse_double_r{ranks}']['theta_response']
        assert abs(large / small - 2) < 1e-4, 'Small-signal response is not linear'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
