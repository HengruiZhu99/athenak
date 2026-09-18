#!/usr/bin/env python3
"""Gauge-only vacuum probes: exact initial geometric preservation and MPI parity.

This short test does not establish long-time stability. Requires NumPy and a
supplied MPI executable. Refined cases place the pulse across an SMR interface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from z4c_background_balance import read_csv, verify


def snapshot(run, operation, cycle, stage):
    records = {}
    for path in sorted(run.glob(f'z4c_snapshot_{operation}_rank*_cycle{cycle}_stage{stage}.json')):
        meta = json.loads(path.read_text())
        assert meta['scalar_bytes'] == 8 and meta['byte_order'] == 'little'
        data = np.fromfile(path.with_suffix('.bin'), dtype='<f8').reshape(meta['shape'])
        assert np.isfinite(data).all()
        ng = meta['ng']
        nx, ny, nz = meta['active_count']
        for block, values in zip(meta['blocks'], data):
            gid = block['gid']
            assert gid not in records
            records[gid] = (block, values[:, ng:ng+nz, ng:ng+ny, ng:ng+nx].copy())
    assert records, f'Missing {operation} snapshots'
    return records


def check_seed(run, component, amplitude, center):
    before = snapshot(run, 'pre_gauge_pulse', 0, 0)
    after = snapshot(run, 'post_gauge_pulse', 0, 0)
    assert before.keys() == after.keys()
    field = 18 + component
    maximum = 0.
    for gid, (block, values) in after.items():
        assert np.count_nonzero(before[gid][1]) == 0
        assert np.count_nonzero(np.delete(values, field, axis=0)) == 0
        nz, ny, nx = values.shape[1:]
        # The metadata's min/max fields are physical Cartesian block bounds.
        axes = [block['xmin'][a] + (np.arange(n)+.5)
                * (block['xmax'][a]-block['xmin'][a])/n
                for a, n in enumerate((nx, ny, nz))]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
        q2 = (x-center)**2 + y*y + z*z
        expected = np.zeros_like(q2)
        selected = (q2 < 1) & (x*x+y*y+z*z > .5**2)
        expected[selected] = amplitude*np.exp(1-1/(1-q2[selected]))
        assert np.max(abs(values[field]-expected)) <= abs(amplitude)*2e-14
        maximum = max(maximum, float(np.max(abs(values[field]))))
    assert maximum > 0
    final = snapshot(run, 'post_recast', 2, 3)
    hashes = {str(gid): hashlib.sha256(values.tobytes()).hexdigest()
              for gid, (_, values) in final.items()}
    geometric_response = max(float(np.max(abs(values[:17])))
                             for _, values in final.values())
    assert geometric_response > 0, 'Gauge perturbation produced no geometric response'
    return maximum, hashes, geometric_response


def check_regions(run):
    # Audit radii are explicit diagnostic settings, separate from the sponge.
    # Check membership so stale defaults cannot mislabel first-injection sites.
    frozen = 0
    for path in sorted(run.glob('z4c_balance_rank*.csv')):
        for row in read_csv(path):
            radius = sum(float(row[a])**2 for a in ('x', 'y', 'z'))**.5
            if row['region'] == 'freeze':
                assert radius <= .5, row
                if row['operation'] == 'post_recast':
                    assert float(row['max_abs']) == 0, row
                    frozen += 1
            elif row['region'] == 'sponge':
                assert .5 < radius < 1., row
            elif row['region'] == 'interior':
                assert 1. <= radius < 2., row
            elif row['region'] == 'exterior':
                assert radius >= 2., row
    assert frozen > 0


def verify_rejections(exe, launcher, output, baseline):
    enabled = baseline.replace('<problem>', '<problem>\nvacuum_gauge_pulse_amplitude = 1e-8')
    cases = {
        'component': enabled.replace('<problem>', '<problem>\nvacuum_gauge_pulse_component = 4'),
        'width': enabled.replace('<problem>', '<problem>\nvacuum_gauge_pulse_width = 0'),
        'matter': enabled.replace('zero_tmunu_feedback = true', 'zero_tmunu_feedback = false'),
        'background': enabled.replace('pure_background = true', 'pure_background = false'),
        'lapse_disabled': enabled.replace('evolve_lapse_residual = true', 'evolve_lapse_residual = false')
                                 .replace('boundary_rhs = characteristic_cpbc', 'boundary_rhs = sommerfeld'),
        'shift_disabled': enabled.replace('<problem>', '<problem>\nvacuum_gauge_pulse_component = 1')
                                 .replace('evolve_shift_residual = true', 'evolve_shift_residual = false')
                                 .replace('boundary_rhs = characteristic_cpbc', 'boundary_rhs = sommerfeld'),
        'mixed_pulses': enabled.replace('<problem>', '<problem>\n'
            'outer_sponge_test_theta_pulse_amplitude = 1e-8\n'
            'outer_sponge_test_theta_pulse_width = 1'),
    }
    results = {}
    for name, config in cases.items():
        run = output.resolve()/('reject_'+name)
        run.mkdir(parents=True, exist_ok=False)
        (run/'input.athinput').write_text(config)
        result = subprocess.run([launcher, '-n', '1', str(exe.resolve()), '-i', 'input.athinput'],
                                cwd=run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        (run/'run.log').write_text(result.stdout)
        assert result.returncode != 0 and 'Vacuum gauge pulse requires' in result.stdout, name
        results[name] = result.returncode
    (output/'rejections.json').write_text(json.dumps(results, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpiexec')
    parser.add_argument('--ranks', nargs='+', type=int, default=[1, 4])
    args = parser.parse_args()
    baseline = (Path(__file__).resolve().parents[1]/'inputs/z4c_ks_background.athinput').read_text()
    baseline = baseline.replace('excision_freeze_radius = 1.0', 'excision_freeze_radius = 0.5')
    baseline = baseline.replace('excision_ramp_radius = 1.4', 'excision_ramp_radius = 1.0')
    baseline = baseline.replace('<problem>', '<problem>\nzero_tmunu = true')
    baseline = baseline.replace('<z4c>', '<z4c>\n'
        'debug_balance_freeze = 0.5\ndebug_balance_ramp = 1.0\n'
        'debug_balance_horizon = 2.0\n'
        'debug_snapshot_operations = pre_gauge_pulse,post_gauge_pulse,post_recast')
    verify_rejections(args.exe, args.launcher, args.output, baseline)
    results = {}
    for ranks in args.ranks:
        for refined in [False, True]:
            text = baseline
            center = 4. if refined else 1.75
            if refined:
                head, tail = text.split('<meshblock>', 1)
                for a in (1, 2, 3):
                    head = re.sub(rf'^nx{a}\s*=.*', f'nx{a} = 32', head, flags=re.M)
                    head = re.sub(rf'^x{a}min\s*=.*', f'x{a}min = -8', head, flags=re.M)
                    head = re.sub(rf'^x{a}max\s*=.*', f'x{a}max = 8', head, flags=re.M)
                text = (head+'<meshblock>'+tail).replace('refinement = none', 'refinement = static')
                text = text.replace('max_nmb_per_rank = 8', 'max_nmb_per_rank = 128')
                text += '\n<refined_region0>\nlevel = 1\nx1min = -1\nx1max = 1\nx2min = -1\nx2max = 1\nx3min = -1\nx3max = 1\n'
            for component in range(4):
                for multiple in [1, 2]:
                    amplitude = multiple*1e-8
                    name = f'{"refined" if refined else "uniform"}_c{component}_a{multiple}_r{ranks}'
                    run = args.output.resolve()/name
                    run.mkdir(parents=True, exist_ok=False)
                    config = text.replace('<problem>', '<problem>\n'
                        f'vacuum_gauge_pulse_amplitude = {amplitude}\n'
                        f'vacuum_gauge_pulse_component = {component}\n'
                        f'vacuum_gauge_pulse_x1 = {center}')
                    (run/'input.athinput').write_text(config)
                    with (run/'run.log').open('w') as log:
                        subprocess.run([args.launcher, '-n', str(ranks), str(args.exe.resolve()),
                                        '-i', 'input.athinput'], cwd=run, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    check_regions(run)
                    result = verify(run, ranks, False, refined, expected_buffer=1.0,
                                    expect_theta_response=False)
                    (result['seed_max'], result['final_active_block_sha256'],
                     result['geometric_response']) = check_seed(run, component, amplitude, center)
                    if refined:
                        levels = {block['level'] for block, _ in snapshot(run, 'post_gauge_pulse', 0, 0).values()}
                        pulse_levels = {block['level'] for block, values in snapshot(run, 'post_gauge_pulse', 0, 0).values() if np.any(values[18+component])}
                        assert len(levels) == len(pulse_levels) == 2, 'Pulse missed refinement interface'
                    results[name] = result
                    print(name, result['theta_response'], flush=True)
                    (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    for refined in ['uniform', 'refined']:
        for component in range(4):
            for ranks in args.ranks:
                a = results[f'{refined}_c{component}_a1_r{ranks}']
                b = results[f'{refined}_c{component}_a2_r{ranks}']
                # Require the physical/gauge response, not spurious constraint
                # generation. Theta is measured but need not be nonzero.
                ratio = b['geometric_response']/a['geometric_response']
                assert abs(ratio-2) < 2e-4, (refined, component, ranks, ratio)
            for multiple in [1, 2]:
                reference = results[f'{refined}_c{component}_a{multiple}_r{args.ranks[0]}']
                for ranks in args.ranks[1:]:
                    actual = results[f'{refined}_c{component}_a{multiple}_r{ranks}']
                    assert ({k: v for k, v in actual.items() if k != 'ranks'} ==
                            {k: v for k, v in reference.items() if k != 'ranks'}), \
                        'Response or active state changed with MPI partition'
    print('All gauge-only seed, response linearity, SMR-interface, and MPI bitwise checks passed.')


if __name__ == '__main__':
    main()
