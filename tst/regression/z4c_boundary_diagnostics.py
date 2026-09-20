#!/usr/bin/env python3
"""Verify opt-in boundary diagnostics are observational, including MPI ghosts.

Run with a double precision MPI MHD+Z4c executable and a small uniform vacuum
input. The same 16^3 domain is split into eight 8^3 blocks. Compare residual
checkpoint bytes for diagnostics off/on and for one/four ranks, and verify
zero equilibrium plus validity-label coverage. This is not a stability test.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from z4c_background_restart import checkpoint, cohort


def set_parameter(text, section, key, value):
    pattern = rf'(<{re.escape(section)}>\n)(.*?)(?=\n<|\Z)'
    def replace(match):
        body = match[2]
        line = rf'(?m)^\s*{re.escape(key)}\s*=.*$'
        if re.search(line, body):
            body = re.sub(line, f'{key} = {value}', body)
        else:
            body += f'\n{key} = {value}\n'
        return match[1] + body
    result, count = re.subn(pattern, replace, text, flags=re.S)
    assert count == 1, f'Missing/duplicate input section {section}'
    return result


def inspect(run, ranks, zero, debug):
    final, records = cohort(run, ranks, 3)
    values = [v for record in records for block in record['state'] for v in block]
    payload = b''.join(block.tobytes() for record in records for block in record['state'])
    assert records[0]['total'] == 8
    if zero:
        assert all(v == 0 for v in values), 'Zero residual was not preserved'
    result = {'time': records[0]['time'], 'cycle': records[0]['cycle'],
              'blocks': 8, 'checkpoint_name': final.name,
              'residual_sha256': hashlib.sha256(payload).hexdigest(),
              'residual_nonzero': sum(v != 0 for v in values)}
    files = list(run.glob('z4c_validity_rank*.csv'))
    if not debug:
        assert not files, 'Diagnostics emitted files while disabled'
        return result
    assert len(files) == ranks
    rows = [row for path in files for row in csv.DictReader(path.open())]
    assert all(int(row['invalid_metric_cells']) == 0 for row in rows
               if row['readiness'] != 'pending_ghosts')
    # rhs_full_vs_bg is a full input state, despite its operation name.
    assert not any(row['operation'] in ('volume_rhs', 'post_ko_rhs',
                                       'post_boundary_rhs') for row in rows)
    regions = {row['region'] for row in rows}
    assert {'internal_ghost', 'physical_ghost_face', 'physical_ghost_edge',
            'physical_ghost_corner', 'active_face_band', 'active_edge_band',
            'active_corner_band'} <= regions
    operations = {row['operation'] for row in rows}
    for axis in (1, 2, 3):
        for when in ('pre', 'post'):
            assert f'{when}_physical_bc_x{axis}' in operations
    assert {'active', 'post_fill', 'pending_ghosts'} <= {r['readiness'] for r in rows}
    assert all(int(row['relative_level']) == 0 for row in rows)
    assert all(0 <= int(row['rank']) < ranks and 0 <= int(row['gid']) < 8
               for row in rows)
    stages = {int(row['stage']) for row in rows
              if row['operation'] == 'post_physical_bc_x3'}
    assert {1, 2, 3} <= stages
    if zero:
        balance = [row for path in run.glob('z4c_balance_rank*.csv')
                   for row in csv.DictReader(path.open())]
        assert all(float(row['max_abs']) == 0 and int(row['nonfinite']) == 0
                   for row in balance if row['region'] != 'ghost' and
                   not row['region'].startswith(('physical_ghost', 'internal_ghost')))
    result.update(validity_rows=len(rows), regions=sorted(regions),
                  axis_rk_stages=sorted(stages), ready_states_valid=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpiexec')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 4])
    args = parser.parse_args()
    exe = args.exe.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    text = args.input.read_text()
    for section, key, value in [('time', 'nlim', 3),
                                ('mesh_refinement', 'refinement', 'none'),
                                ('z4c', 'debug_reduction_stride', 1),
                                ('z4c', 'debug_balance_profiles', 'false'),
                                ('z4c', 'debug_snapshot_operations', '')]:
        text = set_parameter(text, section, key, value)
    for axis in (1, 2, 3):
        text = set_parameter(text, 'mesh', f'nx{axis}', 16)
        text = set_parameter(text, 'meshblock', f'nx{axis}', 8)
    results = {'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
               'cases': {}}
    for kind, amplitude in [('zero', '0'), ('pulse', '1e-8')]:
        hashes = set()
        for ranks in args.ranks:
            for debug in (False, True):
                name = f'{kind}_rank{ranks}_debug{int(debug)}'
                run = args.output / name
                run.mkdir()
                config = set_parameter(text, 'problem', 'vacuum_gauge_pulse_amplitude', amplitude)
                config = set_parameter(config, 'z4c', 'debug_balance', str(debug).lower())
                (run / 'input.athinput').write_text(config)
                with (run / 'run.log').open('w') as log:
                    subprocess.run([args.launcher, '-n', str(ranks), str(exe),
                                    '-i', 'input.athinput'], cwd=run, stdout=log,
                                   stderr=subprocess.STDOUT, check=True,
                                   env=dict(os.environ, OMP_NUM_THREADS='1'))
                result = inspect(run, ranks, kind == 'zero', debug)
                results['cases'][name] = result
                hashes.add(result['residual_sha256'])
                (args.output / 'results.json').write_text(json.dumps(results, indent=2)+'\n')
                print(name, 'passed', flush=True)
        assert len(hashes) == 1, f'{kind}: diagnostic/MPI partition changed saved residual'
    results['passed'] = True
    (args.output / 'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
