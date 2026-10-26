#!/usr/bin/env python3
"""Regular-Z4c regression for the fixed exterior and adaptive Cartesian bulk.

Uses only the Python standard library. All inputs, logs, metrics and source hashes
are retained in --output. Analytic outer ghosts are a verification boundary;
Sommerfeld is tested separately with the localized angular disturbance.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'inputs/z4c/cubed_sphere.athinput'


def make_input(changes, amr=False):
    section = ''
    lines = []
    for line in BASE.read_text().splitlines():
        if line.startswith('<'):
            section = line.strip('<>')
        if '=' in line:
            key = line.split('=')[0].strip()
            if (section, key) in changes:
                line = key + ' = ' + str(changes[section, key])
        lines.append(line)
    if amr:
        lines += ['<amr_criterion>', 'method = user']
        at = lines.index('refinement = adaptive')
        lines[at+1:at+1] = ['num_levels = 2', 'max_nmb_per_rank = 160',
                          'refinement_interval = 1']
    return '\n'.join(lines) + '\n'


def run(binary, output, name, changes, amr=False):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    inp = directory / 'input.athinput'
    inp.write_text(make_input(changes, amr))
    command = [str(binary), '-i', str(inp), '-d', str(directory)]
    with (directory / 'run.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (directory / 'cubed_sphere_errors.csv').open() as table:
        rows = {r['region']: {k: float(v) for k, v in r.items() if k != 'region'}
                for r in csv.DictReader(table)}
    assert len(rows) == 2
    for r in rows.values():
        assert all(math.isfinite(v) for v in r.values()), (name, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['smoke', 'convergence', 'scalar', 'failure'], default='smoke')
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep a multi-case suite on one executable even if another build relinks
    # its original path during the run.
    origin = binary
    binary = output/'athena_test_executable'
    shutil.copy2(origin, binary)
    result = {'suite': args.suite, 'binary': str(binary), 'cases': {},
              'original_binary': str(origin),
              'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    sources = [*ROOT.glob('src/**/*.cpp'), *ROOT.glob('src/**/*.hpp'),
               ROOT/'CMakeLists.txt', ROOT/'src/CMakeLists.txt', BASE, Path(__file__).resolve()]
    result['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(set(sources))}
    cases = result['cases']
    if args.suite == 'smoke':
        for amr in [False, True]:
            name = 'amr_minkowski' if amr else 'minkowski'
            cases[name] = run(binary, output, name, {
                ('problem', 'initial_data'): 'minkowski', ('time', 'nlim'): 7 if amr else 3,
                ('mesh_refinement', 'refinement'): 'adaptive' if amr else 'none'}, amr)
            assert max(r['linf'] for r in cases[name].values()) < 1e-11
            if amr:
                assert cases[name]['bulk']['plan_builds'] >= 3
                log = (output / name / 'run.log').read_text()
                assert '56 MeshBlocks created, 56 deleted by AMR' in log
        cases['amr_gauge_wave'] = run(binary, output, 'amr_gauge_wave', {
            ('time', 'nlim'): 7, ('mesh_refinement', 'refinement'): 'adaptive'}, True)
        assert max(r['linf'] for r in cases['amr_gauge_wave'].values()) < 1e-6
        for data in ['minkowski', 'gauge_wave']:
            name = 'amr_corners_'+data
            cases[name] = run(binary, output, name, {
                ('time', 'nlim'): 7, ('mesh_refinement', 'refinement'): 'adaptive',
                ('problem', 'initial_data'): data, ('problem', 'amr_refine_corners'): 'true'}, True)
            assert max(r['linf'] for r in cases[name].values()) < (1e-11 if data == 'minkowski' else 1e-6)
            assert cases[name]['bulk']['plan_builds'] >= 3
            assert '56 MeshBlocks created, 56 deleted by AMR' in (output/name/'run.log').read_text()
        cases['angular_pulse'] = run(binary, output, 'angular_pulse', {
            ('time', 'nlim'): 5, ('problem', 'initial_data'): 'angular_pulse',
            ('problem', 'amplitude'): 0.2, ('cubed_sphere', 'outer_boundary'): 'sommerfeld'})
        assert max(r['H_l2'] for r in cases['angular_pulse'].values()) < 0.1
        cases['puncture_gauge_pulse'] = run(binary, output, 'puncture_gauge_pulse', {
            ('time', 'nlim'): 5, ('problem', 'initial_data'): 'angular_pulse',
            ('problem', 'amplitude'): 0.2, ('cubed_sphere', 'outer_boundary'): 'sommerfeld',
            ('z4c', 'lapse_harmonic'): 0, ('z4c', 'lapse_oplog'): 2,
            ('z4c', 'shift_Gamma'): 0.75, ('z4c', 'shift_eta'): 2})
        assert max(r['H_l2'] for r in cases['puncture_gauge_pulse'].values()) < 0.1
    elif args.suite == 'convergence':
        for n in [16, 24, 32]:
            changes = {('mesh', 'nx'+str(a)): 2*n for a in [1, 2, 3]}
            changes.update({('cubed_sphere', 'nr'): n, ('cubed_sphere', 'nangular'): n})
            cases[str(n)] = run(binary, output, 'gauge'+str(n), changes)
        result['orders'] = {}
        for region in ['bulk', 'shell']:
            for quantity in ['l2', 'H_l2', 'M_l2', 'weyl_l2']:
                orders = [math.log(cases[str(a)][region][quantity] /
                                   cases[str(b)][region][quantity]) / math.log(b/a)
                          for a, b in [(16, 24), (24, 32)]]
                result['orders'][region + '_' + quantity] = orders
                assert min(orders) > 3.2, (region, quantity, orders)
    elif args.suite == 'scalar':
        result['scalar'] = {}
        for n, t in [(12, 4), (16, .1), (24, .1), (32, .1)]:
            name = 'scalar'+str(n)
            changes = {('mesh', 'nx'+str(a)): 16 if n == 12 else 2*n for a in [1, 2, 3]}
            changes.update({('cubed_sphere', 'nr'): n, ('cubed_sphere', 'nangular'): n,
                            ('problem', 'initial_data'): 'minkowski', ('time', 'nlim'): 0,
                            ('cubed_sphere', 'outer_boundary'): 'sommerfeld',
                            ('cubed_sphere', 'scalar_probe_tlim'): t})
            cases[name] = run(binary, output, name, changes)
            assert max(r['linf'] for r in cases[name].values()) < 1e-11  # restored spacetime
            with (output/name/'cubed_sphere_scalar.csv').open() as table:
                rows = [{k: v if k == 'region' else float(v) for k, v in row.items()}
                        for row in csv.DictReader(table)]
            assert all(math.isfinite(v) for r in rows for k, v in r.items() if k != 'region')
            result['scalar'][str(n)] = rows
            if n == 12:
                assert min(r['minimum'] for r in rows) > -.15
                assert max(r['maximum'] for r in rows) < 1.05
                assert max(r['linf_error'] for r in rows) < .15
                assert max(r['linf_error'] for r in rows[-2:]) < .002
        result['orders'] = {}
        for region in ['bulk', 'shell']:
            errors = {n: [r for r in result['scalar'][str(n)] if r['region'] == region][-1]['linf_error']
                      for n in [16, 24, 32]}
            orders = [math.log(errors[a]/errors[b])/math.log(b/a)
                      for a, b in [(16, 24), (24, 32)]]
            result['orders'][region] = orders
            assert min(orders) > 3.2, (region, orders)
    else:
        # An intentionally underresolved, excessive-CFL disturbance exercises
        # the failure recorder. This run is expected to fail admissibility.
        directory = output/'first_bad_stage'
        directory.mkdir(parents=True, exist_ok=True)
        inp = directory/'input.athinput'
        changes = {('mesh', 'nx'+str(a)): 16 for a in [1, 2, 3]}
        changes.update({('cubed_sphere', 'nr'): 12, ('cubed_sphere', 'nangular'): 12,
                        ('cubed_sphere', 'outer_boundary'): 'sommerfeld',
                        ('problem', 'initial_data'): 'angular_pulse',
                        ('problem', 'amplitude'): .99, ('problem', 'pulse_width'): .1,
                        ('time', 'cfl_number'): 1, ('time', 'tlim'): 2})
        inp.write_text(make_input(changes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        with (directory/'run.log').open('w') as log:
            proc = subprocess.run([str(binary), '-i', str(inp), '-d', str(directory)],
                                  stdout=log, stderr=subprocess.STDOUT)
        assert proc.returncode != 0, 'expected inadmissible RK stage'
        with (directory/'cubed_sphere_first_bad_stage.csv').open() as table:
            metadata = table.readline()
            rows = list(csv.DictReader(table))
        match = re.search(r'dt=([^,]+).*stage=(\d+).*first_kji=(\d+),(\d+),(\d+)', metadata)
        assert match, metadata
        dt, stage = float(match[1]), int(match[2])
        location = tuple(int(match[i]) for i in [3, 4, 5])
        center = [r for r in rows if tuple(int(r[k]) for k in ['k', 'j', 'i']) == location]
        assert {int(r['variable']) for r in center} == set(range(25))
        assert len(rows) == 9**3*25
        with (directory/'cubed_sphere_first_bad_stage_geometry.csv').open() as table:
            geometry = list(csv.DictReader(table))
        assert len(geometry) == 9**3
        assert all(math.isfinite(float(v)) for r in geometry for v in r.values())
        assert (directory/'cubed_sphere_parameters.txt').is_file()
        g0 = [0, .25, 2/3][stage-1]
        beta = [1, .25, 2/3][stage-1]
        for r in center:
            assert all(math.isfinite(float(r[k])) for k in ['input', 'rk_register', 'rhs'])
            current = float(r['current'])
            expected = g0*float(r['input'])+(1-g0)*float(r['rk_register'])+beta*dt*float(r['rhs'])
            assert abs(current-expected) < 1e-11*max(1, abs(expected))
        result['failure'] = {'metadata': metadata.strip(), 'stencil_rows': len(rows),
                             'returncode': proc.returncode}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
