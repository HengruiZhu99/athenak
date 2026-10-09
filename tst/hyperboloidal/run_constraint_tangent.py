"""Audit initial constraint generation for the native nonspherical gauge pulse.

Usage: python run_constraint_tangent.py /abs/audit-executable output_dir
The immutable executable emits JSONL for continuum and native initial probes.
The measurements diagnose spatial error; they do not establish long evolution
stability or replace a continuum characteristic analysis.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--quick', action='store_true',
                        help='Omit native N36/48/64/72 refinement probes')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = output / 'constraint-tangent-audit'
    if executable.exists():
        raise ValueError(
            'Use a fresh output directory to retain run provenance')
    shutil.copy2(args.executable.resolve(), executable)
    sources = [
        'tst/hyperboloidal/test_layer_constraint_tangent.cpp',
        'tst/hyperboloidal/run_constraint_tangent.py',
        'src/z4c/hyperboloidal/layer_reference.hpp',
        'src/z4c/hyperboloidal/layer_gauge.hpp',
        'src/z4c/hyperboloidal/cartesian_patch.hpp',
        'src/z4c/hyperboloidal/athenak_bridge.hpp',
        'src/z4c/hyperboloidal/conformal_rhs.hpp',
        'src/z4c/hyperboloidal/conformal_constraints.hpp',
        'src/z4c/hyperboloidal/spherical_ghosts.hpp',
        'src/z4c/hyperboloidal/interior_dissipation.hpp',
        'src/utils/finite_diff.hpp',
    ]
    receipt = {
        'executable': str(executable), 'sha256': sha256(executable),
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_dirty': bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
        'source_sha256': {path: sha256(ROOT / path) for path in sources},
        'pulse': {'lapse': .1, 'shift': .02, 'width': .5,
                  'angular': True, 'shape': '(1-r^2)^4 exp(-4r^2)'},
        'method': ('Native probes use the actual patch RHS, spherical mask, '
                   'mixed centered derivatives, Lx advection, interior KO, '
                   'algebraic projector and diagnostic reconstruction. '
                   'Continuum probes use analytic initial pulse jets and '
                   'fine fourth-order spatial differences of continuous RHS '
                   'values; signed temporal differences estimate Hdot. '
                   'Native norms estimate the limits of '
                   'H/M/Z(ref+dt*RHS)/abs(dt).'),
        'cases': [],
    }
    cases = []
    for a in [.5, 1]:
        for layer in [0, 1]:
            for r0, r1 in [(.35, .75), (.2, .8)]:
                for dx in [.0004, .0002]:
                    name = f'continuum-a{a}-layer{layer}-{r0}-{r1}-dx{dx}'
                    cases.append((name, ['--continuum', a, layer,
                                         r0, r1, dx, 1e-6]))
    native_cases = [
        ('gentle-N24-degree2', 24, .35, .75, 2, 1, 1),
        ('gentle-N24-degree3', 24, .35, .75, 3, 1, 1),
        ('gentle-N24-degree2-legacy-ghosts', 24, .35, .75, 2, 0, 1),
        ('broad-N24-degree2', 24, .2, .8, 2, 1, 1),
        ('broad-N24-degree3', 24, .2, .8, 3, 1, 1),
        ('broad-N24-degree2-legacy-ghosts', 24, .2, .8, 2, 0, 1),
        ('cmc-N24-degree2', 24, .35, .75, 2, 1, 0),
    ]
    if not args.quick:
        for n in [36, 48]:
            native_cases.append((f'gentle-N{n}-degree2', n, .35, .75, 2, 1, 1))
        for n in [36, 48, 64, 72]:
            native_cases.append((f'broad-N{n}-degree2', n, .2, .8, 2, 1, 1))
    for name, n, r0, r1, degree, symmetric, layer in native_cases:
        cases.append((name, ['--native', n, 2.1, .5, layer, degree,
                             symmetric, r0, r1, 1e-6]))
    for dt in [1e-5, 1e-7]:
        cases.append((f'broad-N24-degree2-dt{dt}',
                      ['--native', 24, 2.1, .5, 1, 2, 1, .2, .8, dt]))
    by_name = {}
    for name, options in cases:
        command = [str(executable)] + list(map(str, options))
        start = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=180)
        (output / (name + '.jsonl')).write_text(completed.stdout)
        (output / (name + '.stderr')).write_text(completed.stderr)
        rows = [json.loads(line) for line in completed.stdout.splitlines()]
        record = {'name': name, 'command': command,
                  'exit_status': completed.returncode,
                  'wall_seconds': time.monotonic() - start,
                  'measurements': rows}
        if completed.returncode:
            record['error'] = completed.stderr
        receipt['cases'].append(record)
        by_name[name] = record
        (output / 'results.json').write_text(
            json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'name': name, 'exit_status': completed.returncode,
                          'wall_seconds': record['wall_seconds']}), flush=True)
        if completed.returncode:
            raise RuntimeError(f'{name} failed: {completed.stderr}')

    def mean(name, field):
        rows = by_name[name]['measurements']
        return sum(row[field] for row in rows) / len(rows)

    receipt['summary'] = {
        'continuum_Hdot_max': max(abs(row['Hdot']) for case in receipt['cases']
                                  for row in case['measurements']
                                  if row['kind'] == 'continuum'),
        'gentle_cubic_quadratic_bulk_peak_difference': abs(
            mean('gentle-N24-degree3', 'Hdot_max')
            - mean('gentle-N24-degree2', 'Hdot_max')),
        'broad_cubic_quadratic_bulk_peak_difference': abs(
            mean('broad-N24-degree3', 'Hdot_max')
            - mean('broad-N24-degree2', 'Hdot_max')),
        'broad_temporal_probe_dt_consistency': {
            str(dt): mean(f'broad-N24-degree2-dt{dt}', 'Hdot_rms')
            for dt in [1e-5, 1e-7]},
        'resolution_orders': {},
    }
    if not args.quick:
        for family, resolutions in [('gentle', [24, 36, 48]),
                                    ('broad', [24, 36, 48, 64, 72])]:
            rows = []
            for coarse, fine in zip(resolutions[:-1], resolutions[1:]):
                ec = mean(f'{family}-N{coarse}-degree2', 'Hdot_rms')
                ef = mean(f'{family}-N{fine}-degree2', 'Hdot_rms')
                rows.append({'coarse': coarse, 'fine': fine,
                             'Hdot_RMS_order': math.log(ec / ef)
                             / math.log(fine / coarse)})
            receipt['summary']['resolution_orders'][family] = rows
    (output / 'results.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['summary'], indent=2))


if __name__ == '__main__':
    main()
