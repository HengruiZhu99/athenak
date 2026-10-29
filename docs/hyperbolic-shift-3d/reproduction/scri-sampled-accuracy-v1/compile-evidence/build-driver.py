#!/usr/bin/env python3
"""Build/run a bounded CPU manufactured accuracy study; never evolves fields.

Each run executes one n in {8,12,16,24}, one native process/worker, timeout <=390s.
Coordinate local process slots with other work before invoking run. Existing
outputs are never replaced. Summary reports observed orders without acceptance
thresholds or stability claims. Only completed, hash-bound results are included.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_suffix('.cpp')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inputs(build_dir):
    paths = [SOURCE, Path(__file__), ROOT/'src/athena.hpp']
    paths += sorted((ROOT/'src/z4c/hyperboloidal').glob('*.hpp'))
    paths += sorted((ROOT/'src/mesh/cubed_sphere').glob('*.hpp'))
    paths += sorted(build_dir.glob('*.hpp'))
    paths += [build_dir/f'kokkos/{p}/src/libkokkos{p}.a'
              for p in ('containers', 'algorithms', 'core', 'simd')]
    return {str(p.resolve()): sha(p) for p in paths}


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    flags_path = args.build_dir/'tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_scri_angular.dir/flags.make'
    flags = {}
    for line in flags_path.read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            flags[k] = shlex.split(v)
    before = inputs(args.build_dir)
    modified_header = ROOT/'src/z4c/hyperboloidal/hyperbolic_composite.hpp'
    header_bytes = modified_header.read_bytes()
    binary = args.output/'scri_sampled_accuracy'
    command = ['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'],
               *flags['CXX_FLAGS'], str(SOURCE), '-o', str(binary)]
    command += [str(args.build_dir/f'kokkos/{p}/src/libkokkos{p}.a')
                for p in ('containers', 'algorithms', 'core', 'simd')]
    start = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    write(args.output/'build.json', dict(command=command, returncode=completed.returncode,
          stdout=completed.stdout, stderr=completed.stderr, seconds=time.monotonic()-start))
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    if before != inputs(args.build_dir):
        raise RuntimeError('source/build inputs changed during compile')
    write(args.output/'source.json', dict(
        scope='New manufactured harness with source-bound current native CPU headers',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        binary_sha256=sha(binary), flags_sha256=sha(flags_path), inputs=before,
        retained_native_header=dict(path='hyperbolic_composite.hpp', source=str(modified_header),
                                    sha256=hashlib.sha256(header_bytes).hexdigest())))
    (args.output/'hyperbolic_composite.hpp').write_bytes(header_bytes)
    (args.output/'build-source.cpp').write_bytes(SOURCE.read_bytes())
    (args.output/'build-driver.py').write_bytes(Path(__file__).read_bytes())
    print(binary)


def run(args):
    if args.n not in (8, 12, 16, 24) or not 1 <= args.timeout <= 390:
        raise ValueError('bounded n/timeout policy')
    binding_path = args.binary.parent/'source.json'
    binding = json.loads(binding_path.read_text())
    if sha(args.binary) != binding['binary_sha256']:
        raise ValueError('binary hash mismatch')
    # The binary is self-contained. Retained source snapshots bind its harness;
    # later native-header work does not retroactively change this executable.
    for filename, key in [('build-source.cpp', str(SOURCE)), ('build-driver.py', str(Path(__file__)) )]:
        if sha(args.binary.parent/filename) != binding['inputs'][key]:
            raise ValueError('retained harness hash mismatch')
    header = binding['retained_native_header']
    if sha(args.binary.parent/header['path']) != header['sha256'] or header['sha256'] != binding['inputs'][header['source']]:
        raise ValueError('retained modified native-header hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.output/'accuracy.json'
    command = [str(args.binary.resolve()), str(args.n), str(report.resolve())]
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    start = time.monotonic()
    result = dict(command=command, n=args.n, timeout_seconds=args.timeout,
                  binary_sha256=sha(args.binary), source_binding_sha256=sha(binding_path),
                  scope='One bounded CPU process; no evolution')
    try:
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=args.timeout)
        result.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
                      status='completed' if completed.returncode == 0 else 'failed')
        if completed.returncode == 0:
            data = json.loads(report.read_text())
            if data['n'] != args.n or data['status'] != 'completed' or len(data['fixtures']) != 4:
                raise ValueError('report identity/fixture mismatch')
            result['report_sha256'] = sha(report)
    except subprocess.TimeoutExpired as exc:
        result.update(status='timeout', error=str(exc))
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    result['seconds'] = time.monotonic()-start
    write(args.output/'execution.json', result)
    print(json.dumps(result, indent=2))
    if result['status'] != 'completed':
        raise SystemExit(1)


def summary(args):
    if args.output.exists():
        raise ValueError('summary output must be new')
    records = []
    for directory in args.run:
        execution = json.loads((directory/'execution.json').read_text())
        if execution['status'] != 'completed':
            raise ValueError('cannot silently include failed/incomplete run')
        path = directory/'accuracy.json'
        if sha(path) != execution['report_sha256']:
            raise ValueError('report hash mismatch')
        records.append((json.loads(path.read_text()), execution, path))
    records.sort(key=lambda item: item[0]['n'])
    if len({r[0]['n'] for r in records}) != len(records):
        raise ValueError('duplicate n')
    if len({(r[1]['binary_sha256'], r[1]['source_binding_sha256']) for r in records}) != 1:
        raise ValueError('resolution comparison needs one exact binary/source binding')
    fixtures = []
    for index in range(4):
        first = records[0][0]['fixtures'][index]
        methods = {}
        for method in first['errors']:
            rows = []
            for record, _, _ in records:
                fixture = record['fixtures'][index]
                if fixture['fixture'] != first['fixture']:
                    raise ValueError('fixture order mismatch')
                values = fixture['errors'][method]['regions']['all']
                row = dict(n=record['n'], h=record['h'], l2=values['l2'], linf_component=values['linf_component'])
                if rows:
                    previous = rows[-1]
                    for norm in ('l2', 'linf_component'):
                        row[f'observed_order_{norm}'] = math.log(previous[norm]/row[norm])/math.log(previous['h']/row['h']) if row[norm] > 0 and previous[norm] > 0 else None
                rows.append(row)
            methods[method] = rows
        fixtures.append(dict(fixture=first['fixture'], exact_boundary_regular=first['exact_boundary_regular'], methods=methods))
    result = dict(scope='Absolute manufactured derivative/rate errors and pairwise observed orders; no stability or evolution admission',
                  global_norm_primary=True, expected_order_imposed=False,
                  binary_sha256=records[0][1]['binary_sha256'], source_binding_sha256=records[0][1]['source_binding_sha256'],
                  reports=[dict(path=str(p), sha256=sha(p), n=r['n']) for r, _, p in records], fixtures=fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--build-dir', type=Path, default=ROOT/'build-hyperbolic-mpi')
    b.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--binary', type=Path, required=True)
    r.add_argument('--n', type=int, required=True)
    r.add_argument('--timeout', type=int, default=390)
    r.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('summary')
    s.add_argument('--run', type=Path, action='append', required=True)
    s.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for name in ('build_dir', 'output', 'binary'):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    globals()[args.command](args)


if __name__ == '__main__':
    main()
