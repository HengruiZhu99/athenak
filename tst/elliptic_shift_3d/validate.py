#!/usr/bin/env python3
"""Bounded reproducible CPU gates; retains scientific failures and all raw rows."""
import argparse
import datetime
import hashlib
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from radial_reference import evolve, interpolate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    files = [ROOT/'CMakeLists.txt']
    for directory in ['src/mesh/cubed_sphere', 'src/multigrid',
                      'src/z4c/hyperboloidal', 'tst/elliptic_shift_3d', 'tst/elliptic_shift_1d']:
        files += [p for p in (ROOT/directory).rglob('*') if p.suffix in ('.hpp', '.cpp', '.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}


def run(command, directory, name, timeout=180):
    stdout = directory/(name+'.stdout')
    stderr = directory/(name+'.stderr')
    start = time.monotonic()
    with stdout.open('w') as out, stderr.open('w') as err:
        try:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=out,
                                    stderr=err, timeout=timeout, check=False)
            returncode, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timed_out = None, True
    parsed = None
    try:
        parsed = json.loads(stdout.read_text())
    except json.JSONDecodeError:
        pass
    return dict(command=list(map(str, command)), returncode=returncode, timeout=timed_out,
                wall_seconds=time.monotonic()-start, stdout=stdout.name, stderr=stderr.name,
                stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), result=parsed)


def compare_profiles(a, b):
    a, b = [np.genfromtxt(path, delimiter=',', names=True) for path in (a, b)]
    for key in ('x', 'y', 'z'):
        np.testing.assert_array_equal(a[key], b[key])
    columns = [key for key in a.dtype.names if key.startswith('state_') or key.startswith('delta_beta')]
    return {key: float(np.max(np.abs(a[key]-b[key]))) for key in columns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=Path('build-shift-3d'))
    parser.add_argument('--mpi-build', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve()
    frozen = sources()
    tag = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    binaries = {name: dict(path=str(build/name), sha256=sha(build/name)) for name in
                ['elliptic_shift_3d_operator_tests', 'elliptic_shift_3d_composite_tests',
                 'elliptic_shift_3d_lifecycle_tests', 'elliptic_shift_3d_coupled_pilot']}
    manifest = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    base_commit='8bef4966e253e92d85761f46d8885e7cc80dc2ed',
                    checkout_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    machine=platform.machine(), python=sys.version, source_hashes=frozen,
                    source_tag=tag, executables=binaries, worker_limit=1, tests={})

    def save():
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    save()
    cases = [('kernel', [build/'elliptic_shift_3d_operator_tests']),
             ('lifecycle', [build/'elliptic_shift_3d_lifecycle_tests', directory/'lifecycle.bin'])]
    for name, command in cases:
        manifest['tests'][name] = run(command, directory, name)
        print(name, manifest['tests'][name]['returncode'], flush=True)
        save()
    for curved in (0, 1):
        for n in (8, 12, 16):
            name = f'mms-curve{curved}-n{n}'
            manifest['tests'][name] = run([build/'elliptic_shift_3d_composite_tests', n, 1, 1, curved, 0], directory, name)
            print(name, manifest['tests'][name]['returncode'], flush=True)
            save()
    for name, case in [('mms-two-shell', [8, 2, 1, 1, 0]),
                       ('mms-amr-interface', [8, 2, 2, 1, 1]),
                       ('mms-tight-tolerance', [12, 1, 1, 1, 0, 1e-11]),
                       ('mms-second-order-control', [12, 1, 1, 1, 0, 1e-9, 2])]:
        manifest['tests'][name] = run([build/'elliptic_shift_3d_composite_tests', *case], directory, name)
        print(name, manifest['tests'][name]['returncode'], flush=True)
        save()

    coupled = [('minkowski-n6', [6, 0, .0005, 4, 0, 1, 1e-11]),
               ('minkowski-n8', [8, 0, .0005, 4, 0, 1, 1e-11]),
               ('minkowski-two-shell-n12', [12, 0, .0005, 4, 0, 2, 1e-11])]
    coupled += [(f'radial-n{n}', [n, 1e-6, .0005, 4, 0, 1, 1e-11]) for n in (6, 8, 12)]
    coupled += [('radial-half-dt', [8, 1e-6, .00025, 8, 0, 1, 1e-11]),
                ('radial-quarter-dt', [8, 1e-6, .000125, 16, 0, 1, 1e-11]),
                ('radial-tight-tolerance', [8, 1e-6, .0005, 4, 0, 1, 1e-13]),
                ('radial-second-order-control', [8, 1e-6, .0005, 4, 0, 1, 1e-11])]
    coupled += [(f'angular-two-shell-n{n}', [n, 1e-6, .0005, 4, 1, 2, 1e-11]) for n in (6, 8, 12)]
    for name, case in coupled:
        profile = directory/(name+'.csv')
        command = [build/'elliptic_shift_3d_coupled_pilot', *case, profile]
        if name in ('minkowski-n6', 'radial-n6'):
            command += [directory/(name+'-restart.bin'), tag]
        elif name == 'radial-second-order-control':
            command += ['', '', 2]
        receipt = run(command, directory, name)
        if profile.exists():
            receipt.update(profile=profile.name, profile_sha256=sha(profile))
        if Path(str(profile)+'.spheres.csv').exists():
            spheres = Path(str(profile)+'.spheres.csv')
            receipt.update(spheres=spheres.name, spheres_sha256=sha(spheres))
        # Exit 1 is an original scientific gate failure, never relabeled success.
        manifest['tests'][name] = receipt
        print(name, receipt['returncode'], flush=True)
        save()
    if args.mpi_build:
        executable = args.mpi_build.resolve()/'elliptic_shift_3d_mpi_tests'
        manifest['executables']['mpi'] = dict(path=str(executable), sha256=sha(executable))
        for ranks in (1, 2):
            name = f'mpi-ranks{ranks}'
            manifest['tests'][name] = run(['mpiexec', '-n', ranks, executable], directory, name)
            print(name, manifest['tests'][name]['returncode'], flush=True)
            save()
    # Independent high-resolution spherical evolution and interpolation control.
    ref = evolve()
    ref_coarse = evolve(n=257)
    ref_time = evolve(dt=.0000625, steps=32)
    np.savez_compressed(directory/'radial-reference.npz', **ref)
    manifest['radial_reference'] = dict(n=513, dt=.000125, time=.002,
        max_coarse_fine_alpha=float(np.max(np.abs(ref['alpha'][::2]-ref_coarse['alpha']))),
        max_half_time_alpha=float(np.max(np.abs(ref['alpha']-ref_time['alpha']))),
        sha256=sha(directory/'radial-reference.npz'), origin='existing independent g1_rhs.py and 1D quadrature shift',
        scope='regular-center symmetric-restoring lapse pilot; not the selected wormhole source')
    comparisons = {}
    for n in (6, 8, 12):
        table = np.genfromtxt(directory/f'radial-n{n}.csv', delimiter=',', names=True)
        comparisons[str(n)] = {}
        for field in ['alpha', 'chi', 'g_rr', 'A_rr', 'Khat', 'Theta', 'Lambda_r', 'beta_r']:
            difference = table[field]-interpolate(ref['r'], ref[field], table['r'])
            weight = table['weight']
            comparisons[str(n)][field] = dict(rms=float(np.sqrt(np.sum(weight*difference**2)/np.sum(weight))),
                                             maximum=float(np.max(np.abs(difference))))
    manifest['radial_comparisons'] = comparisons
    controls = {}
    for name in ['radial-half-dt', 'radial-quarter-dt', 'radial-tight-tolerance', 'radial-second-order-control']:
        controls[name] = compare_profiles(directory/'radial-n8.csv', directory/(name+'.csv'))
    manifest['numerical_controls'] = controls
    angular_comparisons = {}
    for n in (6, 8, 12):
        table = np.genfromtxt(directory/f'radial-n{n}.csv.spheres.csv', delimiter=',', names=True)
        angular_comparisons[str(n)] = []
        for radius in np.unique(table['r']):
            selected = table[table['r'] == radius]
            row = dict(r=float(radius), samples=len(selected))
            for field in ['alpha', 'chi', 'g_rr', 'A_rr', 'Khat', 'Theta', 'Lambda_r', 'beta_r']:
                mean = float(np.mean(selected[field]))
                row[field] = dict(mean=mean, angular_scatter=float(np.std(selected[field])),
                                  reference=float(interpolate(ref['r'], ref[field], np.array([radius]))[0]))
            angular_comparisons[str(n)].append(row)
    manifest['matched_sphere_comparisons'] = angular_comparisons
    # Retain every row losslessly, with hashes of both the original and gzip file.
    for receipt in manifest['tests'].values():
        for key in ('profile', 'spheres'):
            if key not in receipt:
                continue
            path = directory/receipt[key]
            compressed = path.with_name(path.name+'.gz')
            compressed.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            receipt[key+'_uncompressed_sha256'] = receipt[key+'_sha256']
            receipt[key] = compressed.name
            receipt[key+'_sha256'] = sha(compressed)
            path.unlink()
    manifest['source_hashes_unchanged'] = sources() == frozen
    manifest['binary_hashes_unchanged'] = all(sha(item['path']) == item['sha256'] for item in manifest['executables'].values())
    manifest['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest['scope_status'] = 'partial research backend; coupled scri consistency failures retained; production/puncture/scalar milestones incomplete'
    save()
    if not manifest['source_hashes_unchanged'] or not manifest['binary_hashes_unchanged']:
        raise RuntimeError('source/executable changed during validation')
    print('wrote', directory/'manifest.json')


if __name__ == '__main__':
    main()
