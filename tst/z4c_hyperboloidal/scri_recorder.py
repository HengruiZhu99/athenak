#!/usr/bin/env python3
"""Build and validate the native scri recorder against actual AthenaK objects.

Use --build BUILD --output NEW_DIRECTORY. No production files are touched.
Requires NumPy, an MPI build, and mpiexec. All evolutions are only two N12 steps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vis/python'))
import scri_reader as reader
from restart_roundtrip import parse, compare


def validate_archive(path, truth=True):
    s, e = reader.load(path)
    sampled = reader.reconstruct_sampled_jets(s, e)
    np.testing.assert_allclose(sampled, s['sampled_residual_jets'], atol=2e-12, rtol=2e-12)
    lifts = reader.normal_lifts(s, e)
    np.testing.assert_allclose(lifts, s['normal_lift_jets'], atol=2e-11, rtol=2e-11)
    restored = reader.restored_endpoint_jets(s, e)
    np.testing.assert_allclose(restored[:, :28, 0], s['surface_state'][:, :28], atol=2e-12, rtol=2e-12)
    # Geometry round-trip: x and inverse-map derivatives; actual cached references.
    count = 0
    if truth:
        table = np.fromfile(path.parent/f'truth.{s["record"][0]}.r{s["record"][3]}.bin', dtype='f8').reshape(-1, 477)
        index = {tuple(row[:4].astype(int)): q for q, row in enumerate(e['bulk_ids'])}
        for row in table:
            q = index[tuple(row[:4].astype(int))]
            np.testing.assert_array_equal(s['bulk_state'][q], row[4:35])
            np.testing.assert_array_equal(e['bulk_geometry'][q, 3:], row[35:74])
            np.testing.assert_array_equal(e['bulk_reference'][q].ravel(), row[74:])
        count = len(table)
    return dict(points=len(e['surface_ids']), truth_donors=count,
                jet_error=float(np.max(abs(sampled-s['sampled_residual_jets']), initial=0)),
                lift_error=float(np.max(abs(lifts-s['normal_lift_jets']), initial=0)))


def record(directory, number=0, block=1):
    return reader.load_record(sorted(directory.glob(f'*.o{block}.{number:08d}.r????????.scri')))


def surface(parts):
    return {tuple(ids[[0, 2, 3]].astype(int)): values for s, e in parts
            for ids, values in zip(e['surface_ids'], s['surface_state'])}


def analytic_modes(parts):
    """Check reduction wiring against independent Cartesian formulas for l<=1."""
    expected = np.zeros((31, 4), dtype=complex)
    for state, epoch in parts:
        x = epoch['surface_geometry'][:, :3]
        radius = np.linalg.norm(x, axis=1)
        y = np.empty((len(x), 4), dtype=complex)
        y[:, 0] = 1/np.sqrt(4*np.pi)
        y[:, 1] = np.sqrt(3/(8*np.pi))*(x[:, 0]-1j*x[:, 1])/radius
        y[:, 2] = np.sqrt(3/(4*np.pi))*x[:, 2]/radius
        y[:, 3] = -np.sqrt(3/(8*np.pi))*(x[:, 0]+1j*x[:, 1])/radius
        expected += np.einsum('q,qf,qm->fm', epoch['surface_weights'], state['surface_state'], y.conj())
    saved = parts[0][0]['native_modes'][:, :4]
    actual = saved[..., 0]+1j*saved[..., 1]
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    return float(np.max(abs(actual-expected), initial=0))


def agree(a, b):
    aa, bb = surface(a), surface(b)
    assert aa.keys() == bb.keys()
    for key in aa:
        np.testing.assert_array_equal(aa[key], bb[key])
    np.testing.assert_allclose(a[0][0]['native_modes'], b[0][0]['native_modes'], atol=3e-14, rtol=3e-14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = build/'src/athena'
    report = {'runs': {}, 'reconstruction': {}, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    def save():
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    base = (ROOT/'tst/z4c_hyperboloidal/n12_initialize.athinput').read_text()
    scri = '\n<output1>\nfile_type = scri\ndcycle = 1\n'
    settings = ('\n<time>\nevolution = dynamic\nintegrator = rk4_classical\nnlim = 2\ntlim = 1\nstart_time = 0\n'
                '<output9>\nfile_type = rst\ndt = 100\n')
    def run(name, ranks, text, executable=binary, options=None, failure=None):
        directory = output/name;directory.mkdir();(directory/'input').write_text(text)
        command = ['mpiexec', '-n', str(ranks), str(executable), *(options or ['-i', str(directory/'input')])]
        start = time.perf_counter()
        with (directory/'run.log').open('w') as log:
            result = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        log = (directory/'run.log').read_text()
        report['runs'][name] = dict(seconds=time.perf_counter()-start, command=command, returncode=result.returncode)
        if failure:
            assert result.returncode and failure in log, (name, log[-3000:])
        else:
            assert result.returncode == 0, (name, log[-3000:])
        save();print(name, 'PASS', flush=True)
        return directory

    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M)[1])
    obj, exe = output/'scri_recorder.o', output/'scri_recorder'
    subprocess.run(['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                    '-c', str(ROOT/'tst/z4c_hyperboloidal/scri_recorder.cpp'), '-o', str(obj)], check=True)
    link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
    link[link.index('-o')+1] = str(exe)
    subprocess.run(link, cwd=build/'src', check=True)
    for ranks in (2, 8):
        directory = output/f'manufactured{ranks}'
        run(directory.name, ranks, base+scri, executable=exe, options=[str(directory/'input')])
        for phase in range(3):
            parts = record(directory, phase)
            assert parts[0][0]['record'][2] == phase, 'refine/coarsen did not rebuild topology'
            analytic_modes(parts)
        same_time = record(directory, 3)[0][0]
        assert same_time['record'][0] == 2 and same_time['record'][2] == 3
        for path in sorted(directory.glob('*.r????????.scri')):
            if int(path.name.split('.')[-3]) > 3:
                continue
            report['reconstruction'][str(path.relative_to(output))] = validate_archive(path)
        save()
    for phase in range(3):
        agree(record(output/'manufactured2', phase), record(output/'manufactured8', phase))
    disabled = run('disabled', 2, base+settings)
    enabled = run('enabled', 2, base+settings+scri)
    report['evolution_equivalence'] = compare(parse(sorted((disabled/'rst').glob('*.rst'))[-1]),
                                             parse(sorted((enabled/'rst').glob('*.rst'))[-1]))
    serial = run('serial', 1, base+scri)
    empty = run('empty_ranks', 8, base+scri)
    agree(record(serial), record(empty));agree(record(serial), record(enabled))
    assert any(len(e['surface_ids']) == 0 for _, e in record(empty))
    rst = sorted((enabled/'rst').glob('*.rst'))[0]
    restarted = run('restart8', 8, '', options=['-r', str(rst), 'time/nlim=2'])
    for number in (1, 2):
        agree(record(enabled, number), record(restarted, number))
    assert not list(restarted.glob('*.o1.00000000.r*.scri'))
    report['restart_equivalence'] = compare(parse(sorted((enabled/'rst').glob('*.rst'))[-1]),
                                           parse(sorted((restarted/'rst').glob('*.rst'))[-1]))
    multiple = run('multiple', 2, base+settings+scri+
        '\n<output2>\nfile_type = scri\ndcycle = 2\nnodal = false\nlmax = 3\n'
        '<output3>\nfile_type = scri\ndt = .0007\nmodes = false\nlmax = -1\n')
    assert len(list(multiple.glob('*.o1.*.r00000000.scri'))) == 3
    assert len(list(multiple.glob('*.o2.*.r00000000.scri'))) == 2
    assert len(list(multiple.glob('*.o3.*.r00000000.scri'))) == 2
    assert record(multiple, 0, 2)[0][0]['native_modes'].shape == (31, 16, 2)
    assert 'surface_state' not in record(multiple, 0, 2)[0][0]
    assert 'native_modes' not in record(multiple, 0, 3)[0][0]
    for name, setting, failure in [('negative', 'lmax = -1', '[0,128]'),
                                 ('resolution', 'lmax = 12', 'coarsest angular'),
                                 ('empty', 'nodal = false\nmodes = false', 'requires nodal or modes')]:
        run('invalid_'+name, 2, base+scri+setting+'\n', failure=failure)
    # Truncation and mixed distributed output must fail closed.
    path = next(enabled.glob('*.o1.00000000.r00000000.scri'))
    truncated = output/'truncated.scri';truncated.write_bytes(path.read_bytes()[:-1])
    try:
        reader.read(truncated)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted truncated archive')
    try:
        reader.load_record([path, path])
    except ValueError:
        pass
    else:
        raise AssertionError('accepted duplicate rank')
    report['bytes'] = dict(
        static=sum(p.stat().st_size for p in enabled.glob('*.epoch.scri')),
        per_record=sum(p.stat().st_size for p in enabled.glob('*.o1.00000001.r????????.scri')))
    report['bytes']['per_code_time_at_dt_001'] = 100*report['bytes']['per_record']
    save();print('PASS all scri recorder checks', flush=True)


if __name__ == '__main__':
    main()
