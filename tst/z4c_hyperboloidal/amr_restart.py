#!/usr/bin/env python3
"""Ordinary executable AMR restart: continuous4 on2 vs2 on2 +restart2 on8.

No test main, runtime admission override, field replacement or alternate driver
is used. Run only after adaptive evolution has passed its admission gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

import numpy as np
from restart_roundtrip import HEADER, compare, parse, payload, run

DT = 0.00046944203735617445
N, OVERLAP, BULK_SLOTS = 12, 3, 18**3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(info):
    with Path(info['path']).open('rb') as file:
        text = file.read(info['start']).decode()
    values, block = {}, None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line[1:line.index('>')]
        elif '=' in line:
            key, value = line.split('=', 1)
            values[block, key.strip()] = value.strip()
    return values


def expected_topology(root, refined):
    leaves = {(i, j, k, root, 0) for i in range(3) for j in range(3) for k in range(3)}
    leaves.remove((1, 1, 1, root, 0))
    leaves.update((2+i, 2+j, 2+k, root+1, 0)
                  for i in range(2) for j in range(2) for k in range(2))
    leaves.update((layer, 0, 0, root, chart) for chart in range(1, 7) for layer in range(12))
    if refined:
        leaves.remove((11, 0, 0, root, 1))
        leaves.update((22+i, j, k, root+1, 1)
                      for i in range(2) for j in range(2) for k in range(2))
    return leaves


def audit(info, ranks):
    """Check complete native checkpoint shape, topology, slots, policy and history."""
    with Path(info['path']).open('rb') as file:
        file.seek(info['start'])
        fixed = file.read(HEADER)
        root, = struct.unpack_from('=i', fixed, 4)
    assert root == 4, ('N12 root level changed', root)
    locations = list(struct.iter_unpack('=5i', info['topology']))
    refined = info['cycle'] >= 2  # regular outputs precede that cycle's AMR call
    assert len(locations) == len(set(locations))
    assert set(locations) == expected_topology(root, refined), 'unexpected native AMR topology'
    assert info['blocks'] == (113 if refined else 106)
    assert info['record_blocks'] == info['blocks']
    assert np.array_equal(np.frombuffer(info['costs'], dtype='=f4'),
                          np.ones(info['blocks'], dtype='=f4')), 'native costs are not all one'
    time = 0.0
    for _ in range(info['cycle']):
        time += DT
    assert info['time'] == time and info['dt'] == DT, 'native exact timestep/time changed'
    meta = info['metadata']
    assert meta[2] == len(info['identity'])
    assert meta[3] == (BULK_SLOTS+N*N)*39
    assert meta[4:] == (ranks, 0, info['blocks']//ranks, 0), 'wrong shared restart rank metadata'
    values = parameters(info)
    for key, expected in {
            ('time', 'evolution'): 'dynamic', ('time', 'integrator'): 'rk4_classical',
            ('mesh_refinement', 'refinement'): 'adaptive',
            ('amr_criterion0', 'method'): 'location',
            ('output1', 'file_type'): 'rst', ('output1', 'single_file_per_rank'): 'false'}.items():
        assert values[key] == expected, (key, values[key])
    for key, expected in {
            ('time', 'dt'): DT, ('time', 'cfl_number'): .2,
            ('mesh_refinement', 'num_levels'): 2, ('mesh_refinement', 'refinement_interval'): 0,
            ('mesh_refinement', 'max_nmb_per_rank'): 72,
            ('mesh_refinement', 'ncycle_check'): 1,
            ('amr_criterion0', 'location_x1'): .99, ('amr_criterion0', 'location_x2'): 0,
            ('amr_criterion0', 'location_x3'): 0, ('amr_criterion0', 'location_rad'): .005,
            ('z4c_hyperboloidal', 'ko'): .1, ('z4c_hyperboloidal', 'chi_floor'): 1e-10,
            ('output1', 'dcycle'): 1, ('mesh', 'logical_location_version'): 2}.items():
        assert float(values[key]) == expected, (key, values[key])
    assert info['atlas'].startswith(b'athenak-chart-atlas-v2;')
    assert info['identity'].startswith(b'native-hyperboloidal-v1;fields=31;chi-record=8;')

    data = payload(info)
    assert data.shape == (info['blocks'], BULK_SLOTS+N*N, 39)
    assert np.isfinite(data).all(), 'nonfinite field/history/padding in checkpoint'
    active = np.zeros(data.shape[:2], dtype=bool)
    faces = 0
    for m, (i, j, k, level, chart) in enumerate(locations):
        factor = 1 << (level-root)
        lower, upper = [OVERLAP]*3, [OVERLAP+N]*3
        if chart == 0:
            for axis, index in enumerate((i, j, k)):
                lower[axis] -= OVERLAP*(index == 0)
                upper[axis] += OVERLAP*(index == 3*factor-1)
        elif i//factor == 0:
            # This fixture has spherical_inner_radius=.08: no inner radial overlap.
            upper[0] += OVERLAP*(i == factor-1)
            for axis, index in ((1, j), (2, k)):
                lower[axis] -= OVERLAP*(index == 0)
                upper[axis] += OVERLAP*(index == factor-1)
        active[m, :BULK_SLOTS].reshape((18, 18, 18))[
            lower[2]:upper[2], lower[1]:upper[1], lower[0]:upper[0]] = True
        if chart and i == 12*factor-1:
            active[m, BULK_SLOTS:] = True
            faces += 1
    assert faces == (9 if refined else 6)
    bulk_count = int(active[:, :BULK_SLOTS].sum())
    surface_count = int(active[:, BULK_SLOTS:].sum())
    assert bulk_count == (241488 if refined else 229392)
    assert surface_count == faces*N*N
    assert np.array_equal(data[:, :, 31] != 0, active), 'missing/extra evolved checkpoint slot'
    assert (data[~active] == 0).all(), 'nonzero unused restart padding'
    owned = data[active]
    assert (owned[:, 0] > 0).all(), 'nonpositive checkpoint chi'
    assert (owned[:, 31:33] > 0).all(), 'nonpositive chi-history minimum'
    assert (owned[:, 33:38] == 0).all(), 'unexpected floor/nonfinite/correction history'
    assert (owned[:, 38] == 1e300).all(), 'unexpected first floor event'
    assert (owned[:, 28:31] == 0).all(), 'dormant field policy changed'
    return dict(path=info['path'], cycle=info['cycle'], time=info['time'], dt=info['dt'],
                blocks=info['blocks'], surface_patches=faces, bulk_points=bulk_count,
                scri_points=surface_count, all_costs_one=True,
                scalar_values=int(data.size), owned_scalar_values=int(owned.size),
                metadata=list(meta), topology_sha256=hashlib.sha256(info['topology']).hexdigest(),
                fixed_mesh_header_sha256=hashlib.sha256(fixed[4:-20]).hexdigest(),
                atlas_sha256=hashlib.sha256(info['atlas']).hexdigest(),
                identity_sha256=hashlib.sha256(info['identity']).hexdigest(),
                payload_sha256=hashlib.sha256(data.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    names = [name for name in subprocess.check_output(['rg', '--files', 'src'], cwd=root,
             text=True).splitlines() if name.endswith(('.cpp', '.hpp', 'CMakeLists.txt'))]
    names += ['CMakeLists.txt', 'tst/z4c_hyperboloidal/amr_restart.py',
              'tst/z4c_hyperboloidal/restart_roundtrip.py',
              'tst/z4c_hyperboloidal/amr_restart_contract.md',
              'tst/z4c_hyperboloidal/n12_initialize.athinput']
    sources = {name: sha256(root/name) for name in sorted(names)}
    link_path = binary.parent/'CMakeFiles/athena.dir/link.txt'
    objects = {}
    for token in shlex.split(link_path.read_text()):
        if not token.endswith('.o'):
            continue
        obj = binary.parent/token
        objects[str(obj)] = sha256(obj)
        relative = token.split('CMakeFiles/athena.dir/', 1)[-1][:-2]
        cpp = root/'src'/relative
        if cpp.exists():
            assert obj.stat().st_mtime_ns >= cpp.stat().st_mtime_ns, f'stale object: {obj}'
    binary_hash = sha256(binary)
    provenance = dict(binary=str(binary), binary_sha256=binary_hash,
                      link_sha256=sha256(link_path), sources=sources, objects=objects)
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    for name in ('amr_restart.py', 'restart_roundtrip.py', 'amr_restart_contract.md'):
        shutil.copyfile(source/name, output/name)
    input_path = output/'amr_restart.athinput'
    input_path.write_text((source/'n12_initialize.athinput').read_text()+'''
<job>
basename = amr_restart
<mesh_refinement>
refinement = adaptive
num_levels = 2
max_nmb_per_rank = 72
refinement_interval = 0
ncycle_check = 1
<amr_criterion0>
method = location
location_x1 = .99
location_x2 = 0
location_x3 = 0
location_rad = .005
<time>
evolution = dynamic
integrator = rk4_classical
tlim = 1
nlim = 4
ndiag = 1
<output1>
file_type = rst
dcycle = 1
single_file_per_rank = false
''')
    commands, audits, checkpoints = {}, {}, {}
    immutable_metadata = None
    result = dict(passed=False, scope='ordinary native executable AMR/restart, local CPU MPI')
    try:
        def execute(name, ranks, options, expected_cycles):
            nonlocal immutable_metadata
            commands[name] = ['mpiexec', '-n', str(ranks), str(binary), *options]
            last = run(binary, output, name, ranks, options)
            paths = sorted((output/name/'rst').glob('*.rst'))
            by_cycle = {}
            for path in paths:
                info = parse(path)
                checked = audit(info, ranks)
                invariant = tuple(checked[key] for key in
                    ('fixed_mesh_header_sha256', 'atlas_sha256', 'identity_sha256'))
                if immutable_metadata is None:
                    immutable_metadata = invariant
                assert invariant == immutable_metadata, 'immutable checkpoint metadata changed'
                audits.setdefault(name, []).append(checked)
                if info['cycle'] in by_cycle:
                    compare(by_cycle[info['cycle']], info)
                by_cycle[info['cycle']] = info
            assert set(by_cycle) == expected_cycles, (name, sorted(by_cycle))
            checkpoints[name] = by_cycle
            return last

        continuous = execute('continuous4', 2, ['-i', str(input_path)], set(range(5)))
        first = execute('first2', 2, ['-i', str(input_path), 'time/nlim=2'], {0, 1, 2})
        assert first['cycle'] == 2 and first['blocks'] == 113
        prefix = compare(checkpoints['continuous4'][2], first)
        resumed = execute('restart_plus2', 8, ['-r', first['path'], 'time/nlim=4'], {3, 4})
        continuations = {str(cycle): compare(checkpoints['continuous4'][cycle],
                                            checkpoints['restart_plus2'][cycle])
                         for cycle in (3, 4)}
        final = compare(continuous, resumed)
        for info in (first, resumed):
            assert info['atlas'] == continuous['atlas'] and info['identity'] == continuous['identity']
            assert info['costs'] == continuous['costs'] and info['topology'] == continuous['topology']
        result.update(passed=True, test_admission_override=False, ranks=[2, 8],
                      prefix=prefix, resumed_cycles=continuations, final=final)
    finally:
        stable = dict(binary_unchanged=sha256(binary) == binary_hash,
                      sources_unchanged=all(sha256(root/name) == digest
                                            for name, digest in sources.items()),
                      objects_unchanged=all(sha256(Path(name)) == digest
                                            for name, digest in objects.items()))
        result.update(commands=commands, checkpoints=audits, provenance=stable,
                      executable_sha256=binary_hash)
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert all(stable.values()), stable
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
