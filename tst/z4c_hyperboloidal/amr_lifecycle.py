#!/usr/bin/env python3
"""Link actual native objects, run field AMR at 2/8 ranks, compare full snapshots."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import struct
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--ranks', type=int, nargs='+', default=[2, 8])
parser.add_argument('--build-only', action='store_true')
parser.add_argument('--executable', type=pathlib.Path, help='reuse a manifest-verified test executable')
parser.add_argument('--diagnostics', action='store_true',
                    help='also gate accepted-state diagnostic cache across actual regrids')
args = parser.parse_args()
if args.build_only and args.executable:
    parser.error('--build-only and --executable are mutually exclusive')
root = pathlib.Path(__file__).resolve().parents[2]
build, output = args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


sources = [root/name for name in subprocess.check_output(
    ['rg', '--files', 'src'], cwd=root, text=True).splitlines()
    if name.endswith(('.cpp', '.hpp', 'CMakeLists.txt'))]
sources += [root/'CMakeLists.txt', pathlib.Path(__file__).resolve(),
            root/'tst/z4c_hyperboloidal/amr_lifecycle.cpp',
            root/'tst/z4c_hyperboloidal/amr_lifecycle_contract.md',
            root/'tst/z4c_hyperboloidal/n12_initialize.athinput']
source_manifest = {str(p.relative_to(root)): sha256(p) for p in sorted(sources)}
(output/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
obj = output/'amr_lifecycle.o'
exe = args.executable.resolve() if args.executable else output/'amr_lifecycle'
compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                   '-c', str(root/'tst/z4c_hyperboloidal/amr_lifecycle.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if p == 'CMakeFiles/athena.dir/main.cpp.o' else p for p in link]
assert str(obj) in link
link[link.index('-o')+1] = str(exe)
objects = [build/'src'/p for p in link if p.endswith('.o') and p != str(obj)]
object_manifest = {str(p.relative_to(build)): sha256(p) for p in sorted(objects)}
(output/'object-manifest.json').write_text(json.dumps(object_manifest, indent=2)+'\n')
build_provenance = {
    'source_manifest_sha256': sha256(output/'source-manifest.json'),
    'object_manifest_sha256': sha256(output/'object-manifest.json')}
if args.executable:
    saved = json.loads((exe.parent/'build-provenance.json').read_text())
    assert all(saved[key] == value for key, value in build_provenance.items())
    assert saved['executable_sha256'] == sha256(exe)
    build_provenance = saved
else:
    makefile = output/'amr_lifecycle.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    build_provenance.update(executable_sha256=sha256(exe), harness_object_sha256=sha256(obj))
    (output/'build-provenance.json').write_text(json.dumps(build_provenance, indent=2)+'\n')
if args.build_only:
    assert all(sha256(root/path) == sha for path, sha in source_manifest.items())
    assert all(sha256(build/path) == sha for path, sha in object_manifest.items())
    print(json.dumps(build_provenance, indent=2))
    raise SystemExit(0)


def snapshot(folder, phase, ranks, expected):
    blocks, owners, last = {}, {}, {}
    count = 0
    size = 9*4+39*8
    for rank in range(ranks):
        stem = folder/f'phase{phase}-rank{rank}'
        leaves = [tuple(map(int, line.split())) for line in
                  stem.with_suffix('.txt').read_text().splitlines()]
        for leaf in leaves:
            assert leaf not in owners, ('duplicated native leaf', leaf)
            owners[leaf] = rank
            blocks[leaf] = bytearray()
        raw = stem.with_suffix('.bin').read_bytes()
        assert len(raw) % size == 0
        for offset in range(0, len(raw), size):
            key = struct.unpack_from('<9i', raw, offset)
            leaf, point = key[:5], key[5:]
            assert leaf in leaves, ('point outside rank-owned topology', key)
            assert leaf not in last or last[leaf] < point, ('duplicate/unordered point', key)
            last[leaf] = point
            blocks[leaf].extend(raw[offset:offset+size])
            count += 1
    assert len(blocks) == expected['blocks']
    assert count == expected['bulk_points']+expected['scri_points']
    digest = hashlib.sha256()
    with (folder/f'phase{phase}-canonical.bin').open('wb') as target:
        for leaf in sorted(blocks):
            assert blocks[leaf], ('owned leaf has no state', leaf)
            digest.update(blocks[leaf])
            target.write(blocks[leaf])
    return blocks, owners, {'points': count, 'values': 39*count,
                            'sha256': digest.hexdigest()}


reports = {}
try:
    for ranks in args.ranks:
        folder = output/f'ranks{ranks}'
        folder.mkdir(exist_ok=True)
        command = ['mpiexec', '-n', str(ranks), str(exe),
                   str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
        if args.diagnostics:
            command.append('--diagnostics')
        with (folder/'run.log').open('w') as log:
            run = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        text = (folder/'run.log').read_text()
        print(text, flush=True)
        run.check_returncode()
        phases = [json.loads(line.removeprefix('AMR_REPORT '))
                  for line in text.splitlines() if line.startswith('AMR_REPORT ')]
        assert [p['phase'] for p in phases] == [0, 1, 2]
        assert [p['blocks'] for p in phases] == [106, 127, 106]
        assert [p['scri_points'] for p in phases] == [864, 1296, 864]
        previous = previous_owners = None
        for phase in phases:
            blocks, owners, canonical = snapshot(folder, phase['phase'], ranks, phase)
            phase['canonical'] = canonical
            if previous is not None:
                unchanged = sorted(previous.keys() & blocks.keys())
                migrated = [leaf for leaf in unchanged if previous_owners[leaf] != owners[leaf]]
                assert unchanged and migrated, 'missing unchanged/migrated AMR coverage'
                for leaf in unchanged:
                    assert previous[leaf] == blocks[leaf], ('unchanged leaf differs', leaf)
                phase['unchanged_leaves'] = len(unchanged)
                phase['unchanged_migrated_leaves'] = len(migrated)
                phase['unchanged_values'] = sum(len(blocks[leaf])//348*39 for leaf in unchanged)
            previous, previous_owners = blocks, owners
        diagnostics = [json.loads(line.removeprefix('DIAGNOSTICS_REPORT '))
                       for line in text.splitlines() if line.startswith('DIAGNOSTICS_REPORT ')]
        if args.diagnostics:
            assert [p['phase'] for p in diagnostics] == [0, 1, 2], diagnostics
            assert [p['refreshes'] for p in diagnostics] == [1, 2, 3], diagnostics
            for phase, diagnostic in zip(phases, diagnostics):
                assert diagnostic['constraint_weight_scalars'] == 23*phase['bulk_points']
                assert diagnostic['owned_scalars'] == 39*(phase['bulk_points']+phase['scri_points'])
                assert diagnostic['rk_status_values'] > 0
                assert all(diagnostic[key] is True for key in
                           ('fresh_exact', 'repeat_cached', 'readonly'))
        else:
            assert not diagnostics
        reports[str(ranks)] = {'command': command, 'phases': phases,
                              'diagnostics': diagnostics}
    cross_rank = len(reports) > 1
    if cross_rank:
        first = next(iter(reports.values()))['phases']
        for report in reports.values():
            assert [p['canonical'] for p in report['phases']] == [p['canonical'] for p in first]
    result = {'passed': True, 'scope': 'native field AMR transfer, local CPU MPI',
              'coupled_evolution_validated': False,
              'actual_remesh_diagnostics_validated': args.diagnostics, 'runs': reports,
              'cross_rank_complete_state_exact': cross_rank,
              'executable_sha256': sha256(exe), 'build_provenance': build_provenance}
finally:
    provenance = {'sources_unchanged': all(sha256(root/p) == digest
                    for p, digest in source_manifest.items()),
                  'objects_unchanged': all(sha256(build/p) == digest
                    for p, digest in object_manifest.items())}
    result = locals().get('result', {'passed': False, 'runs': reports})
    result['provenance'] = provenance
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
assert all(provenance.values()), provenance
print(json.dumps(result, indent=2))
