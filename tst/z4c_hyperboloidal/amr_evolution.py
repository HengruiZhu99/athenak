#!/usr/bin/env python3
"""Exercise actual Driver evolution across AMR and compare complete 2/8-rank state."""
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
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, output = args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = [root/name for name in subprocess.check_output(
    ['rg', '--files', 'src'], cwd=root, text=True).splitlines()
    if name.endswith(('.cpp', '.hpp', 'CMakeLists.txt'))]
sources += [root/'CMakeLists.txt', pathlib.Path(__file__).resolve(),
            root/'tst/z4c_hyperboloidal/amr_evolution.cpp',
            root/'tst/z4c_hyperboloidal/amr_evolution_contract.md',
            root/'tst/z4c_hyperboloidal/n12_initialize.athinput']
source_manifest = {str(p.relative_to(root)): sha256(p) for p in sorted(sources)}
(output/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
obj, exe = output/'amr_evolution.o', output/'amr_evolution'
compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                   '-c', str(root/'tst/z4c_hyperboloidal/amr_evolution.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if p == 'CMakeFiles/athena.dir/main.cpp.o' else p for p in link]
assert str(obj) in link
link[link.index('-o')+1] = str(exe)
objects = [build/'src'/p for p in link if p.endswith('.o') and p != str(obj)]
object_manifest = {str(p.relative_to(build)): sha256(p) for p in sorted(objects)}
(output/'object-manifest.json').write_text(json.dumps(object_manifest, indent=2)+'\n')
makefile = output/'amr_evolution.mk'
makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                   stderr=subprocess.STDOUT, check=True)

def snapshot(folder, cycle, ranks, expected):
    records, owners, required = {}, {}, set()
    size = 9*4+39*8
    for rank in range(ranks):
        stem = folder/f'cycle{cycle}-rank{rank}'
        local = set()
        for line in stem.with_suffix('.txt').read_text().splitlines():
            data = tuple(map(int, line.split()))
            assert len(data) == 12
            leaf = data[:5]
            assert leaf not in owners, ('duplicate native leaf', leaf)
            owners[leaf] = rank
            local.add(leaf)
            il, iu, jl, ju, kl, ku, face = data[5:]
            for k in range(kl, ku+1):
                for j in range(jl, ju+1):
                    for i in range(il, iu+1):
                        required.add(leaf+(0, k, j, i))
            if face:
                for k in range(12):
                    for j in range(12):
                        required.add(leaf+(1, k, j, 12))
        raw = stem.with_suffix('.bin').read_bytes()
        assert len(raw) % size == 0
        for offset in range(0, len(raw), size):
            key = struct.unpack_from('<9i', raw, offset)
            assert key[:5] in local, ('point outside rank-owned leaf', key)
            assert key not in records, ('duplicate evolved point', key)
            records[key] = raw[offset:offset+size]
    assert records.keys() == required, ('missing/extra evolved points',
                                      len(required-records.keys()), len(records.keys()-required))
    assert len(owners) == expected['blocks']
    assert len(records) == expected['bulk_points']+expected['scri_points']
    assert sum(key[5] == 1 for key in records) == expected['scri_points']
    digest = hashlib.sha256()
    with (folder/f'cycle{cycle}-canonical.bin').open('wb') as target:
        for key in sorted(records):
            digest.update(records[key])
            target.write(records[key])
    return {'points': len(records), 'values': 39*len(records), 'sha256': digest.hexdigest()}

runs = {}
try:
    for ranks in args.ranks:
        folder = output/f'ranks{ranks}'
        folder.mkdir(exist_ok=True)
        command = ['mpiexec', '-n', str(ranks), str(exe),
                   str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
        with (folder/'run.log').open('w') as log:
            run = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        text = (folder/'run.log').read_text()
        print(text, flush=True)
        run.check_returncode()
        cycles = [json.loads(line[len('AMR_CYCLE '):]) for line in text.splitlines()
                  if line.startswith('AMR_CYCLE ')]
        stages = [json.loads(line[len('AMR_STAGE '):]) for line in text.splitlines()
                  if line.startswith('AMR_STAGE ')]
        assert [c['cycle'] for c in cycles] == list(range(5))
        assert [c['blocks'] for c in cycles] == [106, 127, 127, 106, 106]
        assert [c['scri_points'] for c in cycles] == [864, 1296, 1296, 864, 864]
        assert [(s['cycle'], s['stage']) for s in stages] == [
            (c, s) for c in range(1, 5) for s in range(1, 5)]
        for record in stages:
            previous = cycles[record['cycle']-1]
            assert record['bulk_values'] == 31*previous['bulk_points']
            assert record['scri_values'] == 31*previous['scri_points']
            assert record['compatibility'] <= 1e-7
        expected_time = 0
        for cycle in cycles:
            assert cycle['dt'] == cycles[0]['dt'] and cycle['time'] == expected_time
            expected_time += cycle['dt']
            if ranks >= 8:
                assert cycle['zero_scri_ranks'] > 0
            cycle['canonical'] = snapshot(folder, cycle['cycle'], ranks, cycle)
        runs[str(ranks)] = {'command': command, 'cycles': cycles, 'stages': stages}
    cross_rank = len(runs) > 1
    if cross_rank:
        first = next(iter(runs.values()))
        for run in runs.values():
            assert [c['canonical'] for c in run['cycles']] == [
                c['canonical'] for c in first['cycles']], 'cross-rank accepted-state mismatch'
            assert run['stages'] == first['stages'], 'cross-rank stage diagnostic mismatch'
    result = {'passed': True, 'scope': 'four native Driver cycles/two AMR events, local CPU MPI',
              'test_only_adaptive_admission_override': True, 'legacy_comparison': False,
              'cross_rank_complete_state_exact': cross_rank, 'runs': runs,
              'executable_sha256': sha256(exe)}
finally:
    provenance = {'sources_unchanged': all(sha256(root/p) == digest
                    for p, digest in source_manifest.items()),
                  'objects_unchanged': all(sha256(build/p) == digest
                    for p, digest in object_manifest.items())}
    result = locals().get('result', {'passed': False, 'runs': runs})
    result['provenance'] = provenance
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
assert all(provenance.values()), provenance
print(json.dumps(result, indent=2))
