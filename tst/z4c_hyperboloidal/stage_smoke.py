#!/usr/bin/env python3
"""Run one instrumented cycle through native Driver and its real task lists.

Link the candidate's AthenaK objects, replacing only main with the test harness.
The production dynamic gate stays in place. This is no legacy numerical oracle.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import struct

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--ranks', type=int, default=2)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, output = args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_paths = [root/name for name in subprocess.check_output(
    ['rg', '--files', 'src'], cwd=root, text=True).splitlines()
                if name.endswith(('.cpp', '.hpp', 'CMakeLists.txt'))]
source_paths += [root/'CMakeLists.txt', pathlib.Path(__file__).resolve(),
                 root/'tst/z4c_hyperboloidal/stage_smoke.cpp',
                 root/'tst/z4c_hyperboloidal/n12_initialize.athinput']
source_manifest = {str(path.relative_to(root)): sha256(path)
                   for path in sorted(source_paths)}
(output/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
obj, exe = output/'stage_smoke.o', output/'stage_smoke'
compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                   '-c', str(root/'tst/z4c_hyperboloidal/stage_smoke.cpp'), '-o', str(obj)]
link_command = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link_command = [str(obj) if part == 'CMakeFiles/athena.dir/main.cpp.o' else part
                for part in link_command]
assert str(obj) in link_command
link_command[link_command.index('-o')+1] = str(exe)
object_paths = [build/'src'/part for part in link_command
                if part.endswith('.o') and part != str(obj)]
object_manifest = {str(path.relative_to(build)): sha256(path)
                   for path in sorted(object_paths)}
(output/'object-manifest.json').write_text(json.dumps(object_manifest, indent=2)+'\n')
makefile = output/'stage_smoke.mk'
makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'
                    +shlex.join(link_command)+'\n')
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                   stderr=subprocess.STDOUT, check=True)
command = ['mpiexec', '-n', str(args.ranks), str(exe),
           str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
with (output/'run.log').open('w') as log:
    result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
report = {'command': command, 'exit_code': result.returncode,
          'scope': 'one native Driver RK4 cycle, local CPU MPI ranks',
          'legacy_comparison': False, 'production_dynamic_enabled': False,
          'timestep_integration_verified': False,
          'provenance': {
              'executable_sha256': sha256(exe),
              'harness_object_sha256': sha256(obj),
              'source_manifest_sha256': sha256(output/'source-manifest.json'),
              'object_manifest_sha256': sha256(output/'object-manifest.json'),
              'native_object_count': len(object_manifest),
              'source_count': len(source_manifest),
              'sources_unchanged_during_test': all(sha256(root/path) == digest
                  for path, digest in source_manifest.items()),
              'objects_unchanged_during_test': all(sha256(build/path) == digest
                  for path, digest in object_manifest.items())}}
assert report['provenance']['sources_unchanged_during_test']
assert report['provenance']['objects_unchanged_during_test']
if result.returncode == 0:
    report['accepted_state'] = {}
    for kind, dimensions, expected in [('bulk', 4, 229392), ('scri', 3, 864)]:
        key_format = '<'+'i'*dimensions
        record_size = dimensions*4+39*8
        records = []
        for rank in range(args.ranks):
            raw = (output/f'accepted-{kind}-rank{rank}.bin').read_bytes()
            count = struct.unpack_from('<Q', raw)[0]
            assert len(raw) == 8+count*record_size
            for offset in range(8, len(raw), record_size):
                records.append((struct.unpack_from(key_format, raw, offset),
                                raw[offset:offset+record_size]))
        records.sort(key=lambda item: item[0])
        assert len(records) == expected
        assert len({item[0] for item in records}) == expected
        digest = hashlib.sha256()
        with (output/f'accepted-{kind}-canonical.bin').open('wb') as target:
            for _, record in records:
                digest.update(record)
                target.write(record)
        report['accepted_state'][kind] = {
            'count': expected, 'key': 'native_gid,k,j,i' if dimensions == 4 else 'native_gid,k,j',
            'state_fields': 31, 'chi_record_fields': 8,
            'encoding': 'little-endian int32 key, float64 state and records',
            'sha256': digest.hexdigest()}
(output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
print((output/'run.log').read_text())
result.check_returncode()
