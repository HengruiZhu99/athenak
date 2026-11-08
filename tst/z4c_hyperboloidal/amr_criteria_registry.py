#!/usr/bin/env python3
"""Targeted real-native criteria/registry checks against one built AthenaK.

This test supplies synthetic field values. It neither evolves nor tests regrids;
the separate coupled AMR oracle covers transfer operators and lifecycle wiring.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess

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

stem = 'amr_criteria_registry'
source = root/f'tst/z4c_hyperboloidal/{stem}.cpp'
fixture = root/'tst/z4c_hyperboloidal/n12_initialize.athinput'
sources = [source, pathlib.Path(__file__).resolve(), fixture]
sources += [root/f'src/mesh/{name}' for name in
            ('mesh_refinement.cpp', 'mesh_refinement.hpp', 'load_balance.cpp',
             'refinement_criteria.cpp', 'refinement_criteria.hpp')]
source_manifest = {str(path.relative_to(root)): sha256(path) for path in sources}
(output/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
obj, exe = output/f'{stem}.o', output/stem
compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                   '-c', str(source), '-o', str(obj)]
link_command = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link_command = [str(obj) if part == 'CMakeFiles/athena.dir/main.cpp.o' else part
                for part in link_command]
assert str(obj) in link_command
link_command[link_command.index('-o')+1] = str(exe)
objects = [build/'src'/part for part in link_command
           if part.endswith('.o') and part != str(obj)]
object_manifest = {str(path.relative_to(build)): sha256(path) for path in objects}
(output/'object-manifest.json').write_text(json.dumps(object_manifest, indent=2)+'\n')
makefile = output/f'{stem}.mk'
makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'
                   +shlex.join(link_command)+'\n')
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                   stderr=subprocess.STDOUT, check=True)
command = ['mpiexec', '-n', str(args.ranks), str(exe), str(fixture)]
with (output/'run.log').open('w') as log:
    result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
report = {'command': command, 'exit_code': result.returncode,
          'scope': 'synthetic native criteria and temporary CC registry; no AMR evolution',
          'executable_sha256': sha256(exe), 'harness_object_sha256': sha256(obj),
          'source_manifest_sha256': sha256(output/'source-manifest.json'),
          'object_manifest_sha256': sha256(output/'object-manifest.json'),
          'sources_unchanged': all(sha256(root/path) == digest
                                    for path, digest in source_manifest.items()),
          'objects_unchanged': all(sha256(build/path) == digest
                                    for path, digest in object_manifest.items())}
(output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
print((output/'run.log').read_text())
result.check_returncode()
assert report['sources_unchanged'] and report['objects_unchanged']
