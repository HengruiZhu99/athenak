#!/usr/bin/env python3
"""Compare native mapped donor plans with the preserved N12 legacy selector.

The legacy constructor ordinarily materializes all global halo rows twice. The
test overlay suppresses that single constructor call only; native rows are
compared with unmodified legacy Interpolate on demand. Production is read-only.
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
parser.add_argument('--legacy', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--ranks', type=int, default=2)
parser.add_argument('--shell-smr', action='store_true')
parser.add_argument('--full-rows', action='store_true')
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, legacy, output = args.build.resolve(), args.legacy.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
tracked = ['src/z4c_hyperboloidal/mapped_halo.cpp',
           'src/z4c_hyperboloidal/mapped_halo.hpp',
           'src/z4c_hyperboloidal/fields.hpp',
           'src/bvals/mapped_exchange.cpp', 'src/bvals/mapped_exchange.hpp',
           'src/mesh/chart_atlas.cpp', 'src/mesh/chart_atlas.hpp',
           'tst/z4c_hyperboloidal/mapped_halo_oracle.cpp']
source_hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in tracked}
relative = pathlib.Path('mesh/cubed_sphere/composite_difference.hpp')
source = (legacy/'src'/relative).read_text()
needle = '    BuildHalos();\n  }\n  int Index('
assert source.count(needle) == 1
overlay = output/'legacy-test-overlay'/relative
overlay.parent.mkdir(parents=True, exist_ok=True)
overlay.write_text(source.replace(needle,
    '    // Test harness: query unchanged Interpolate on demand.\n  }\n  int Index('))
(output/'overlay-provenance.json').write_text(json.dumps({
    'source': str(legacy/'src'/relative),
    'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'test_overlay_sha256': hashlib.sha256(overlay.read_bytes()).hexdigest(),
    'change': 'Suppress the single constructor BuildHalos call; Interpolate unchanged',
    'weight_tolerance': '1e-12 + 5e-10 * abs(reference)',
    'fill_absolute_tolerance': 1e-12,
}, indent=2)+'\n')
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
objects = []
rules = []
for name, path in [('test', root/'tst/z4c_hyperboloidal/mapped_halo_oracle.cpp'),
                   ('halo', root/'src/z4c_hyperboloidal/mapped_halo.cpp')]:
    obj = output/(name+'.o'); objects.append(obj)
    command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off'] + includes
    command += ['-I'+str(output/'legacy-test-overlay'), '-I'+str(legacy/'src'),
                '-I'+str(legacy), '-c', str(path), '-o', str(obj)]
    rules.append(str(obj)+': FORCE '+str(path)+' '+str(root/'src/z4c_hyperboloidal/mapped_halo.hpp')+
                 '\n\t'+shlex.join(command)+'\n')
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(objects[0]) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
link = [x for x in link if not x.endswith('/mapped_halo.cpp.o')]
link.append(str(objects[1]))
exe = output/'mapped_halo_oracle'
link[link.index('-o')+1] = str(exe)
makefile = output/'oracle.mk'
makefile.write_text('all: '+shlex.join([str(x) for x in objects])+'\n\t'+
                   shlex.join(link)+'\n\n.PHONY: all FORCE\nFORCE:\n\n'+'\n'.join(rules))
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                   stderr=subprocess.STDOUT, check=True)
command = ['mpiexec', '-n', str(args.ranks), str(exe),
           str(root/'tst/z4c_hyperboloidal'/('shell_smr.athinput' if args.shell_smr
                                          else 'n12_initialize.athinput'))]
if args.shell_smr:
    command.append('shell')
if args.full_rows:
    command.append('full')
with (output/'run.log').open('w') as log:
    subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
assert source_hashes == {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in tracked}, \
    'test sources changed during oracle execution'
result = {'ranks': args.ranks, 'shell_smr': args.shell_smr, 'full_rows': args.full_rows,
          'source_sha256': source_hashes,
          'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
          'command': command,
          'pass_line': next(line for line in (output/'run.log').read_text().splitlines()
                            if line.startswith('PASS '))}
(output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print((output/'run.log').read_text())
