#!/usr/bin/env python3
"""Link the refined scri regression against the current native AthenaK objects."""
import argparse
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
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
obj, exe = output/'surface_smr.o', output/'surface_smr'
compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off'] + includes
compile_command += ['-c', str(root/'tst/z4c_hyperboloidal/surface_smr.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
assert str(obj) in link
link[link.index('-o')+1] = str(exe)
makefile = output/'surface_smr.mk'
makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                   stderr=subprocess.STDOUT, check=True)
command = ['mpiexec', '-n', str(args.ranks), str(exe),
           str(root/'tst/z4c_hyperboloidal/surface_smr.athinput')]
with (output/'run.log').open('w') as log:
    result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                            timeout=240)
text = (output/'run.log').read_text()
status = {'passed': result.returncode == 0, 'local_mpi_ranks': args.ranks,
          'native_build': str(build), 'gate': '256*epsilon*max(1,row_L1)*4'}
match = re.search(r'outer_smr blocks=(\d+) surface_patches=(\d+) comparisons=(\d+) '
                  r'failures=(\d+) max_error_over_bound=([^ ]+) max_absolute=([^ ]+) '
                  r'collective_rebuild_rejection=(\w+)', text)
if match:
    status.update(blocks=int(match[1]), surface_patches=int(match[2]),
                  comparisons=int(match[3]), failed_values=int(match[4]),
                  max_error_over_bound=float(match[5]), max_absolute_error=float(match[6]),
                  collective_rebuild_rejection=match[7])
volume = re.search(r'nodal_volume comparisons=(\d+) internal_radial_starts=(\d+) '
                   r'analytic_jacobian=(\w+)', text)
if volume:
    status.update(nodal_volume_comparisons=int(volume[1]),
                  internal_radial_starts=int(volume[2]), analytic_jacobian=volume[3])
cache = re.search(r'radial_cache comparisons=(\d+) stale_changed=(\d+) '
                  r'poison_refresh=passed rebuild_refresh=passed bitwise=passed', text)
if cache:
    status['radial_cache'] = {'comparisons': int(cache[1]), 'stale_changed': int(cache[2]),
                             'poison_refresh': True, 'rebuild_refresh': True, 'bitwise': True}
layout=re.search(r'cache_layout rows=(\d+) fields=(\d+) jet_bytes=(\d+) bytes=(\d+)',text)
if layout:
    status['cache_layout']=dict(zip(('rows','fields','jet_bytes','bytes'),map(int,layout.groups())))
cache_valid=bool(cache and (int(cache[1]),int(cache[2]))==(3942432,101088) and layout and tuple(map(int,layout.groups()))==(3888,26,104,3888*26*104))
status['passed']=status['passed'] and cache_valid
(output/'results.json').write_text(json.dumps(status, indent=2)+'\n')
print(text)
result.check_returncode()

assert cache_valid, 'missing/incomplete radial-cache poison/allocation coverage'
