#!/usr/bin/env python3
"""Exercise radial subdivisions through the native mesh, donor and scri code."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--ranks', type=int, default=2)
p.add_argument('--inner-blocks', type=int, default=1)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
build, output = a.build.resolve(), a.output.resolve()
output.mkdir(parents=True, exist_ok=True)
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M)[1])
obj, exe = output/'radial_blocks.o', output/'radial_blocks'
compile_cmd = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
               '-c', str(root/'tst/z4c_hyperboloidal/radial_blocks.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
assert str(obj) in link
link[link.index('-o')+1] = str(exe)
with (output/'build.log').open('w') as log:
    for command in (compile_cmd, link):
        subprocess.run(command, cwd=build/'src', stdout=log, stderr=subprocess.STDOUT, check=True)
base = (root/'tst/z4c_hyperboloidal/n12_initialize.athinput').read_text()
assert a.inner_blocks > 0
counts = ','.join(map(str, [a.inner_blocks]+[1]*9+[3,4]))
text = base.replace('<mesh_atlas>', '<mesh_atlas>\nshell_radial_blocks = '+counts)
inp = output/'radial_blocks.athinput'
inp.write_text(text)
command = ['mpiexec', '-n', str(a.ranks), str(exe), str(inp)]
with (output/'run.log').open('w') as log:
    subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                   check=True, timeout=300)
summary = re.search(r'PASS radial polynomials=(\d+) scri_points=(\d+) blocks=(\d+)',
                    (output/'run.log').read_text())
blocks = 136+6*(a.inner_blocks-1)
assert summary and tuple(map(int, summary.groups())) == (756,864,blocks)
# The real input parser must reject malformed or inconsistent counts before
# allocating/evolving the mesh. Use mesh-only mode for these small checks.
rejected = []
for specification in ['1,2', ','.join(['1']*11+['0']), '1,2.5', '1,2,', '1,,2', '2147483648']:
    bad = output/'invalid.athinput'
    bad.write_text(base.replace('<mesh_atlas>', '<mesh_atlas>\nshell_radial_blocks = '+specification))
    result = subprocess.run([str(build/'src/athena'), '-i', str(bad), '-m', '1'],
                            cwd=output, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and ('shell_radial_blocks' in result.stdout+result.stderr
                                     or 'shell radial block counts' in result.stdout+result.stderr)
    rejected.append(specification)
report = dict(passed=True, ranks=a.ranks, radial_polynomial_checks=756,
              scri_points=864, blocks=blocks, inner_blocks=a.inner_blocks, rejected_inputs=rejected,
              scope='Native interpolation and scri polynomial consistency; not long-time stability')
(output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
