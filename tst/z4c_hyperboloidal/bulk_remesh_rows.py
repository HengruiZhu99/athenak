#!/usr/bin/env python3
"""Bounded geometry-only polynomial/convergence tests of native remesh rows."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--build', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--ranks', type=int, default=2)
args = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, output = args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
tracked = sorted(list((root/'src').rglob('*.cpp'))+list((root/'src').rglob('*.hpp')))
tracked += [root/'tst/z4c_hyperboloidal'/name for name in
            ['bulk_remesh_rows.cpp', 'bulk_remesh_rows_contract.md']]
hashes = {str(x.relative_to(root)): hashlib.sha256(x.read_bytes()).hexdigest() for x in tracked}
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M)[1])
obj, exe = output/'bulk_remesh_rows.o', output/'bulk_remesh_rows'
compile = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off']+includes+[
    '-c', str(root/'tst/z4c_hyperboloidal/bulk_remesh_rows.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
link[link.index('-o')+1] = str(exe)
makefile = output/'test.mk'
makefile.write_text('all:\n\t'+shlex.join(compile)+'\n\t'+shlex.join(link)+'\n')
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src',
                   stdout=log, stderr=subprocess.STDOUT, check=True)
objects = [(pathlib.Path(x) if pathlib.Path(x).is_absolute() else build/'src'/x)
           for x in link if x.endswith('.o')]
object_hashes = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in objects}
command = ['mpiexec', '-n', str(args.ranks), str(exe),
           str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
with (output/'run.log').open('w') as log:
    run = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
text = (output/'run.log').read_text()
metrics = []
for line in text.splitlines():
    if line.startswith(('bulk_rows ', 'overlap_rows ', 'convergence ', 'overlap_convergence ', 'fallback_convergence ')):
        record = {'kind': line.split()[0]}
        record.update({k: float(v) for k, v in re.findall(r'(\w+)=([^ ]+)', line)})
        metrics.append(record)
unchanged = hashes == {str(x.relative_to(root)): hashlib.sha256(x.read_bytes()).hexdigest() for x in tracked}
objects_unchanged = object_hashes == {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in objects}
result = {'passed': run.returncode == 0, 'source_unchanged': unchanged,
    'objects_unchanged': objects_unchanged, 'object_sha256': object_hashes,
    'scope': 'Geometry-only rows at N12/N24/N48; no fields allocated, no evolution or performance claim',
    'local_mpi_ranks': args.ranks, 'metrics': metrics, 'source_sha256': hashes,
    'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
    'compile_command': compile, 'command': command,
    'polynomial_gate': '4096*epsilon*max(1,sum(abs(weight*polynomial)),abs(expected))',
    'convergence_gate': 'observed order >=4.5 unless relative error <=4096*epsilon'}
(output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print('\n'.join(line for line in text.splitlines() if line.startswith(
    ('bulk_rows ', 'overlap_rows ', 'convergence ', 'overlap_convergence ', 'fallback_convergence ', 'PASS '))))
run.check_returncode()
assert unchanged, 'source changed during metadata test'
assert objects_unchanged, 'linked objects changed during metadata test'
assert len(metrics) == 18+3+12+2+4, 'incomplete metadata checks'
