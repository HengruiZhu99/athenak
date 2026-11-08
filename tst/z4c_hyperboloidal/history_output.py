#!/usr/bin/env python3
"""Run native history output dispatch/reduction at two and eight MPI ranks."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--ranks', type=int, nargs='+', default=[2, 8])
parser.add_argument('--modes', choices=['synthetic', 'real'], nargs='+', default=['synthetic', 'real'])
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
            root/'tst/z4c_hyperboloidal/history_output.cpp',
            root/'tst/z4c_hyperboloidal/history_output_contract.md',
            root/'tst/z4c_hyperboloidal/n12_initialize.athinput']
source_manifest = {str(p.relative_to(root)): sha256(p) for p in sorted(sources)}
(output/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
obj, exe = output/'history_output.o', output/'history_output'
compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                   '-c', str(root/'tst/z4c_hyperboloidal/history_output.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if p == 'CMakeFiles/athena.dir/main.cpp.o' else p for p in link]
assert str(obj) in link
link[link.index('-o')+1] = str(exe)
objects = [build/'src'/p for p in link if p.endswith('.o') and p != str(obj)]
object_manifest = {str(p.relative_to(build)): sha256(p) for p in sorted(objects)}
(output/'object-manifest.json').write_text(json.dumps(object_manifest, indent=2)+'\n')
makefile = output/'history_output.mk'
makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                   stderr=subprocess.STDOUT, check=True)

LABELS = ['Volume', 'H-rms', 'M-rms', 'Theta-rms', 'Z-rms',
          'eVolume', 'eH-rms', 'eM-rms', 'eTheta-rms', 'eZ-rms']

def read_history(path):
    text = path.read_text()
    headers = [line for line in text.splitlines() if line.startswith('#')]
    assert len(headers) == 2, 'history header repeated or omitted'
    columns = re.findall(r'\[\d+\]=(\S+)', headers[1])
    rows = [[float(x) for x in line.split()] for line in text.splitlines()
            if line and not line.startswith('#')]
    assert all(len(row) == len(columns) for row in rows)
    assert all(math.isfinite(x) for row in rows for x in row)
    return columns, rows

def synthetic_expected(amplitude):
    raw = [0.0]*10
    for point in range(24):
        for region in range(2):
            if region and point % 2:
                continue
            volume = point+1
            raw[5*region] += volume
            for field in range(1, 5):
                raw[5*region+field] += volume*(amplitude*(point % 5+field))**2
    output_values = raw.copy()
    for region in range(2):
        for field in range(1, 5):
            output_values[5*region+field] = math.sqrt(raw[5*region+field]/raw[5*region])
    return raw, output_values

runs = {}
try:
    for mode in args.modes:
        runs[mode] = {}
        for ranks in args.ranks:
            folder = output/f'{mode}-ranks{ranks}'
            folder.mkdir(exist_ok=False)
            command = ['mpiexec', '-n', str(ranks), str(exe),
                       str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput'), mode]
            with (folder/'run.log').open('w') as log:
                run = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
            print((folder/'run.log').read_text(), flush=True)
            run.check_returncode()
            history = folder/f'{mode}.z4c_hyperboloidal.hst'
            columns, rows = read_history(history)
            assert columns == ['time', 'dt']+LABELS
            assert len(rows) == 2, 'invalid integral cases wrote history rows'
            ordinary = {}
            if mode == 'synthetic':
                for sample, row in enumerate(rows, 1):
                    assert row[:2] == [.25*sample, .03125], 'coordinate time/dt was rescaled'
                    _, expected = synthetic_expected(sample)
                    assert all(abs(a-b) <= 1e-15*(1+abs(b)) for a, b in zip(row[2:], expected))
                expected_raw, _ = synthetic_expected(1)
                for suffix in ('hydro', 'mhd', 'z4c.user', 'user'):
                    path = folder/f'ordinary.{suffix}.hst'
                    old_columns, old_rows = read_history(path)
                    assert old_columns == ['time', 'dt', 'sum-a', 'sum-b']
                    assert old_rows == [[.5, .03125, expected_raw[0], expected_raw[1]]]
                    ordinary[suffix] = sha256(path)
            else:
                assert rows[0] == rows[1], 'repeated accepted-state output differs'
                assert rows[0][:2] == [0.0, 0.00046944203735617445]
                assert rows[0][2] > 0 and rows[0][7] > 0
                summary = folder/'native_n12.horizon_summary_0.txt'
                searches = [line.split() for line in summary.read_text().splitlines()
                            if line and not line.startswith('#')]
                assert [int(row[0]) for row in searches] == [0, 0, 0, 1], \
                    'cache refreshes did not write exactly once per new accepted state'
            runs[mode][str(ranks)] = dict(command=command, columns=columns, rows=rows,
                                        history_sha256=sha256(history), ordinary_sha256=ordinary)
        first = next(iter(runs[mode].values()))
        for report in runs[mode].values():
            if mode == 'synthetic':
                assert report['history_sha256'] == first['history_sha256']
                assert report['ordinary_sha256'] == first['ordinary_sha256']
            else:
                errors = [abs(a-b) for a, b in zip(report['rows'][0], first['rows'][0])]
                assert all(error <= 1e-12+5e-10*abs(reference)
                           for error, reference in zip(errors, first['rows'][0]))
                report['cross_rank_max_absolute'] = max(errors)
                report['cross_rank_exact'] = report['rows'] == first['rows']
    result = dict(passed=True, scope='native HistoryOutput reduction/dispatch, local CPU MPI',
                  synthetic_constraint_formula_oracle=False, runs=runs,
                  executable_sha256=sha256(exe))
finally:
    stable = dict(sources_unchanged=all(sha256(root/name) == digest
                                      for name, digest in source_manifest.items()),
                  objects_unchanged=all(sha256(build/name) == digest
                                      for name, digest in object_manifest.items()))
    result = locals().get('result', dict(passed=False, runs=runs))
    result['provenance'] = stable
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
assert all(stable.values()), stable
print(json.dumps(result, indent=2))
