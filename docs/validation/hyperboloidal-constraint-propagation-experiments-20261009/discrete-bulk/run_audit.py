"""Compile, run and hash the scratch bulk gates, stopping on every failure."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

p = Path(__file__).resolve().parent
repo = p.parents[2]


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


production = subprocess.check_output(
    ['git', 'ls-files', 'src', 'CMakeLists.txt'], cwd=repo, text=True).splitlines()
inputs = [repo/f for f in production]
inputs += [p/f for f in ['discrete_kernel.cpp', 'dual_helpers.hpp',
                         'check_discrete.py', 'run_audit.py', 'DERIVATION.md']]
inputs += [repo/'tst/hyperboloidal/check_kernel_symbol.py',
           repo/'build-layer-release/config.hpp']
before = {str(f.relative_to(repo)): sha(f) for f in inputs}
cmd = ['/usr/bin/c++', '-std=c++17', '-O3', '-DNDEBUG', '-DKOKKOS_DEPENDENCE',
       '-Isrc', '-Ibuild-layer-release', '-Ibuild-layer-release/kokkos',
       '-Ibuild-layer-release/kokkos/core/src', '-Ikokkos/core/src',
       '-Ibuild-layer-release/kokkos/containers/src', '-Ikokkos/containers/src',
       '-Ibuild-layer-release/kokkos/algorithms/src', '-Ikokkos/algorithms/src',
       '-Ibuild-layer-release/kokkos/simd/src', '-Ikokkos/simd/src',
       '-isystem', 'kokkos/tpls/desul/include', '-isystem', 'kokkos/tpls/mdspan/include',
       str(p/'discrete_kernel.cpp'), '-o', str(p/'discrete_kernel')]
results = []
t = time.monotonic()


def run(command, output=None):
    started = time.monotonic()
    if output is None:
        r = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    else:
        with (p/output).open('w') as stream:
            r = subprocess.run(command, cwd=repo, text=True,
                               stdout=stream, stderr=subprocess.PIPE)
    result = {'command': command, 'returncode': r.returncode,
              'stderr': r.stderr, 'seconds': time.monotonic()-started}
    if output is None:
        result['stdout'] = r.stdout
    else:
        result.update(stdout_file=output, stdout_sha256=sha(p/output))
    results.append(result)
    r.check_returncode()


run(cmd)
run([str(p/'discrete_kernel')], 'kernel-matrices.json')
run([str(p/'discrete_kernel'), '--manufactured'], 'manufactured.json')
run([sys.executable, str(p/'check_discrete.py'), str(p/'kernel-matrices.json')],
    'check.log')
after = {str(f.relative_to(repo)): sha(f) for f in inputs}
assert before == after
receipt = {'passed': True, 'launch_head': subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'runtime_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
    'scope': 'Scratch constant-coefficient interior full20 discrete symbol and '
             'constraint gauge columns; no native runtime changes or global claim.',
    'source_before': before, 'source_after': after, 'sources_unchanged': True,
    'commands': results, 'compiler': subprocess.check_output(
        ['/usr/bin/c++', '--version'], text=True),
    'binary_sha256': sha(p/'discrete_kernel'),
    'check_report_sha256': sha(p/'check-report.json'),
    'seconds': time.monotonic()-t}
(p/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('PASS', sha(p/'receipt.json'), receipt['seconds'])
