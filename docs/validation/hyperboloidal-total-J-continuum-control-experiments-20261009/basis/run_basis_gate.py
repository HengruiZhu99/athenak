from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = ['total_j_basis.py', 'check_basis.py', 'reference_conversion.hpp',
           'probe_basis.cpp', 'check_compiled.py', 'run_basis_gate.py']
before = {name: sha(P/name) for name in sources}
commands = [
    ('symbolic', [sys.executable, str(P/'check_basis.py')], None),
    ('compile-release', ['/usr/bin/c++', '-std=c++17', '-O3', '-DNDEBUG', str(P/'probe_basis.cpp'), '-o', str(P/'probe-release')], None),
    ('probe-release', [str(P/'probe-release')], P/'probe-release.json'),
    ('compile-debug', ['/usr/bin/c++', '-std=c++17', '-O0', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(P/'probe_basis.cpp'), '-o', str(P/'probe-debug')], None),
    ('probe-debug', [str(P/'probe-debug')], P/'probe-debug.json'),
    ('compiled-check', [sys.executable, str(P/'check_compiled.py')], None),
]
receipt = {'kind': 'math-only-Cartesian-total-J-basis-prototype',
           'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'source_before': before, 'commands': [], 'no_PDE_kernel_or_evolution': True,
           'python_binary_sha256': sha(Path(sys.executable).resolve()),
           'compiler_version': subprocess.check_output(['/usr/bin/c++', '--version'], text=True)}
for name, cmd, output in commands:
    t = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True)
    seconds = time.monotonic()-t
    (P/(name+'.stdout')).write_bytes(result.stdout)
    (P/(name+'.stderr')).write_bytes(result.stderr)
    if output is not None:
        output.write_bytes(result.stdout)
    receipt['commands'].append({'name': name, 'command': cmd, 'returncode': result.returncode,
                                'seconds': seconds, 'stdout': name+'.stdout', 'stderr': name+'.stderr'})
    (P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if result.returncode or result.stderr:
        print(json.dumps({'failed': name, 'returncode': result.returncode,
                          'stderr': result.stderr.decode(errors='replace')}, indent=2))
        raise SystemExit(1)
receipt['source_after'] = {name: sha(P/name) for name in sources}
receipt['source_unchanged'] = receipt['source_before'] == receipt['source_after']
assert receipt['source_unchanged']
receipt['generated_header_sha256'] = sha(P/'total_j_basis.hpp')
receipt['basis_data_sha256'] = sha(P/'basis-data.json')
receipt['binaries'] = {name: sha(P/name) for name in ['probe-release', 'probe-debug']}
(P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'passed': True, 'commands': len(receipt['commands']),
                  'seconds': sum(c['seconds'] for c in receipt['commands']),
                  'receipt_sha256': sha(P/'receipt.json')}, indent=2))
