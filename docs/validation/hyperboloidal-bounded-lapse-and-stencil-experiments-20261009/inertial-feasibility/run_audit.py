"""Fresh five-command mathematical feasibility gate; no alternative evolution."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((P.parent/'covariant-z4-candidate/receipt.json').read_text())
flags = prior['commands'][0]['command'][:-3]
debug = flags.copy()
debug[debug.index('-O3')] = '-O1'
debug.remove('-DNDEBUG')
debug += ['-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
files = [ROOT/name for name in subprocess.check_output(
    ['git', 'ls-files', 'src', 'CMakeLists.txt'], cwd=ROOT, text=True).splitlines()]
files += [P/name for name in ('normal_kernel.cpp', 'check_projection.py', 'run_audit.py')]
files += [P.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp']
before = {str(f.relative_to(ROOT)): sha(f) for f in files}
commands = [(flags+[str(P/'normal_kernel.cpp'), '-o', str(P/'normal_kernel')], None),
            (debug+[str(P/'normal_kernel.cpp'), '-o', str(P/'normal_debug')], None),
            ([str(P/'normal_kernel')], 'normal-release.json'),
            ([str(P/'normal_debug')], 'normal-debug.json'),
            ([sys.executable, str(P/'check_projection.py')], 'projection.log')]
rows = []
started = time.monotonic()
for command, output in commands:
    start = time.monotonic()
    q = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    row = {'command': command, 'returncode': q.returncode,
           'stderr': q.stderr, 'seconds': time.monotonic()-start}
    if output:
        (P/output).write_text(q.stdout)
        row.update(stdout_file=output, stdout_sha256=sha(P/output))
    else:
        row['stdout'] = q.stdout
    rows.append(row)
    (P/'commands-in-progress.json').write_text(json.dumps(rows, indent=2)+'\n')
    if q.returncode:
        (P/'failed-receipt.json').write_text(json.dumps(
            {'commands': rows, 'source_before': before}, indent=2)+'\n')
        q.check_returncode()
assert (P/'normal-release.json').read_bytes() == (P/'normal-debug.json').read_bytes()
after = {str(f.relative_to(ROOT)): sha(f) for f in files}
assert before == after
result = {'status': 'PASS', 'scope': 'General-t damping math feasibility, no altered kernel/native admission',
          'commands': rows, 'source_before': before, 'source_after': after,
          'sources_unchanged': True, 'source_count': len(before),
          'seconds': time.monotonic()-started,
          'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'runtime_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
          'projection_sha256': sha(P/'projection.json'),
          'binaries': {n: sha(P/n) for n in ('normal_kernel', 'normal_debug')}}
(P/'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print('PASS', sha(P/'receipt.json'))
