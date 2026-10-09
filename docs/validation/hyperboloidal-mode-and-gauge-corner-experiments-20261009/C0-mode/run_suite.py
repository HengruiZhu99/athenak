"""Execute separate pinned reduced-state searches, without new propagation."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = P/'reduced_modes.py'
before = sha(source)
env = dict(os.environ)
env['OPENBLAS_NUM_THREADS'] = '1'
env['PYTHONPATH'] = str(ROOT/'build-layer-research/boundary/python-deps')
names = ['all-combined-s2', 'late2-combined-s2', 'late4-combined-s1',
         'late4-combined-s2', 'late4-combined-s4', 'late4-combined-raw-s2',
         'late4-gauge-s2', 'late4-shell-s2', 'late5-combined-s1']
results = []
start = time.monotonic()
for case in names:
    command = ['/usr/bin/python3', str(source), '--case', case, '--max-rank', '96']
    started = time.monotonic()
    with (P/(case+'.log')).open('w') as log:
        q = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.PIPE, text=True)
    results.append({'case': case, 'command': command, 'returncode': q.returncode,
                    'stderr': q.stderr, 'seconds': time.monotonic()-started,
                    'stdout_sha256': sha(P/(case+'.log'))})
    (P/'commands-in-progress.json').write_text(json.dumps(results, indent=2)+'\n')
    q.check_returncode()
    print(case, 'done', results[-1]['seconds'], flush=True)
assert sha(source) == before
receipt = {'status': 'SEARCHES_COMPLETED_NOT_MODE_ADMISSION', 'commands': results,
           'source_sha256': before, 'runner_sha256': sha(Path(__file__)),
           'sources_unchanged': True, 'seconds': time.monotonic()-start,
           'environment': {'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': env['PYTHONPATH']},
           'scope': 'Read-only frozen C0 N16 late-state subspace search, no propagation/native/LU',
           'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
(P/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print('DONE suite', sha(P/'receipt.json'), flush=True)
