"""Run independent C1 saved-state searches; never propagate or modify inputs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = P / 'search.py'
before = sha(source)
env = dict(os.environ)
env['OPENBLAS_NUM_THREADS'] = '1'
env['PYTHONPATH'] = str(ROOT / 'build-layer-research/boundary/python-deps')
names = ['all-combined-s2', 'late2-combined-s2', 'late4-combined-s1',
         'late4-combined-s2', 'late4-combined-s4', 'late4-combined-raw-s2',
         'late4-gauge-s2', 'late4-shell-s2', 'late5-combined-s1']
results = []
start = time.monotonic()
for name in names:
    command = ['/usr/bin/python3', str(source), '--case', name, '--max-rank', '96']
    started = time.monotonic()
    with (P / (name + '.log')).open('w') as stdout:
        q = subprocess.run(command, env=env, cwd=ROOT, stdout=stdout,
                           stderr=subprocess.PIPE, text=True)
    results.append({'case': name, 'command': command, 'returncode': q.returncode,
                    'stderr': q.stderr, 'seconds': time.monotonic() - started,
                    'stdout_sha256': sha(P / (name + '.log'))})
    (P / 'commands-in-progress.json').write_text(json.dumps(results, indent=2) + '\n')
    q.check_returncode()
    print(name, 'done', results[-1]['seconds'], flush=True)
assert sha(source) == before
receipt = {'status': 'SEARCHES_COMPLETED_NOT_EIGENVALUE_ADMISSION',
           'commands': results, 'source_sha256': before,
           'runner_sha256': sha(Path(__file__)),
           'config_sha256': sha(P / 'input-pins.json'),
           'sources_unchanged': True, 'seconds': time.monotonic() - start,
           'environment': {'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': env['PYTHONPATH']},
           'scope': 'Read-only frozen C1 N16 saved-state subspaces; no propagation/native evolution/LU',
           'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
(P / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print('DONE suite', sha(P / 'receipt.json'), flush=True)
