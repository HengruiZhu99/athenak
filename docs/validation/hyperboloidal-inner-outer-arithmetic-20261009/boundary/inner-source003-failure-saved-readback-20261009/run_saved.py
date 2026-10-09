#!/usr/bin/env python3
"""One-shot outer capture of stdlib saved-failure association, not science."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if sys.flags.optimize or not sys.dont_write_bytecode:
        raise RuntimeError('Unoptimized -B required')
    root = Path(__file__).resolve().parent
    recipe = json.loads((root/'recipe.json').read_text())
    outer = root/'invocation001'
    outer.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1')
    command = [sys.executable, '-I', '-B', str(root/'read_saved.py'),
               '--recipe', str(root/'recipe.json'), '--attempt', str(root/'attempt001')]
    invocation = {'command': command, 'environment': {k:env[k] for k in recipe['environment']},
                  'source_sha256': digest(root/'read_saved.py'), 'runner_sha256':digest(__file__),
                  'recipe_sha256':digest(root/'recipe.json'), 'scope':recipe['scope']}
    (outer/'invocation.json').write_text(json.dumps(invocation,indent=2,sort_keys=True)+'\n')
    start = time.monotonic()
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (outer/'stdout.log').write_bytes(result.stdout)
    (outer/'stderr.log').write_bytes(result.stderr)
    receipt = {**invocation, 'returncode':result.returncode, 'elapsed_seconds':time.monotonic()-start,
               'completed':result.returncode==0, 'stdout_sha256':digest(outer/'stdout.log'),
               'stderr_sha256':digest(outer/'stderr.log')}
    (outer/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(result.stdout.decode(),end='')
    if result.stderr:
        print(result.stderr.decode(),end='',file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
