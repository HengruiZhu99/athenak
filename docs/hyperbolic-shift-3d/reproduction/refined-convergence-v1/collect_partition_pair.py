"""Verify a terminal ownership experiment and release only its matching lock."""
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
pf = root / sys.argv[2]
sys.path.insert(0, str(root / 'launcher-v7'))
from control import read, write, digest, bindings, verify

plan = read(pf)
bindings(root, plan)
assert plan['purpose'] == 'static_ownership_pair'
submission = read(pf.parent / 'submission.json')
q = subprocess.run(['qstat', '-xf', submission['job_id']], text=True, capture_output=True)
(pf.parent / 'terminal-qstat.txt').write_text(q.stdout + q.stderr)
assert q.returncode == 0 and re.search(r'job_state\s*=\s*F\b', q.stdout)
exit_code = int(re.search(r'Exit_status\s*=\s*(-?\d+)', q.stdout)[1])
out = root / plan['output']
worker = read(out / 'worker.json')
assert worker['job_id'] == submission['job_id']
assert worker['plan_sha256'] == digest(pf) == submission['plan_sha256']
for name, record in worker['artifacts'].items():
    p = out / name
    assert digest(p) == record['sha256'] and p.stat().st_size == record['bytes'], name
result = dict(job_id=submission['job_id'], terminal=True, scheduler_exit=exit_code,
              worker_status=worker['status'], cases=[], parity_passed=False,
              adopted_for_physics=False)
if worker['status'] == 'completed':
    assert exit_code == 0 and len(worker['cases']) == 2
    for case, record in zip(plan['cases'], worker['cases']):
        assert record['name'] == case['name'] and record['native_returncode'] == 0
        v = verify(root, case, root / case['output'])
        assert v['status'] == 'completed'
        assert v['steps'] == case['start_steps'] + plan['measured_steps']
        result['cases'].append(v)
    result['parity_passed'] = result['cases'][0]['sha256'] == result['cases'][1]['sha256']
    result['qualification'] = ('Bitwise checkpoint parity' if result['parity_passed'] else
        'Checkpoints differ; inspect physical fields before any adoption. No automatic replay.')
else:
    assert exit_code != 0
    result['qualification'] = 'Failed bounded benchmark; no automatic replay or physics restart'
write(pf.parent / 'terminal-verification.json', result)
lock = root / 'allocation-lock.json'
if lock.exists():
    saved = read(lock)
    assert saved['job_id'] == submission['job_id'] and saved['plan_sha256'] == digest(pf)
    write(pf.parent / 'allocation-released.json', dict(job_id=submission['job_id'],
          plan_sha256=digest(pf), time=time.time(),
          terminal_verification_sha256=digest(pf.parent / 'terminal-verification.json')))
    lock.unlink()
print(result)
