"""One-time retirement of the three explicitly preserved obsolete checkpoints."""
from pathlib import Path
import hashlib
import json
import sys
import time

root = Path(sys.argv[1]).resolve()
request = Path(sys.argv[2]).resolve()
receipt = root / 'retired-before-boosted-transition.json'
assert not receipt.exists(), 'Existing receipt: reconcile; do not repeat retirement'
x = json.loads(request.read_text())
assert x['status'] == 'local_reverification_complete_remote_retirement_pending'
allowed = {
    'campaigns/stationary20-radial18-ownership-pair-v1/output/equal/state.checkpoint',
    'campaigns/stationary20-radial18-ownership-pair-v1/output/weighted/state.checkpoint',
    'campaigns/stationary20-reference-time-dt4-n16-segment01/output/state.checkpoint',
}
assert {v['remote_relative'] for v in x['files']} == allowed and len(x['files']) == 3
lock = json.loads((root / 'allocation-lock.json').read_text())
plan = json.loads((root / lock['plan']).read_text())
assert lock['job_id'].startswith('9040850.')
for v in x['files']:
    assert v['remote_relative'] not in plan['bindings']
    assert v['remote_relative'] != plan.get('restart')
    p = (root / v['remote_relative']).resolve()
    assert p.is_relative_to(root) and p.is_file() and not p.is_symlink()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    assert h.hexdigest() == v['sha256'] and p.stat().st_size == v['bytes']
    v.update(remote_hash_verified=True, allocated_bytes=p.stat().st_blocks*512,
             links=p.stat().st_nlink, retired=False)
x.update(status='remote_hashes_verified_before_retirement', verified_time=time.time(),
         active_job_preserved=lock['job_id'], request_sha256=hashlib.sha256(request.read_bytes()).hexdigest())

def save():
    tmp = receipt.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(x, indent=2) + '\n')
    tmp.replace(receipt)

save()
for v in x['files']:
    (root / v['remote_relative']).unlink()
    v.update(retired=True, retired_time=time.time())
    save()
x['status'] = 'completed'
x['logical_bytes_retired'] = sum(v['bytes'] for v in x['files'])
x['unique_file_allocated_bytes_retired'] = sum(v['allocated_bytes'] for v in x['files'] if v['links'] == 1)
save()
print(json.dumps(x))
