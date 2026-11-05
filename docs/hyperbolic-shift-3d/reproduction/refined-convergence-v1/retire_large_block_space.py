"""Retire five individually preserved obsolete outputs, with a durable receipt."""
from pathlib import Path
import hashlib,json,sys,time
root=Path(sys.argv[1]).resolve();request=Path(sys.argv[2]).resolve()
receipt=root/'retired-before-large-block-pair.json'
assert not receipt.exists(),'Existing receipt must be reconciled, never repeated'
x=json.loads(request.read_text());assert x['status']=='local_reverification_complete_remote_retirement_pending'
assert str(root)==x['root']
allowed={'stationary-reference-v2/campaigns/stationary20-reference-n16-segment01/output/state.checkpoint',
 'stationary-reference-v2/campaigns/stationary20-reference-n16-segment02/output/state.checkpoint',
 'campaigns/stationary6-n16-v2-segment01/output/state.checkpoint',
 'frame-separation-v1/campaigns/stationary20-separated-n16-segment01/output/state.failed-stage.bin',
 'frame-separation-v1/campaigns/stationary20-constant-n16-segment01/output/state.failed-stage.bin'}
assert len(x['files'])==5 and {v['remote_relative'] for v in x['files']}==allowed
active=[]
for base in [root]+[p for p in root.iterdir() if p.is_dir() and not p.is_symlink()]:
    lock=base/'allocation-lock.json'
    if not lock.exists():continue
    state=json.loads(lock.read_text());plan=json.loads((base/state['plan']).read_text())
    active.append(state)
    bound={(base/p).resolve() for p in plan['bindings']}
    for v in x['files']:assert (root/v['remote_relative']).resolve() not in bound
for v in x['files']:
    assert v['local_fresh_hash_verified']
    p=(root/v['remote_relative']).resolve();assert p.is_relative_to(root) and p.is_file() and not p.is_symlink()
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    assert h.hexdigest()==v['sha256'] and p.stat().st_size==v['bytes']
    v.update(remote_hash_verified=True,allocated_bytes=p.stat().st_blocks*512,links=p.stat().st_nlink,retired=False)
x.update(status='remote_hashes_verified_before_retirement',active_allocations_preserved=active,
 request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),verified_time=time.time())
def save():
    p=receipt.with_suffix('.json.tmp');p.write_text(json.dumps(x,indent=2)+'\n');p.replace(receipt)
save()
for v in x['files']:
    (root/v['remote_relative']).unlink();v.update(retired=True,retired_time=time.time());save()
x.update(status='completed',logical_bytes_retired=sum(v['bytes'] for v in x['files']),
 unique_file_allocated_bytes_retired=sum(v['allocated_bytes'] for v in x['files'] if v['links']==1))
save();print(json.dumps(x))
