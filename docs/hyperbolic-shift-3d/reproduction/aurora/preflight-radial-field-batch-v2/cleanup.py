from pathlib import Path
import hashlib,json,sys,time
root=Path('/home/hzhu/hyperbolic-core-shell-20261028')
sys.path.insert(0,str(root/'source-v24/tst/hyperbolic_shift_3d/aurora'))
import storage
m=json.loads((root/'purge-radial-v2-copies.json').read_text());receipt=root/'purge-radial-v2-copies-receipt.json'
assert not receipt.exists() and len(m)==6
campaign=root/'campaigns/preflight-radial-field-batch-v2'
terminal=json.loads((campaign/'collected.json').read_text())
assert terminal['scheduler']['job_state']=='F' and terminal['scheduler']['Exit_status']==0
assert (campaign/'allocation-released.json').is_file()
lock=root/'.hyperbolic-debug-allocation-lock/owner.json'
owner=lock.read_bytes() if lock.exists() else None
active_plan=None
if owner is not None:
 name=json.loads(owner)['plan'];assert '..' not in Path(name).parts and not Path(name).is_absolute()
 active_plan=(root/name).read_text()
 assert all(name not in active_plan for name in m), 'Other active plan references a checkpoint to purge'
for name,item in m.items():
 p=root/name
 assert name.startswith('campaigns/preflight-radial-field-batch-v2/output/') and '..' not in Path(name).parts and p.suffix=='.checkpoint'
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
assert (lock.read_bytes() if lock.exists() else None)==owner,'Allocation changed during cleanup verification'
before=storage.footprint(root)
for name in m:(root/name).unlink()
r=dict(status='passed',removed=m,verified_local_copies_before_remote_removal=True,
 other_allocation_owner=json.loads(owner) if owner else None,
 other_plan_does_not_reference_removed_files=True,before=before,after=storage.footprint(root),timestamp=time.time())
receipt.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['status','before','after','other_allocation_owner']}))
