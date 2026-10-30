from pathlib import Path
import hashlib,json,sys,time
root=Path('/home/hzhu/hyperbolic-core-shell-20261028')
sys.path.insert(0,str(root/'source-v24/tst/hyperbolic_shift_3d/aurora'))
import storage,control
manifest=root/'old-stability-remote-copies-v24.json'
m=json.loads(manifest.read_text());receipt=root/'purge-old-stability-v24-receipt.json'
assert not receipt.exists()
assert not (root/'.hyperbolic-debug-allocation-lock').exists()
assert len(m)==19
for name,item in m.items():
 p=root/name
 assert '..' not in Path(name).parts and name.startswith('campaigns/stability-')
 assert p.suffix=='.checkpoint' or p.name.endswith('.failed-stage.bin')
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 record=json.loads((p.parent.parent/'collected.json').read_text())
 assert record['scheduler']['job_state']=='F'
before=storage.footprint(root)
for name in m:(root/name).unlink()
r=dict(status='passed',manifest_sha256=control.digest(manifest),removed=m,
 verified_local_copies_before_remote_removal=True,before=before,after=storage.footprint(root),timestamp=time.time())
receipt.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['status','before','after']}))
