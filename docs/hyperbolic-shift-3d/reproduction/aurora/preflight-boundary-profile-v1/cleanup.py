from pathlib import Path
import hashlib,json,sys,time
root=Path('/home/hzhu/hyperbolic-core-shell-20261028')
sys.path.insert(0,str(root/'source-v23/tst/hyperbolic_shift_3d/aurora'))
import storage
m=json.loads((root/'purge-boundary-profile-copies.json').read_text())
receipt=root/'purge-boundary-profile-copies-receipt.json'
assert not receipt.exists()
assert not (root/'.hyperbolic-debug-allocation-lock').exists()
for name,item in m.items():
 p=root/name
 assert p.is_relative_to(root) and '..' not in p.parts and (p.suffix=='.checkpoint' or p.name=='preflight-boundary-profile-v1-evidence.tar.gz')
 assert p.is_file() and p.stat().st_size==item['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
for name in m:(root/name).unlink()
result=dict(status='passed',removed=m,verified_local_copies_before_remote_removal=True,after=storage.footprint(root),timestamp=time.time())
receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
