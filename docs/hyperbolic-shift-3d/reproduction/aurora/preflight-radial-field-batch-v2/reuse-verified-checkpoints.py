#!/usr/bin/env python3
"""Use local full files only after matching freshly rechecked remote SHA256/size."""
from pathlib import Path,PurePosixPath
import hashlib,json,shutil
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
old=root/'output/hyperbolic-core-shell/aurora/preflight-boundary-profile-v1'
new=root/'output/hyperbolic-core-shell/aurora/preflight-radial-field-batch-v2'
def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
proof=read(old/'verification.json');assert proof['passed']
remote=read(new/'checkpoint-hashes-rechecked.json');worker=read(new/'worker-status.json');assert worker['status']=='passed'
artifacts={n:v for c in worker['cases'] for n,v in c['artifacts'].items() if n.endswith('.checkpoint')}
assert remote==artifacts and len(remote)==6
prior={a['sha256']:a for a in proof['artifacts'] if a['remote_path'].endswith('.checkpoint')}
records=[]
for name,expected in remote.items():
 assert PurePosixPath(name).parent.as_posix()=='campaigns/preflight-radial-field-batch-v2/output'
 a=prior[expected['sha256']];source=old/'output'/PurePosixPath(a['remote_path']).name
 assert source.stat().st_size==expected['bytes']==a['bytes'] and digest(source)==expected['sha256']
 target=new/'output'/PurePosixPath(name).name;assert not target.exists()
 shutil.copyfile(source,target);assert target.stat().st_size==expected['bytes'] and digest(target)==expected['sha256']
 records.append(dict(remote_path=name,local_source=str(source.relative_to(root)),**expected))
receipt=dict(passed=True,method='Local full-file copies matched to freshly rehashed remote artifacts and the new worker manifest; no reconstructed fields.',
 prior_verification_sha256=digest(old/'verification.json'),remote_hash_receipt_sha256=digest(new/'checkpoint-hashes-rechecked.json'),files=records)
p=new/'checkpoint-cache-verification.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(passed=True,full_checkpoints_verified_and_copied=len(records),network_checkpoint_bytes=0)))
