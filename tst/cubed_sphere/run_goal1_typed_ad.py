#!/usr/bin/env python3
"""Read-only typed AD checks on preserved fixtures; no evolution."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/typed-native-ad-v1';work=root/'build-goal1-minkowski-20261018/typed-native-ad-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();binary=work/'ad_probe';assert sha(binary)==json.loads((out/'build.json').read_text())['binary_sha256'];runs=[]
for label,name in[('synthetic',None),('native-stationary4','full-stationary4'),('native-pulse8','full-resume8')]:
 path=out/(label+'.jsonl');assert not path.exists(),'Changed tests need a new version.';cmd=[str(binary)];inputs={}
 if name:
  case=root/'build-goal1-minkowski-20261018/full-canonical-v3/native-admission'/(name+'-segment0');ck=case/'probe_checkpoint.bin';ref=case/'cubed_sphere_reference_before.bin';cmd+=[str(ck),str(ref),label];inputs={'checkpoint_sha256':sha(ck),'reference_sha256':sha(ref)}
 t=time.time()
 with path.open('w')as f,(out/(label+'.stderr')).open('w')as e:done=subprocess.run(cmd,stdout=f,stderr=e,timeout=90)
 runs.append({'label':label,'command':cmd,'exit_code':done.returncode,'seconds':time.time()-t,'input_payloads':inputs,'stdout_sha256':sha(path)});(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert done.returncode==0,(out/(label+'.stderr')).read_text();part=[json.loads(line)for line in path.read_text().splitlines()if line.startswith('{')];assert sum(x['kind']!='native-core-AD'for x in part)==(120 if name is None else 170)
print('typed-AD point checks complete, zero evolution')
