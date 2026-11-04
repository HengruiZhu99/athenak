"""Reproduce preserved local controls; refuses existing output records."""
from pathlib import Path
import subprocess,json,time,hashlib
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists());base=root/'output/refined-convergence/initial-native-rhs';doc=root/'docs/hyperbolic-shift-3d/reproduction/refined-convergence-v1'
manifest_path=doc/'stationary-reference-v2-build/source-manifest.json';manifest=json.loads(manifest_path.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
checked=0
for f,v in manifest['files'].items():
 if f.startswith(('src/','tst/hyperbolic_shift_3d/')):
  assert sha(root/f)==v['sha256'],f;checked+=1
body=doc/'initial-native-rhs/body-v1.inc'
assert sha(body)==json.loads((base/'build.json').read_text())['body_sha256']
if not (base/'body-v1.inc').exists():(base/'body-v1.inc').write_bytes(body.read_bytes())
expected=json.loads((doc/'reference-initial-resolution/comparison.json').read_text())
for n in (12,14,16):
 cp=root/(f'output/refined-convergence/reference-initial-resolution/n{n}.checkpoint' if n!=16 else 'output/refined-convergence/reference-v2-stationary-n16-mpi2.checkpoint')
 prefix=base/f'n{n}';record=base/f'n{n}-command.json';assert not record.exists()
 config=root/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial12_reference.cfg'
 command=['mpiexec','-n',str(2 if n==16 else 1),str(base/'reader'),str(n),str(cp),str(config),str(prefix),'reconstruct-initial']
 r=dict(command=command,checkpoint_sha256=sha(cp),analysis_binary_sha256=sha(base/'reader'),production_manifest_sha256=sha(manifest_path),matching_local_production_files=checked,started=time.time())
 record.write_text(json.dumps(r,indent=2)+'\n')
 with (base/f'n{n}.log').open('w') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 r.update(returncode=p.returncode,finished=time.time());record.write_text(json.dumps(r,indent=2)+'\n');assert p.returncode==0
 assert sha(cp)==r['checkpoint_sha256'];x=json.loads(Path(str(prefix)+'.json').read_text());row=expected['initial_rows'][str(n)]
 for key,y in [('full_H',row['H']),('full_M',row['M']),('cut03_H',row['fixed'][1]['H']),('cut03_M',row['fixed'][1]['M'])]:assert abs(x[key]-y)<1e-11,(key,x[key],y)
 r.update(checkpoint_unchanged=True,initial_norms_match=True);record.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(n=n,seconds=r['finished']-r['started'],initial_norms_match=True)),flush=True)
