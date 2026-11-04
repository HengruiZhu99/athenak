"""Locate remaining native boosted initial error and isolate shell-halo effects."""
from pathlib import Path
import json,subprocess,hashlib,time,shutil,math
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
B=R/'output/refined-convergence/shell-angular-candidate-v1';A=R/'output/refined-convergence/shell-angular-initial-reader';E=D/'shell-angular-initial-diagnostic';E.mkdir(exist_ok=False)
assert json.loads((D/'shell-angular-checks/verification.json').read_text())['passed']
cp=B/'checks/angular-fixed-mpi1-step2.checkpoint';cfg=B/'checks/angular-uniform-core.cfg'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
before=sha(cp);commands=[]
initial=json.loads((B/'checks/angular-fixed-mpi1-step2.history.jsonl').read_text().splitlines()[0])
for mode in ['native-initial','analytic-shell-halos']:
 prefix=A/mode;assert not prefix.with_suffix('.json').exists()
 cmd=['mpiexec','-n','1',str(A/'reader'),'12',str(cp),str(1/24000),str(cfg),'uniform',str(prefix),'.3',mode]
 start=time.time()
 with (E/(mode+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 record=dict(command=cmd,returncode=p.returncode,elapsed=time.time()-start,checkpoint_sha256=before,checkpoint_unchanged=sha(cp)==before,analysis_binary_sha256=sha(A/'reader'),config_sha256=sha(cfg));commands.append(record)
 (E/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert p.returncode==0 and record['checkpoint_unchanged']
 x=json.loads(prefix.with_suffix('.json').read_text());assert x['steps']==0 and x['t_over_M']==0
 if mode=='native-initial':
  norms=[math.sqrt(row[j]/row[0]) for row in x['integrals'] for j in [1,2]]
  expected=[initial['H'],initial['M'],initial['fixed'][1]['H'],initial['fixed'][1]['M']]
  record['native_initial_norm_relative_errors']=[abs(a-b)/max(abs(b),1e-30) for a,b in zip(norms,expected)];assert max(record['native_initial_norm_relative_errors'])<1e-9
 for f in A.glob(mode+'.*'):
  if f.suffix in ['.json','.csv']:shutil.copy2(f,E/f.name)
 (E/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(mode,'completed',flush=True)
shutil.copy2(A/'build.json',E/'build.json')
