"""Finish matching-binary checks after preserving the pre-fix donor control."""
from pathlib import Path
import json,subprocess,time,hashlib,importlib.util,shutil
import numpy as np
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
B=R/'output/refined-convergence/shell-angular-candidate-v1';O=B/'checks';E=D/'shell-angular-checks'
binary=B/'attempt02/evolve';build=json.loads((B/'attempt02/build.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(binary)==next(t['binary_sha256'] for t in build['targets'] if t['name']=='evolve')
commands=[]
def run(name,cfg,restart='-',expected=0):
 prefix=O/name;assert not Path(str(prefix)+'.checkpoint').exists()
 cmd=['mpiexec','-n','1',str(binary),'12','boost','2',str(1/24000),str(prefix),str(restart),'uniform',str(cfg)]
 start=time.time()
 with (E/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 commands.append(dict(name=name,command=cmd,binary_sha256=sha(binary),returncode=p.returncode,elapsed=time.time()-start))
 (E/'corrected-commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert p.returncode==expected,(name,p.returncode)
 if expected:return prefix
 report=json.loads(Path(str(prefix)+'.report.json').read_text());assert report['status']=='completed' and report['checkpoint_admitted'] and report['steps']==2 and report['dt']==1/24000
 for suffix in ['.report.json','.history.jsonl']:shutil.copy2(str(prefix)+suffix,E/(name+suffix))
 verification=json.loads(subprocess.check_output([str(R/'output/refined-convergence/verify-checkpoint'),str(prefix)+'.checkpoint'],text=True));assert verification['identity']==report['identity'];verification['sha256']=sha(str(prefix)+'.checkpoint');(E/(name+'-verification.json')).write_text(json.dumps(verification,indent=2)+'\n');print(name,'passed',flush=True)
 return prefix
one=run('angular-fixed-mpi1-step2',O/'angular-uniform-core.cfg')
base=run('baseline-new-binary',O/'baseline-uniform-core.cfg')
assert sha(str(base)+'.checkpoint')==sha(O/'baseline-old-binary.checkpoint'),'default production checkpoint changed'
spec=importlib.util.spec_from_file_location('reader',D/'check_smr_restart.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a,at,aa,af=m.read(str(one));b,bt,ba,bf=m.read(str(O/'angular-mpi2-restart-step2'))
assert a['identity']==b['identity'] and a['config']==b['config'] and at==bt and aa==ba and a['time']==b['time']
comparison=[]
for i,(x,y) in enumerate(zip(af,bf)):
 assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
 error=float(np.max(abs(x-y)/(1+np.maximum(abs(x),abs(y))))) if x.size else 0
 comparison.append(dict(array=i,scaled_max=error,byte_equal=bool(np.array_equal(x,y))))
assert max(x['scaled_max'] for x in comparison)<1e-12
run('reject-restart-topology',O/'baseline-uniform-core.cfg',O/'angular-mpi2-step1.checkpoint',2)
assert 'checkpoint formulation identity mismatch' in (E/'reject-restart-topology.log').read_text()
result=dict(passed=True,binary_sha256=sha(binary),arrays=comparison,default_old_new_checkpoint_byte_equal=True,donor_topology_mpi1_mpi2_equal=True,identity_mismatch_rejected=True,prior_cross_binary_comparison_rejected=True,qualification='Corrected attempt02 MPI1 uninterrupted vs attempt02 MPI2 restart. Same uniform coarse N12 core isolates shell interfaces; not full SMR candidate, GPU evidence or long stability/convergence. Pre-fix attempt01 is preserved separately and never admitted as matching-binary parity.')
shutil.copy2(B/'attempt02/build.json',E/'build.json');(E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
