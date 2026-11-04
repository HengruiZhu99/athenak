"""Bounded native angular-interface test with unchanged uniform N12 core.

This avoids mistaking under-resolved puncture data for a long-run candidate;
full production uses the separately qualified SMR tube. No remote job/build.
"""
from pathlib import Path
import hashlib,json,subprocess,time,shutil,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
B=R/'output/refined-convergence/shell-angular-candidate-v1';C=B/'attempt02';O=B/'checks';O.mkdir(exist_ok=False)
E=D/'shell-angular-checks';E.mkdir(exist_ok=False)
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_angular2_reference.cfg'
text=config.read_text();key='shell_angular_subdivisions=';factors=next(x for x in text.splitlines() if x.startswith(key)).split('=',1)[1]
commands=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def run(name,cmd,expected=0):
 cmd=[str(x) for x in cmd];start=time.time()
 with (E/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 commands.append(dict(name=name,command=cmd,returncode=p.returncode,elapsed_wall=time.time()-start))
 (E/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 assert p.returncode==expected,(name,p.returncode)
 print(name,'passed',flush=True)
run('topology-mpi1',['mpiexec','-n','1',C/'support',config,E/'topology-mpi1.json'])
run('topology-mpi2',['mpiexec','-n','2',C/'support',config,E/'topology-mpi2.json'])
assert json.loads((E/'topology-mpi1.json').read_text())==json.loads((E/'topology-mpi2.json').read_text())
for name,value in [('invalid','3,'+','.join(factors.split(',')[1:])),('short','1,2'),('trailing',factors+','),('core','2,'+','.join(factors.split(',')[1:])),('scri',','.join(factors.split(',')[:-1])+',2')]:
 cfg=O/(name+'.cfg');cfg.write_text(text.replace(key+factors,key+value))
 run('reject-'+name,['mpiexec','-n','1',C/'support',cfg,O/'unused.json','config-only'],2)
# Same physical data/core/timestep; only shell angular topology differs.
uniform=text.replace('smr_levels=3','smr_levels=0').replace('smr_half_widths=0.075,0.025,0.025','')
assert 'smr_levels=0' in uniform
refined=O/'angular-uniform-core.cfg';refined.write_text(uniform)
baseline=O/'baseline-uniform-core.cfg';baseline.write_text('\n'.join(x for x in uniform.splitlines() if not x.startswith(key))+'\n')
verifier=R/'output/refined-convergence/verify-checkpoint'
def evolve(name,binary,ranks,steps,cfg,restart='-'):
 prefix=O/name
 run(name,['mpiexec','-n',ranks,binary,12,'boost',steps,1/24000,prefix,restart,'uniform',cfg])
 report=json.loads(Path(str(prefix)+'.report.json').read_text())
 assert report['status']=='completed' and report['checkpoint_admitted'] and report['steps']==steps and report['dt']==1/24000
 v=json.loads(subprocess.check_output([verifier,str(prefix)+'.checkpoint'],text=True));assert v['leaves']==64 and v['n']==12 and v['identity']==report['identity']
 rows=[json.loads(x) for x in Path(str(prefix)+'.history.jsonl').read_text().splitlines()]
 assert max(x['cfl'] for x in rows)<=.2+1e-12
 v['sha256']=sha(str(prefix)+'.checkpoint');(E/(name+'-verification.json')).write_text(json.dumps(v,indent=2)+'\n')
 for suffix in ['.report.json','.history.jsonl']:shutil.copy2(str(prefix)+suffix,E/(name+suffix))
 return prefix,rows
oldbinary=R/'output/refined-convergence/smr-tube-candidate-v1/evolve'
base,br=evolve('baseline-old-binary',oldbinary,1,2,baseline)
one,ur=evolve('angular-mpi1-step2',C/'evolve',1,2,refined)
two,tr=evolve('angular-mpi2-step1',C/'evolve',2,1,refined)
resume,rr=evolve('angular-mpi2-restart-step2',C/'evolve',2,2,refined,str(two)+'.checkpoint')
spec=importlib.util.spec_from_file_location('reader',D/'check_smr_restart.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
a,at,aa,af=reader.read(str(one));b,bt,ba,bf=reader.read(str(resume));assert a['identity']==b['identity'] and a['config']==b['config'] and at==bt and aa==ba and a['time']==b['time']
comparison=[]
for i,(x,y) in enumerate(zip(af,bf)):
 assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
 error=float(np.max(abs(x-y)/(1+np.maximum(abs(x),abs(y))))) if x.size else 0
 comparison.append(dict(array=i,scaled_max=error,byte_equal=np.array_equal(x,y)))
assert max(x['scaled_max'] for x in comparison)<1e-12
run('reject-restart-topology',['mpiexec','-n',1,C/'evolve',12,'boost',2,1/24000,O/'rejected',str(two)+'.checkpoint','uniform',baseline],2)
assert 'checkpoint formulation identity mismatch' in (E/'reject-restart-topology.log').read_text()
assert not (O/'rejected.checkpoint').exists()
result=dict(passed=True,arrays=comparison,geometry_rank_equal=True,identity_mismatch_rejected=True,config_sha256=sha(config),qualification='Native Serial MPI1 uninterrupted vs MPI2 checkpoint/restart. Uniform coarse core held fixed for bounded angular-interface attribution; this is not the fully refined puncture candidate, a GPU gate or a long-time/convergence result.')
shutil.copy2(C/'build.json',E/'build.json');(E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
