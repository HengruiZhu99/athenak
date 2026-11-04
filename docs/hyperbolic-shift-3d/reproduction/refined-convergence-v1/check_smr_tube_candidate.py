"""Qualify new static-box input, live support diagnostics and native MPI restart."""
from pathlib import Path
import hashlib,json,subprocess,time,shutil,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
B=R/'output/refined-convergence/smr-tube-candidate-v1';O=B/'checks';O.mkdir(exist_ok=False)
E=D/'smr-tube-checks';E.mkdir(exist_ok=False)
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial18_reference.cfg'
original=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial18_reference.cfg'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
commands=[]
def run(name,cmd,expected=0):
 start=time.time()
 with (E/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 commands.append(dict(name=name,command=[str(x) for x in cmd],returncode=p.returncode,elapsed=time.time()-start))
 (E/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 assert p.returncode==expected,(name,p.returncode)
 print(name,'passed',round(time.time()-start,2),flush=True)
for ranks in [1,2]:
 run(f'topology-mpi{ranks}',['mpiexec','-n',str(ranks),B/'support',config,E/f'topology-mpi{ranks}.json'])
a=json.loads((E/'topology-mpi1.json').read_text());b=json.loads((E/'topology-mpi2.json').read_text());assert a==b
for name,value,extra in [('zero','0,.025,.025',''),('too-wide','.4,.025,.025',''),('arity','.075,.025',''),('trailing','.075,.025,.025,',''),('both','.075,.025,.025','\nsmr_radius=.025\n')]:
 cfg=O/(name+'.cfg');cfg.write_text(config.read_text().replace('smr_half_widths=0.075,0.025,0.025','smr_half_widths='+value)+extra)
 run('reject-'+name,['mpiexec','-n','1',B/'support',cfg,O/'unused.json','config-only'],2)
# Native candidate must leave the old spherical SMR formulation unchanged.
prefix=O/'baseline-n12';run('baseline-n12',['mpiexec','-n','2',B/'evolve','12','boost','2',str(1/24000),prefix,'-','smr',original])
old=R/'output/refined-convergence/radial18-resolution-check/n12/state.checkpoint'
assert digest(str(prefix)+'.checkpoint')==digest(old),'unchanged spherical-policy checkpoint changed'
verifier=R/'output/refined-convergence/verify-checkpoint'
def evolve(name,ranks,steps,restart='-'):
 prefix=O/name;cmd=['mpiexec','-n',str(ranks),B/'evolve','12','boost',str(steps),str(1/24000),prefix,restart,'smr',config];run(name,cmd)
 report=json.loads(Path(str(prefix)+'.report.json').read_text());assert report['status']=='completed' and report['checkpoint_admitted'] and report['steps']==steps and report['dt']==1/24000
 result=json.loads(subprocess.check_output([verifier,str(prefix)+'.checkpoint'],text=True));assert result['leaves']==400 and result['n']==12 and result['identity']==report['identity'] and ';smr-box-origin-v1=' in result['identity']
 rows=[json.loads(x) for x in Path(str(prefix)+'.history.jsonl').read_text().splitlines()]
 assert max(x['cfl'] for x in rows)<=.2+1e-12 and all(x['finest_level']==3 and x['puncture_finest_margin']>0 for x in rows)
 result['sha256']=digest(str(prefix)+'.checkpoint');(E/(name+'-verification.json')).write_text(json.dumps(result,indent=2)+'\n')
 for suffix in ['.report.json','.history.jsonl']:shutil.copy2(str(prefix)+suffix,E/(name+suffix))
 return prefix,result
one,u=evolve('boost-mpi1-step2',1,2);two,v=evolve('boost-mpi2-step2',2,2);assert u['sha256']==v['sha256'],'MPI physics parity failed'
full,w=evolve('boost-mpi2-step4',2,4);resume,z=evolve('boost-mpi1-restart-step4',1,4,str(two)+'.checkpoint')
spec=importlib.util.spec_from_file_location('reader',D/'check_smr_restart.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
a,at,aa,af=reader.read(str(full));b,bt,ba,bf=reader.read(str(resume));assert a['identity']==b['identity'] and a['config']==b['config'] and at==bt and aa==ba and a['time']==b['time'] and a['dt']==b['dt']
comparisons=[]
for i,(x,y) in enumerate(zip(af,bf)):
 assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
 err=float(np.max(abs(x-y)/(1+np.maximum(abs(x),abs(y))))) if x.size else 0
 comparisons.append(dict(array=i,scaled_max=err,byte_equal=np.array_equal(x,y)))
assert max(x['scaled_max'] for x in comparisons)<1e-12
# Wrong topology policy cannot reinterpret an admitted box checkpoint.
wrong=O/'wrong-policy.cfg';wrong.write_text(config.read_text().replace('smr_half_widths=0.075,0.025,0.025','smr_radius=.025'))
run('reject-restart-policy',['mpiexec','-n','1',B/'evolve','12','boost','4',str(1/24000),O/'rejected',str(two)+'.checkpoint','smr',wrong],2)
assert 'checkpoint formulation identity mismatch' in (E/'reject-restart-policy.log').read_text()
assert not (O/'rejected.checkpoint').exists()
shutil.copy2(B/'build.json',E/'build.json')
result=dict(passed=True,topology_rank_equal=True,leaves=400,old_spherical_policy_checkpoint_byte_equal=True,mpi1_mpi2_checkpoint_byte_equal=True,restart_comparisons=comparisons,policy_mismatch_rejected=True,config_sha256=digest(config),qualification='Local Serial MPI1/2 short exact-boost tests and restart, not GPU/long-time convergence or dynamic-AMR qualification. Support margins describe nominal finest volumes with four-cell buffer, not resolution error.')
(E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
