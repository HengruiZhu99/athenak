#!/usr/bin/env python3
"""Bounded local V12 preflight, not an interface-crossing or duration admission."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for key in ('binary','control','mpiexec','output'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
records=[];start=time.monotonic();deadline=start+410

def run(name,kind='minkowski',layers=1,kappa=0,shifted=1,ranks=1,end=.001,restart='-',control=False,expected_failure=False):
 prefix=out/name
 env=dict(os.environ,HYPERBOLIC_TRACE_LAYERS=str(layers),HYPERBOLIC_TRACE_KAPPA=str(kappa),HYPERBOLIC_TRACE_SHIFTED=str(shifted))
 cmd=[str(a.mpiexec.resolve()),'-n',str(ranks),str((a.control if control else a.binary).resolve()),'12',str(prefix),kind,str(end),'.00025','-',str(restart),'100','.01','.001','6','12','2']
 t=time.monotonic();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(120,deadline-t))
 prefix.with_suffix('.log').write_text(r.stdout+r.stderr)
 record=dict(name=name,command=cmd,environment={k:v for k,v in env.items() if k.startswith('HYPERBOLIC_TRACE_')},returncode=r.returncode,seconds=time.monotonic()-t)
 records.append(record);reportpath=prefix.with_suffix('.report.json')
 if reportpath.exists():record['report']=json.loads(reportpath.read_text())
 (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
 if expected_failure:
  assert r.returncode!=0 and 'trace configuration mismatch' in r.stdout+r.stderr
  return prefix
 assert r.returncode==0,(name,r.stdout,r.stderr)
 report=record['report'];assert report['status']=='completed' and report['t']==end and report['checkpoint_admitted']
 assert report['lapse_floor_node_interventions']==0
 assert report['max_null_surface']<5e-13 and report['max_null_ghost']<5e-13
 if kind=='minkowski':assert report['max_state_deviation']<1e-10
 cp=prefix.with_suffix('.checkpoint');record['checkpoint_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
 (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
 print(name,'passed',round(record['seconds'],2),flush=True)
 return prefix

base=run('frozen-control',kind='pulse',control=True)
legacy=run('three-zero',kind='pulse',layers=3,shifted=0)
b=base.with_suffix('.checkpoint').read_bytes();c=legacy.with_suffix('.checkpoint').read_bytes()
assert b[16:152]==c[16:152] and b[152:]==c[184:],'three-layer zero penalty changed V11 trajectory'
for layers in (1,2,3):
 for kappa in (0,1):run(f'minkowski-l{layers}-k{kappa}',layers=layers,kappa=kappa)
one=run('oblique-one',kind='oblique',kappa=1,end=.002)
two=run('oblique-two',kind='oblique',kappa=1,ranks=2,end=.002)
assert one.with_suffix('.checkpoint').read_bytes()==two.with_suffix('.checkpoint').read_bytes(),'partition trajectory differs'
part=run('oblique-part',kind='oblique',kappa=1,end=.001)
restart=run('oblique-restart',kind='oblique',kappa=1,ranks=2,end=.002,restart=part.with_suffix('.checkpoint'))
assert one.with_suffix('.checkpoint').read_bytes()==restart.with_suffix('.checkpoint').read_bytes(),'restart trajectory differs'
run('reject-penalty-change',kind='oblique',kappa=2,end=.002,restart=part.with_suffix('.checkpoint'),expected_failure=True)
(out/'verification.json').write_text(json.dumps(dict(status='passed',scope='short CPU preflight; not crossing, convergence, GPU performance or 3M puncture',seconds=time.monotonic()-start),indent=2)+'\n')
