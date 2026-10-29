#!/usr/bin/env python3
"""Poisoned multi-rank V12 communication equivalence; unchanged discretization."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['binary','control','mpiexec','initial','preflight','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);records=[];deadline=time.monotonic()+410

def run(name,kind='oblique',ranks=2,control=False,restart='-'):
 n=16 if kind=='puncture' else 12;end=.0005 if kind=='puncture' else .002
 prefix=out/name;binary=a.control if control else a.binary
 cmd=[str(a.mpiexec.resolve()),'-n',str(ranks),str(binary.resolve()),str(n),str(prefix),kind,str(end),'.00025',str(a.initial.resolve()) if kind=='puncture' else '-',str(restart),'120','.01','.001','6','12','2']
 env=dict(os.environ,HYPERBOLIC_TRACE_LAYERS='1',HYPERBOLIC_TRACE_SHIFTED='1',HYPERBOLIC_TRACE_KAPPA='1',HYPERBOLIC_SPARSE_POISON='1')
 t=time.monotonic();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(150,deadline-t))
 prefix.with_suffix('.log').write_text(r.stdout+r.stderr);record=dict(name=name,command=cmd,returncode=r.returncode,seconds=time.monotonic()-t)
 records.append(record);(out/'results.json').write_text(json.dumps(records,indent=2)+'\n');assert r.returncode==0,(name,r.stdout,r.stderr)
 report=json.loads(prefix.with_suffix('.report.json').read_text());assert report['status']=='completed' and report['checkpoint_admitted'] and report['t']==end
 assert report['lapse_floor_node_interventions']==0
 assert max(report['max_null_surface'],report['max_null_ghost'])<5e-13
 if not control:
  assert report['sparse_poison_unused'] and report['volume_exchange']=='cached_sparse_alltoallv'
  assert all(row[6]==1 for row in report['sparse_exchange_by_rank'])
 cp=prefix.with_suffix('.checkpoint');record['checkpoint_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
 (out/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(name,'passed',record['seconds'],flush=True);return cp

two=run('two');restarted=run('restart',restart=(a.preflight/'oblique-part.checkpoint').resolve())
assert two.read_bytes()==restarted.read_bytes()==(a.preflight/'oblique-one.checkpoint').read_bytes()
control=run('puncture-control',kind='puncture',ranks=1,control=True)
candidate=run('puncture-sparse',kind='puncture')
assert control.read_bytes()==candidate.read_bytes()
(out/'verification.json').write_text(json.dumps(dict(status='passed',scope='poisoned two-rank dependency coverage, partition/restart and short puncture checkpoint equivalence; not 3M stability or convergence'),indent=2)+'\n')
