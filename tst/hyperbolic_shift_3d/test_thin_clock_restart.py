#!/usr/bin/env python3
"""Isolate decimal endpoint clipping from matched-clock sparse restart parity."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['control','sparse','mpiexec','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);records=[];deadline=time.monotonic()+410

def run(name,end,dt,sparse=False,restart='-'):
 binary=a.sparse if sparse else a.control;prefix=out/name
 cmd=[str(a.mpiexec.resolve()),'-n','2' if sparse else '1',str(binary.resolve()),'12',str(prefix),'oblique',str(end),str(dt),'-',str(restart),'120',str(end),'.001','6','12','2']
 env=dict(os.environ,HYPERBOLIC_TRACE_LAYERS='1',HYPERBOLIC_TRACE_SHIFTED='1',HYPERBOLIC_TRACE_KAPPA='1',HYPERBOLIC_TRACE_NORMAL_RATIO='.25')
 if sparse:env['HYPERBOLIC_SPARSE_POISON']='1'
 start=time.monotonic();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(150,deadline-start))
 prefix.with_suffix('.log').write_text(r.stdout+r.stderr)
 record=dict(name=name,command=cmd,returncode=r.returncode,seconds=time.monotonic()-start);records.append(record)
 (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 assert r.returncode==0,(name,r.stdout,r.stderr)
 report=json.loads(prefix.with_suffix('.report.json').read_text());assert report['status']=='completed' and report['t']==end
 assert report['checkpoint_admitted'] and report['lapse_floor_node_interventions']==0
 assert max(report['max_null_surface'],report['max_null_ghost'])<1e-12
 cp=prefix.with_suffix('.checkpoint');record['checkpoint_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
 (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n');print(name,'completed',record['seconds'],flush=True);return cp
pairs={}
for tag,dt,full,part in [('decimal',.00025,.005,.0025),('dyadic',1/4096,20/4096,10/4096)]:
 direct=run(tag+'-full',full,dt);half=run(tag+'-part',part,dt,sparse=True);resumed=run(tag+'-resume',full,dt,sparse=True,restart=half)
 pairs[tag]=dict(full_sha256=hashlib.sha256(direct.read_bytes()).hexdigest(),resumed_sha256=hashlib.sha256(resumed.read_bytes()).hexdigest(),bitwise_equal=direct.read_bytes()==resumed.read_bytes())
 (out/'comparison.json').write_text(json.dumps(pairs,indent=2)+'\n')
assert pairs['dyadic']['bitwise_equal'],'Matched dyadic-clock restart changed the state'
print(json.dumps(pairs,indent=2))
