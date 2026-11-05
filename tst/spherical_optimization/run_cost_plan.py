#!/usr/bin/env python3
"""Controlled serial repeats from explicit immutable-run configurations."""
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--specification',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repeats',type=int,default=3);a=ap.parse_args()
 if a.repeats<2:raise ValueError('at least two controlled repeats required')
 spec=json.loads(a.specification.read_text());cases=spec['cases'];names=[c['name'] for c in cases]
 if not cases or len(set(names))!=len(names) or any(not name.replace('-','').replace('_','').isalnum() for name in names):raise ValueError('nonempty unique simple case names required')
 root=Path(__file__).resolve().parents[2];execution_files=[root/'tst/spherical_optimization'/name for name in ('run.py','evolution.py','kernel.py','physical_constraints.py')]+[root/'tst/elliptic_shift_1d/numerics.py']
 def identities():return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in execution_files}
 sources=identities();binary=hashlib.sha256(Path(a.library).read_bytes()).hexdigest();configured=[];inputs={}
 for case in cases:
  p=Path(case['reference_run'])/'manifest.json';inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();cfg=json.loads(p.read_text())['config'];cfg=dict(cfg,**case.get('overrides',{}));cfg['library']=a.library
  if cfg['duration']!=10 or cfg['floors'] or cfg['probes'] or cfg['perturbation']!='none':raise ValueError('cost controls require 10M, no floors/probes and no pulse')
  if cfg['radius']!=20 or cfg['mode']!=1:raise ValueError('cost controls must preserve central20M scale')
  configured.append(dict(case,config=cfg))
 a.output.mkdir(parents=True,exist_ok=False);env=os.environ.copy();thread_env={key:'1' for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']};env.update(thread_env);plan=[]
 for repeat in range(a.repeats):
  offset=repeat%len(configured)
  for case in configured[offset:]+configured[:offset]:
   name=f"{case['name']}-repeat{repeat+1}";cfg=dict(case['config'],output=str(a.output/name));command=[sys.executable,str(Path(__file__).with_name('run.py'))]
   for key,value in cfg.items():
    flag='--'+key.replace('_','-')
    if isinstance(value,bool):
     if value:command.append(flag)
    elif value is not None:command.extend([flag,str(value)])
   plan.append(dict(name=name,case=case['name'],family=case['family'],n=cfg['n'],repeat=repeat+1,configuration=cfg,command=command))
 manifest=dict(specification=spec,specification_sha256=hashlib.sha256(a.specification.read_bytes()).hexdigest(),inputs=inputs,plan=plan,platform=platform.platform(),thread_environment=thread_env,binary_sha256=binary,execution_sources=sources,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Sequential local CPU measurements with rotated order and identical diagnostic policy. No exclusivity claim for unrelated OS activity. Accuracy, boundary and perturbation qualification are separate.')
 (a.output/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n');started=time.perf_counter();rows=[]
 for item in plan:
  if identities()!=sources or hashlib.sha256(Path(a.library).read_bytes()).hexdigest()!=binary:raise RuntimeError('execution source or binary changed during timing batch')
  print('START '+item['name'],flush=True);begin=time.perf_counter()
  with (a.output/(item['name']+'.log')).open('x') as log:process=subprocess.run(item['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
  elapsed=time.perf_counter()-begin;p=a.output/item['name']/'result.json';result=json.loads(p.read_text()) if p.exists() else None
  row=dict(name=item['name'],returncode=process.returncode,process_seconds=elapsed,status=result['status'] if result else 'process_failed',stepping_seconds=result['stepping_seconds'] if result else None);rows.append(row)
  (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 (a.output/'completion.json').write_text(json.dumps(dict(runs=len(plan),total_wall_seconds=time.perf_counter()-started,results=rows),indent=2)+'\n')
if __name__=='__main__':main()
