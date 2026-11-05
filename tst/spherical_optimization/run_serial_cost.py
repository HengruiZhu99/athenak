#!/usr/bin/env python3
"""Serial, rotated-order 10M cost controls; no automatic stability admission."""
import argparse,json,math,os,platform,subprocess,sys,time,hashlib
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repeats',type=int,default=3);a=ap.parse_args()
 if a.repeats<1:raise ValueError('positive repeat count required')
 a.output.mkdir(parents=True,exist_ok=False)
 cases=[('areal',128),('compact',128),('areal',256),('compact',256),('areal',512),('compact',64)]
 env=os.environ.copy();thread_env={key:'1' for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']};env.update(thread_env)
 plan=[]
 for repeat in range(a.repeats):
  order=cases[repeat:]+cases[:repeat]
  for family,n in order:
   name=f'{family}-n{n}-repeat{repeat+1}';dt=.4/((n-.5)*(1+math.sqrt(2)))
   cmd=[sys.executable,str(Path(__file__).with_name('run.py')),'--library',a.library,'--output',str(a.output/name),'--mode','1','--n',str(n),'--radius','20','--slicing-ratio','2','--compact-end','.97' if family=='compact' else '0','--separated','--algebraic-chart','--chi-reference-power','2','--normal-policy','analytic','--reference-jets','--reference-step','.000125','--reference-tail','scri','--duration','10','--fixed-dt',repr(dt),'--experimental-timestep']
   plan.append(dict(name=name,family=family,n=n,repeat=repeat+1,command=cmd))
 manifest=dict(plan=plan,platform=platform.platform(),thread_environment=thread_env,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Sequential local CPU runs; rotated case order, identical diagnostic cadence, no probes. Setup separated. No exclusivity claim for unrelated OS activity. Exterior-only comparison; boundary and late response gates still fail.')
 (a.output/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n');started=time.perf_counter()
 for item in plan:
  print('START '+item['name'],flush=True)
  with (a.output/(item['name']+'.log')).open('w') as log:
   subprocess.run(item['command'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((a.output/item['name']/'result.json').read_text())
  print(json.dumps(dict(name=item['name'],status=result['status'],time_M=result['time_M'],seconds=result['stepping_seconds'])),flush=True)
 (a.output/'completion.json').write_text(json.dumps(dict(runs=len(plan),total_wall_seconds=time.perf_counter()-started),indent=2)+'\n')
if __name__=='__main__':main()
