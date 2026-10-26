#!/usr/bin/env python3
"""Run only a predeclared, source-bound CPU matrix within one allocation."""
import argparse,hashlib,json,os,time
from pathlib import Path
from run_cpu import run

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 plan=json.loads(a.plan.read_text());root=a.plan.parent;records=[]
 if plan['gpu_evolution'] or plan['kind']!='cpu_validation':raise RuntimeError('CPU-only plan required')
 if plan.get('binary_sha256') and sha(a.exe)!=plan['binary_sha256']:raise RuntimeError('Preserved CPU executable hash mismatch')
 for name,digest in plan['source_sha256'].items():
  if sha(root/'source/src/test_scalar'/name)!=digest:raise RuntimeError('Uploaded operator differs from the declared source')
 deadline=float(os.environ['SCALAR_JOB_STARTED'])+(int(os.environ.get('SCALAR_ALLOCATION_WALL_SECONDS','1800'))-300)
 os.environ['SCALAR_CPU_THREADS']=str(plan['threads'])
 os.environ['SCALAR_CPU_PROC_BIND']='spread';os.environ['SCALAR_CPU_PLACES']='cores'
 for case in plan['cases']:
  remaining=deadline-time.time()-120
  if remaining<120:break
  d=run(a.exe,a.output,case['label'],case['arguments']+['--wall-seconds',str(remaining)])
  records.append(dict(label=case['label'],complete=(d/'complete.txt').exists(),checkpointed=(d/'checkpointed.txt').exists(),receipt_sha256=sha(d/'receipt.json')))
  out=a.output/'cpu-worker-receipt.json'
  out.write_text(json.dumps(dict(kind=plan['kind'],plan_sha256=sha(a.plan),binary_sha256=sha(a.exe),records=records,all_cases_complete=len(records)==len(plan['cases']) and all(r['complete'] for r in records)),indent=2)+'\n')
  if not records[-1]['complete']:break
if __name__=='__main__':main()
