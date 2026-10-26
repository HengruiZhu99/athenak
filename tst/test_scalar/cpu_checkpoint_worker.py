#!/usr/bin/env python3
"""Execute bounded source-bound CPU runs or exact checkpoint continuations."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from run_cpu import run

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());root=a.plan.parent;records=[]
 if plan['kind']!='cpu_validation' or plan['gpu_evolution'] or sha(a.exe)!=plan['binary_sha256']:raise RuntimeError('Exact preserved CPU validation executable is required')
 for name,digest in plan['source_sha256'].items():
  if sha(root/'source/src/test_scalar'/name)!=digest:raise RuntimeError('Declared scalar source mismatch')
 os.environ['SCALAR_CPU_THREADS']=str(plan['threads']);os.environ['SCALAR_CPU_PROC_BIND']='spread';os.environ['SCALAR_CPU_PLACES']='cores';deadline=float(os.environ['SCALAR_JOB_STARTED'])+(int(os.environ.get('SCALAR_ALLOCATION_WALL_SECONDS','1800'))-300)
 for case in plan['cases']:
  remaining=deadline-time.time()-120
  if remaining<120:break
  label=case['label'];action=case.get('action','run');out=a.output/label
  if action=='resume':
   parent=Path(case['parent'])
   if (parent/'complete.txt').exists():
    records.append(dict(label=label,action=action,skipped=True,complete=True,parent=str(parent),reason='Parent already complete; no duplicate continuation launched'))
    (a.output/'cpu-checkpoint-worker-receipt.json').write_text(json.dumps(dict(records=records,plan_sha256=sha(a.plan),all_cases_complete=len(records)==len(plan['cases']) and all(r['complete'] for r in records)),indent=2)+'\n');continue
   if not (parent/'checkpointed.txt').exists():raise RuntimeError('Parent has no confirmed wall checkpoint')
   if out.exists():raise RuntimeError('Existing continuation is immutable; inspect its receipt')
   subprocess.run([sys.executable,str(Path(__file__).with_name('resume_cpu.py')),'--exe',str(a.exe),'--parent',str(parent),'--output',str(out),'--tlim',str(case['tlim']),'--wall-seconds',str(remaining)],check=True)
   receipt=out/'restart-receipt.json'
  elif action=='run':
   out=run(a.exe,a.output,label,case['arguments']+['--wall-seconds',str(remaining)]);receipt=out/'receipt.json'
  else:raise RuntimeError('Unknown CPU validation action')
  complete=(out/'complete.txt').exists();records.append(dict(label=label,action=action,complete=complete,checkpointed=(out/'checkpointed.txt').exists(),receipt_sha256=sha(receipt)))
  (a.output/'cpu-checkpoint-worker-receipt.json').write_text(json.dumps(dict(records=records,plan_sha256=sha(a.plan),binary_sha256=sha(a.exe),all_cases_complete=len(records)==len(plan['cases']) and all(r['complete'] for r in records)),indent=2)+'\n')
  if not complete:break
if __name__=='__main__':main()
