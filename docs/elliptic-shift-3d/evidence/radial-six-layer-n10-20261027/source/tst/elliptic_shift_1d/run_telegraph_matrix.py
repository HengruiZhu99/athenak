#!/usr/bin/env python3
"""Bounded two-process local matrix from one frozen source/library snapshot."""
import argparse,concurrent.futures,hashlib,json,shutil,subprocess,sys,time,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--fast-library',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2],default=2);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);a.output=a.output.resolve();source=a.output/'matrix-source';source.mkdir()
for path in Path(__file__).resolve().parent.glob('*.py'):shutil.copy2(path,source/path.name)
for name in ['fast_integrator.cpp','telegraph_integrator.cpp']:shutil.copy2(Path(__file__).with_name(name),source/name)
for path in [a.library,a.fast_library]:shutil.copy2(path,source/path.name)
plan=json.loads(a.input.read_text());(a.output/'input.json').write_text(json.dumps(plan,indent=2)+'\n')
state=dict(status='running',pid=__import__('os').getpid(),source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()},cases=[])
lock=threading.Lock()
def save():
 temp=a.output/'launch.tmp.json';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'launch.json')
def run(case):
 args=dict(plan['common'],**case.get('arguments',{}));out=a.output/case['name'];cmd=[sys.executable,str(source/'run_telegraph.py'),'--library',str(source/a.library.name),'--fast-library',str(source/a.fast_library.name),'--output',str(out)]
 for k,v in args.items():cmd += ['--'+k.replace('_','-'),str(v)]
 record=dict(name=case['name'],command=cmd,status='starting')
 with (a.output/(case['name']+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  with lock:record.update(pid=proc.pid,status='running');state['cases'].append(record);save()
  rc=proc.wait()
 with lock:record.update(status='finished' if rc==0 else 'failed',returncode=rc);save()
 return rc
save()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:codes=list(pool.map(run,plan['cases']))
state.update(status='finished',failed_cases=sum(c!=0 for c in codes));save();print(json.dumps(state),flush=True)
