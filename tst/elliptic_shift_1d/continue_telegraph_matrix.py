#!/usr/bin/env python3
"""Continue completed telegraph parents using their own exact frozen code/binaries."""
import argparse,concurrent.futures,hashlib,json,subprocess,sys,threading,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--parents',type=Path,nargs='+',required=True);p.add_argument('--t',type=float,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2],default=2);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);a.output=a.output.resolve();lock=threading.Lock()
state=dict(status='running',pid=os.getpid(),target=a.t,cases=[])
def save():
 temp=a.output/'launch.tmp.json';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'launch.json')
def run(parent):
 parent=parent.resolve();m=json.loads((parent/'manifest.json').read_text());assert m['status']=='finished'
 assert m['time']<a.t and m['initial_data']['mass']==m['arguments']['mass']
 source=parent/'frozen-source';binding=m['binding']
 for name,sha in binding['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
 native=source/Path(m['arguments']['library']).name;fast=source/Path(m['arguments']['fast_library']).name
 assert hashlib.sha256(native.read_bytes()).hexdigest()==binding['library_sha256']
 assert hashlib.sha256(fast.read_bytes()).hexdigest()==binding['fast_sha256']
 out=a.output/parent.name;cmd=[sys.executable,str(source/'run_telegraph.py'),'--t',str(a.t),'--output',str(out),'--resume',str(parent/'checkpoint.npz'),'--library',str(native),'--fast-library',str(fast)]
 for k,v in m['arguments'].items():
  if k not in ['t','output','resume','library','fast_library']:cmd+=['--'+k.replace('_','-'),str(v)]
 record=dict(name=parent.name,parent=str(parent),command=cmd)
 with (a.output/(parent.name+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  with lock:record.update(pid=proc.pid,status='running');state['cases'].append(record);save()
  rc=proc.wait()
 with lock:record.update(returncode=rc,status='finished' if rc==0 else 'failed');save()
 return rc
save()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:codes=list(pool.map(run,a.parents))
state.update(status='finished',failed_cases=sum(c!=0 for c in codes));save();print(json.dumps(state),flush=True)
