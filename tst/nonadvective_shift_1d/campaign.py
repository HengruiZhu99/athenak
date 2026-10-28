"""Sequential immutable experiment; reserve the main thread's evolution slots.

Only launches when a main-thread series/queue has actually become terminal.
A transient child transition never releases a reserved slot. No retries.
"""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def command(pid):return subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout.strip()
def main_reservations(root):
 records=[]
 for name in ['gamma4-duration-series-v1/n513/series.json','gamma4-duration-series-v1/n1025/series.json','gamma4-time-100M-queue-v1/queue.json']:
  p=root/name;s=json.loads(p.read_text());terminal=s['status'] in ['completed','failed','bounded_stop','stopped','interrupted']
  if name.endswith('queue.json') and s['status']=='waiting':continue # the waiting queue shares its spatial gate's slot
  if not terminal:
   actual=command(s['pid']);tail=' '.join(s['command'][1:]) if s['command'][0].endswith('python3') else ' '.join(s['command'])
   # Missing/changed coordinator is an unresolved reservation, never permission
   # to exceed capacity or restart the main thread's work.
   records.append(dict(path=str(p),status=s['status'],pid=s['pid'],command_verified=tail in actual))
 return records

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--main-output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[2];a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'source';frozen.mkdir()
 sources=[]
 for folder in ['tst/nonadvective_shift_1d','tst/hyperbolic_shift_1d','tst/elliptic_shift_1d']:
  sources+=list((root/folder).glob('*.py'))
 sources+=list((root/'src/z4c/hyperboloidal').glob('*.hpp'))+[root/'tst/elliptic_shift_1d/spherical_api.cpp',root/'inputs/nonadvective_shift_1d/protocol-v1.json']
 hashes={}
 for src in sources:
  rel=src.relative_to(root);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);hashes[str(rel)]=digest(dest)
 shutil.copy2(a.library,frozen/'libspherical_z4c.dylib');hashes['libspherical_z4c.dylib']=digest(frozen/'libspherical_z4c.dylib')
 state=dict(status='waiting_for_slot',pid=os.getpid(),command=sys.argv,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_hashes=hashes,finished=[],active_child=None,main_output=str(a.main_output))
 atomic(a.output/'campaign.json',state)
 def run(name,n,target,eta,sigma,puncture,parent=None):
  while True:
   reservations=main_reservations(a.main_output);state['main_reservations']=reservations
   if len(reservations)<2:break
   state.update(status='waiting_for_slot',next_case=name);atomic(a.output/'campaign.json',state);time.sleep(30)
  # Verify there are no unrelated numerical children using the newly free slot.
  ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
  children=[line for line in ps.splitlines() if ('frozen_continuation.py --parent' in line or 'nonadvective_shift_1d/run_segment.py --campaign' in line) and '/bin/zsh -c' not in line]
  if len(children)>=2:raise RuntimeError('Two actual evolution children still occupy slots')
  out=a.output/name
  cmd=[sys.executable,str(frozen/'tst/nonadvective_shift_1d/run_segment.py'),'--campaign',str(a.output),'--output',str(out),'--n',str(n),'--target',str(target),'--eta',str(eta),'--sigma',str(sigma)]
  if puncture:cmd+=['--puncture']
  if parent:cmd+=['--parent',str(parent)]
  with (a.output/(name+'.log')).open('w') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   state.update(status='running',active_child=dict(pid=child.pid,command=cmd,path=str(out)));atomic(a.output/'campaign.json',state);code=child.wait()
  m=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else dict(status='launch_failed',t=0,error='No manifest')
  state['finished'].append(dict(name=name,exit_code=code,status=m['status'],t=m.get('t'),error=m.get('error')));state['active_child']=None;atomic(a.output/'campaign.json',state)
  if code or m['status']!='completed':return None
  return out
 try:
  for label,eta,sigma in [('direct',0,0),('adaptive',1,.1)]:
   qualified=True
   for n in [65,129]:
    first=run(f'{label}-pulse-n{n}-short',n,.1,eta,sigma,False)
    if not first:qualified=False;break
    final=run(f'{label}-pulse-n{n}-t1',n,1.,eta,sigma,False,first)
    if not final:qualified=False;break
   # Preserve an initial puncture control even after a pulse failure, but never
   # promote a failed regular-center configuration to a long trajectory.
   parent=run(f'{label}-puncture-n257-short',257,.2,eta,sigma,True)
   if not parent or not qualified:continue
   target=.2
   for segment in range(1,68):
    target=min(33.3333984375,.2+.5*segment)
    parent=run(f'{label}-n257-segment-{segment:04d}',257,target,eta,sigma,True,parent)
    if not parent or target>=33.3333984375:break
  state['status']='completed_campaign'
 except Exception as ex:state.update(status='controller_failed',error=str(ex));raise
 finally:atomic(a.output/'campaign.json',state)
if __name__=='__main__':main()
