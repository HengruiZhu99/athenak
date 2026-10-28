"""Fresh N129 pulse resolution controls only when v1 never tested that grid.

Waits for the primary campaign to finish; never retries any N129 trajectory,
including a failed short one. Uses exactly the primary frozen scientific code.
"""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from campaign import main_reservations,atomic,digest
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.campaign=a.campaign.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
s=dict(status='waiting_for_primary',pid=os.getpid(),command=sys.argv,script_sha256=digest(Path(__file__)),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),finished=[],active_child=None)
atomic(a.output/'supplement.json',s)
try:
 while True:
  primary=json.loads((a.campaign/'campaign.json').read_text())
  if primary['status']=='completed_campaign' and primary['active_child'] is None:break
  if primary['status']=='controller_failed':raise RuntimeError('Primary controller failed; no automatic retry')
  time.sleep(30)
 for label,eta,sigma in [('direct',0,0),('adaptive',1,.1)]:
  already=False
  for item in primary['finished']:
   m=json.loads((a.campaign/item['name']/'manifest.json').read_text());cfg=m['configuration']
   if not cfg['puncture'] and cfg['n']==129 and cfg['reference_damping']==eta:already=True
  if already:continue
  while len(main_reservations(Path(primary['main_output'])))>=2:time.sleep(30)
  ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
  children=[line for line in ps.splitlines() if ('frozen_continuation.py --parent' in line or 'nonadvective_shift_1d/run_segment.py --campaign' in line) and '/bin/zsh -c' not in line]
  if len(children)>=2:raise RuntimeError('Two existing evolution children occupy slots')
  name=f'{label}-pulse-n129-resolution';out=a.campaign/name
  cmd=[sys.executable,str(a.campaign/'source/tst/nonadvective_shift_1d/run_segment.py'),'--campaign',str(a.campaign),'--output',str(out),'--n','129','--target','1','--eta',str(eta),'--sigma',str(sigma)]
  with (a.output/(name+'.log')).open('w') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);s.update(status='running',active_child=dict(pid=child.pid,command=cmd,path=str(out)));atomic(a.output/'supplement.json',s);code=child.wait()
  m=json.loads((out/'manifest.json').read_text());s['finished'].append(dict(name=name,exit_code=code,status=m['status'],t=m.get('t'),error=m.get('error')));s['active_child']=None;atomic(a.output/'supplement.json',s)
 s['status']='completed'
except Exception as ex:s.update(status='controller_failed',error=str(ex));raise
finally:atomic(a.output/'supplement.json',s)
