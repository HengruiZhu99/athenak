import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path.cwd();E=R/'docs/elliptic-shift-1d/evidence';B=R/'output/elliptic-shift-1d/telegraph-scri-pchip-convergence-v1';T=R/'output/elliptic-shift-1d/telegraph-scri-pchip-time-v1';S=R/'tst/elliptic_shift_1d'
D=E/'telegraph-wrapup-v1';D.mkdir(exist_ok=False)
state=dict(pid=os.getpid(),status='waiting_for_finest',reason='User requested wrap-up after the finest grid finishes and an updated LaTeX manuscript.',actions=[])
def save():
 p=D/'status.tmp.json';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(D/'status.json')
def command(pid,needle):
 p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
 return p.returncode==0 and needle in p.stdout
save();print('Wrap-up monitor PID',os.getpid(),flush=True)
try:
 while True:
  l=json.loads((B/'launch.json').read_text());m=json.loads((B/'n4097/manifest.json').read_text())
  state['finest_time']=m['time'];save()
  if m['status']=='failed':raise RuntimeError('Finest run failed; manual assessment required')
  if l['status']=='finished':
   if l.get('failed_cases',0) or m['status']!='finished' or m['time']!=10:raise RuntimeError('Incomplete finest trajectory')
   break
  if not command(70295,str(B/'n4097')):
   time.sleep(2)
   if json.loads((B/'launch.json').read_text())['status']!='finished':raise RuntimeError('Finest worker missing before terminal matrix status')
  time.sleep(30)
 # Stop only the still-active independent control at the requested milestone.
 l=json.loads((T/'launch.json').read_text());quarter=next(c for c in l['cases'] if c['name']=='dt-quarter')
 if quarter['status']=='running':
  pid=quarter['pid'];assert pid==70889 and command(pid,str(T/'dt-quarter'))
  os.kill(pid,signal.SIGSTOP)
  assert command(l['pid'],'telegraph-scri-pchip-time-v1')
  os.kill(l['pid'],signal.SIGSTOP)
  q=T/'dt-quarter';mp=q/'manifest.json';cp=q/'checkpoint.npz'
  (D/'quarter-manifest-at-pause.json').write_bytes(mp.read_bytes())
  with np.load(cp,allow_pickle=False) as z:checkpoint_time=float(z['t']);checkpoint_step=int(z['step'])
  state['quarter_control']=dict(status='paused_by_user_stopping_scope',pid=pid,dispatcher_pid=l['pid'],manifest_time=json.loads(mp.read_text())['time'],checkpoint_time=checkpoint_time,checkpoint_step=checkpoint_step,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),ps=subprocess.check_output(['ps','-p',str(pid)+','+str(l['pid']),'-o','pid,state,command'],text=True))
 else:
  state['quarter_control']=dict(status=quarter['status'],pid=quarter['pid'])
 state['status']='assessing_completed_spatial_run';save();print('Finest completed; quarter control:',state['quarter_control']['status'],flush=True)
 spatial=E/'telegraph-scri-pchip-convergence-v1';rays=E/'telegraph-scri-pchip-rays-v1'
 commands=[
 [sys.executable,str(S/'analyze_telegraph.py'),'--cases',*[str(B/f'n{n}') for n in [1025,2049,4097]],'--output',str(spatial)],
 [sys.executable,str(S/'assess_telegraph.py'),'--review',str(spatial/'review.json'),'--output',str(spatial/'acceptance.json')],
 [sys.executable,str(S/'analyze_telegraph_ray_convergence.py'),'--cases',*[str(B/f'n{n}') for n in [1025,2049,4097]],'--output',str(rays)]]
 if state['quarter_control']['status']=='finished':
  commands.append([sys.executable,str(S/'analyze_telegraph_time.py'),'--baseline',str(B/'n2049'),'--half',str(T/'dt-half'),'--quarter',str(T/'dt-quarter'),'--spatial',str(B/'n4097'),'--output',str(E/'telegraph-scri-pchip-time-v1/review.json')])
 for i,cmd in enumerate(commands):
  with (D/f'analysis-{i+1}.log').open('w') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  state['actions'].append(dict(command=cmd,returncode=run.returncode));save()
  if run.returncode:raise RuntimeError('Assessment failed: '+cmd[1])
 state['status']='spatial_assessment_finished';save();print(json.loads((spatial/'acceptance.json').read_text())['spatial_status'],flush=True)
except BaseException as exc:
 state.update(status='failed',error=repr(exc));save();raise
