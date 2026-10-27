import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path.cwd()
base=root/'output/elliptic-shift-1d/telegraph-scri-pchip-convergence-v1'
timing=root/'output/elliptic-shift-1d/telegraph-scri-pchip-time-v1'
evidence=root/'docs/elliptic-shift-1d/evidence'
receipt=evidence/'telegraph-main-assessment-dispatch-v1'
receipt.mkdir(exist_ok=False)
scripts=root/'tst/elliptic_shift_1d'
spatial=evidence/'telegraph-scri-pchip-convergence-v1'
temporal=evidence/'telegraph-scri-pchip-time-v1'
rays=evidence/'telegraph-scri-pchip-rays-v1'
commands=[
 [sys.executable,str(scripts/'analyze_telegraph.py'),'--cases',*[str(base/f'n{n}') for n in [1025,2049,4097]],'--output',str(spatial)],
 [sys.executable,str(scripts/'assess_telegraph.py'),'--review',str(spatial/'review.json'),'--output',str(spatial/'acceptance.json')],
 [sys.executable,str(scripts/'analyze_telegraph_ray_convergence.py'),'--cases',*[str(base/f'n{n}') for n in [1025,2049,4097]],'--output',str(rays)],
 [sys.executable,str(scripts/'analyze_telegraph_time.py'),'--baseline',str(base/'n2049'),'--half',str(timing/'dt-half'),'--quarter',str(timing/'dt-quarter'),'--spatial',str(base/'n4097'),'--output',str(temporal/'review.json')]
]
source_names=['analyze_telegraph.py','assess_telegraph.py','analyze_telegraph_ray_convergence.py','analyze_telegraph_time.py','telegraph_analysis_io.py','numerics.py']
hashes={name:hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in source_names}
state=dict(pid=os.getpid(),status='waiting',commands=commands,analysis_sha256=hashes,observations=[],completed_commands=[],scope='Wait for existing run matrices only, then analyze. Never starts or restarts an evolution, relaxes acceptance, dispatches robustness cases or declares the goal achieved.')
def save():
 p=receipt/'status.tmp.json';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(receipt/'status.json')
last=None
try:
 for path in [spatial,temporal,rays]:
  if path.exists():raise FileExistsError(path)
 save();print('Assessment waiter PID',os.getpid(),flush=True)
 while True:
  observations=[];ready=True
  for matrix in [base,timing]:
   launch=json.loads((matrix/'launch.json').read_text())
   if launch['status']!='finished':ready=False
   if launch.get('failed_cases',0):raise RuntimeError('Matrix contains failed cases: '+str(matrix))
   for case in launch['cases']:
    folder=matrix/case['name'];m=json.loads((folder/'manifest.json').read_text())
    observations.append(dict(matrix=matrix.name,case=case['name'],pid=case.get('pid'),status=m['status'],time=m['time']))
    if case['status']=='failed' or m['status']=='failed':raise RuntimeError('Evolution failed: '+str(folder))
    if case['status']=='running':
     ready=False
     ps=subprocess.run(['ps','-p',str(case['pid']),'-o','command='],text=True,capture_output=True)
     if ps.returncode!=0 or str(folder) not in ps.stdout or 'run_telegraph.py' not in ps.stdout:
      time.sleep(2)
      fresh=json.loads((matrix/'launch.json').read_text())
      if next(c for c in fresh['cases'] if c['name']==case['name'])['status']=='running':
       raise RuntimeError('Running worker identity missing; manual inspection required: '+str(folder))
    elif case['status']!='finished' or case.get('returncode')!=0:ready=False
    if m['status']!='finished' or m['time']!=10.0:ready=False
  state['observations']=observations;save()
  summary=[(o['case'],o['status'],o['time']) for o in observations]
  if summary!=last:print(summary,flush=True);last=summary
  if ready:break
  time.sleep(30)
 for name,digest in hashes.items():
  if hashlib.sha256((scripts/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Analysis changed while waiting: '+name)
 state['status']='analyzing';save()
 for index,cmd in enumerate(commands):
  print('Analysis',index+1,Path(cmd[1]).name,flush=True)
  with (receipt/f'analysis-{index+1}.log').open('w') as stream:
   done=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
  state['completed_commands'].append(dict(index=index+1,returncode=done.returncode));save()
  if done.returncode:raise RuntimeError('Analysis command failed: '+cmd[1])
 acceptance=json.loads((spatial/'acceptance.json').read_text());metrics=json.loads((temporal/'review.json').read_text())['metrics']
 state.update(status='finished',spatial_status=acceptance['spatial_status'],failed_spatial_checks=[k for k,v in acceptance['checks'].items() if not v['passed']],failed_time_metrics=[k for k,v in metrics.items() if not v['below_ten_percent']],goal_status='incomplete')
 save();print(json.dumps({k:state[k] for k in ['status','spatial_status','failed_spatial_checks','failed_time_metrics','goal_status']}),flush=True)
except BaseException as error:
 state.update(status='failed',error=repr(error));save();raise
