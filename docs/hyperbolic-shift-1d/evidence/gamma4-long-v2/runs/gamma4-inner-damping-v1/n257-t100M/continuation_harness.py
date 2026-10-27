"""Continue a completed puncture checkpoint using its verified frozen code.

No numerical input overrides. The target must lie on the parent's time lattice.
Failed/bounded-stop trajectories are never restarted by this tool.
"""
import argparse,hashlib,importlib,inspect,json,shutil,sys,time
from pathlib import Path
import numpy as np

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--t',type=float,required=True);p.add_argument('--wall-seconds',type=float,default=420.)
 p.add_argument('--report-dt',type=float,default=.025);p.add_argument('--profile-dt',type=float,default=.1);a=p.parse_args()
 parent=a.parent.resolve();mp=parent/'manifest.json';m=json.loads(mp.read_text())
 if m['status']!='completed' or not m['arguments'].get('puncture'):raise ValueError('Need a completed puncture parent')
 if m.get('stage_stats',{}).get('min_stage_shift_lapse_gap_times_g',1.)<=0:raise ValueError('Parent has lapse-shift speed coincidence evidence')
 for name in ['profiles.npz','history.json']:
  if m.get('checkpoint_hashes',{}).get(name)!=digest(parent/name):raise ValueError('Missing or mismatched checkpoint hash: '+name)
 if 'continuation_harness_sha256' in m and digest(parent/'continuation_harness.py')!=m['continuation_harness_sha256']:raise ValueError('Parent continuation harness mismatch')
 for rel,sha in m['sources'].items():
  if digest(parent/'source'/rel)!=sha:raise ValueError('Source hash mismatch: '+rel)
 libname=Path(m['arguments']['library']).name
 if digest(parent/'source'/libname)!=m['library_sha256']:raise ValueError('Library hash mismatch')
 z=np.load(parent/'profiles.npz');dt=m['dt'];start_t=m['t'];start_step=m['last_accepted_step']
 if float(z['last_time'])!=start_t or abs(start_step*dt-start_t)>1e-12:raise ValueError('Parent time mismatch')
 if not np.isfinite([a.t,a.wall_seconds,a.report_dt,a.profile_dt]).all() or a.t<=start_t or min(a.wall_seconds,a.report_dt,a.profile_dt)<=0:raise ValueError('Invalid continuation interval')
 steps=int(round(a.t/dt))
 if abs(steps*dt-a.t)>1e-12:raise ValueError('Target must lie on the inherited time lattice')
 q=z['last'].copy();comp=z['compensation'].copy()
 if not np.isfinite(q).all() or not np.isfinite(comp).all() or q.shape!=comp.shape or q.shape!=(m['arguments']['n'],10):raise ValueError('Invalid checkpoint arrays')
 a.output.mkdir(parents=True,exist_ok=False);frozen=a.output.resolve()/'source';shutil.copytree(parent/'source',frozen)
 harness=a.output/'continuation_harness.py';shutil.copy2(__file__,harness)
 ancestry={name:digest(parent/name) for name in ['manifest.json','profiles.npz','history.json']}
 args=m['arguments'].copy();args.update(t=a.t,output=str(a.output),wall_seconds=a.wall_seconds,report_dt=a.report_dt)
 record=dict(m,arguments=args,status='running',parent=dict(path=str(parent),hashes=ancestry,t=start_t),
             continuation_harness_sha256=digest(harness),command=sys.argv,
             scope='Verified completed-checkpoint continuation with unchanged frozen numerical code and timestep')
 atomic(a.output/'manifest.json',record)
 sys.path.insert(0,str(frozen/'tst/hyperbolic_shift_1d'))
 module=importlib.import_module('puncture');cls=module.PunctureEvolution
 if Path(module.__file__).resolve().parent!=frozen/'tst/hyperbolic_shift_1d':raise RuntimeError('Wrong source import')
 kwargs={k:v for k,v in args.items() if k in inspect.signature(cls).parameters and k not in ['n','library']}
 e=cls(args['n'],frozen/libname,**kwargs);e.stats=m['stage_stats'].copy();e.compensation=comp
 if not np.array_equal(e.r,z['r']):raise ValueError('Reconstructed grid mismatch')
 history=json.loads((parent/'history.json').read_text());profiles=[q.copy()];times=[start_t]
 cadence=max(1,int(a.report_dt/dt));profile_steps=max(1,int(a.profile_dt/dt));next_profile=start_step+profile_steps
 t=start_t;step=start_step;error=None;begin=time.monotonic();status='running'
 with (a.output/'history.jsonl').open('w') as stream:
  for row in history:stream.write(json.dumps(row)+'\n')
  stream.flush()
  try:
   for step in range(start_step+1,steps+1):
    q=e.step(q,dt);t=step*dt
    if step%cadence==0 or step==steps:
     row=e.report(q,t);history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
     atomic(a.output/'progress.json',dict(t=t,step=step,dt=dt,steps_requested=steps,wall_seconds=time.monotonic()-begin,latest=row))
     if step>=next_profile or step==steps:profiles.append(q.copy());times.append(t);next_profile=step+profile_steps
     if row['max_relative_null']>5e-14:raise FloatingPointError('Nullness guard')
     if row['pole']>1e-2:raise FloatingPointError('Scri numerator guard')
     if row['max_speed']*dt/e.h>.4:raise FloatingPointError('CFL guard')
     if row.get('min_stage_shift_lapse_gap_times_g',row.get('shift_lapse_gap_times_g_min',1.))<=0:
      raise FloatingPointError('Lapse-shift speed coincidence guard')
     if abs(row.get('apparent_horizon_mass',0)/(1/3)-1)>.05:raise FloatingPointError('Exploratory 5 percent horizon-mass guard')
     if time.monotonic()-begin>a.wall_seconds:raise TimeoutError('Bounded walltime')
   status='completed'
  except (FloatingPointError,ValueError,TimeoutError) as ex:
   status='bounded_stop' if isinstance(ex,TimeoutError) else 'failed';error=str(ex)
 np.savez_compressed(a.output/'profiles.npz',r=e.r,t=np.array(times),q=np.array(profiles),last=q,last_time=t,compensation=e.compensation)
 if hasattr(e,'failure_state'):np.savez_compressed(a.output/'failed-stage.npz',r=e.r,q=e.failure_state,last_accepted_time=t,attempted_step=step)
 if hasattr(e,'speed_crossing_state'):np.savez_compressed(a.output/'speed-crossing-state.npz',**e.speed_crossing_state)
 record.update(status=status,t=t,last_accepted_step=int(round(t/dt)),attempted_step=step,steps_requested=steps,
               error=error,wall_seconds=time.monotonic()-begin,stage_stats=e.stats,profile_scope='Current segment only; histories include verified parent history')
 atomic(a.output/'history.json',history)
 record['checkpoint_hashes']={name:digest(a.output/name) for name in ['profiles.npz','history.json']}
 if (a.output/'speed-crossing-state.npz').exists():record['checkpoint_hashes']['speed-crossing-state.npz']=digest(a.output/'speed-crossing-state.npz')
 atomic(a.output/'manifest.json',record)
 print(json.dumps({k:record[k] for k in ['status','t','error','wall_seconds','stage_stats']}),flush=True)

if __name__=='__main__':main()
