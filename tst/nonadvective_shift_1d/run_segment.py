"""Immutable bounded initial/continuation trial for the isolated candidate."""
import argparse,datetime,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from candidate import NonadvectiveEvolution

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def verify(case,campaign):
 m=json.loads((case/'manifest.json').read_text())
 if m['status']!='completed':raise ValueError('Continuation requires completed parent')
 for name,sha in m['checkpoint_hashes'].items():
  if digest(case/name)!=sha:raise ValueError('Checkpoint hash mismatch')
 if m['frozen_hashes']!=campaign['frozen_hashes']:raise ValueError('Frozen source mismatch')
 if m.get('parent'):
  for name,sha in m['parent']['hashes'].items():
   if digest(Path(m['parent']['path'])/name)!=sha:raise ValueError('Ancestry mismatch')
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path)
 p.add_argument('--n',type=int,required=True);p.add_argument('--target',type=float,required=True);p.add_argument('--eta',type=float,required=True);p.add_argument('--sigma',type=float,required=True);p.add_argument('--puncture',action='store_true');a=p.parse_args()
 campaign=json.loads((a.campaign/'campaign.json').read_text());frozen=a.campaign/'source'
 for rel,sha in campaign['frozen_hashes'].items():
  if digest(frozen/rel)!=sha:raise ValueError('Source/library changed: '+rel)
 cfg=dict(n=a.n,puncture=a.puncture,reference_damping=a.eta,reference_rate=a.sigma,gamma_coefficient=4.,ko=.5,lapse_floor=1e-10,cfl=.025)
 parent=verify(a.parent,campaign) if a.parent else None
 if parent and cfg!=parent['configuration']:raise ValueError('Numerical override rejected')
 a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
 m=dict(status='running',pid=os.getpid(),command=sys.argv,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=cfg,target=a.target,wall_seconds_limit=420,frozen_hashes=campaign['frozen_hashes'],campaign=str(a.campaign),mass=1/3)
 if parent:m['parent']=dict(path=str(a.parent),hashes={name:digest(a.parent/name) for name in ['manifest.json','history.json','profiles.npz']})
 atomic(a.output/'manifest.json',m)
 e=NonadvectiveEvolution(a.n,frozen/'libspherical_z4c.dylib',**{k:v for k,v in cfg.items() if k!='n'})
 dt=.025*e.h;target_step=round(a.target/dt)
 if abs(target_step*dt-a.target)>1e-12:raise ValueError('Unaligned target')
 history=[];profiles=[];physical=[];times=[];t=0.;step=0;error=None
 if parent:
  with np.load(a.parent/'profiles.npz') as z:
   q=z['last'].copy();e.compensation=z['compensation'].copy();t=float(z['last_time']);step=parent['last_accepted_step'];assert abs(step*dt-t)<1e-12
  e.stats=parent['stage_stats'].copy();e.base.stats=e.stats;history=json.loads((a.parent/'history.json').read_text())
 else:q=e.initial(amplitude=0.01 if not a.puncture else 0.)
 cadence=max(1,round(.025/dt));profile_cadence=max(1,round(.1/dt));start_step=step
 def record(save_profile=False):
  row=e.report(q,t)
  if not history or row['t']!=history[-1]['t']:history.append(row)
  if save_profile:
   profiles.append(q.copy());times.append(t);x=e.last;physical.append(np.c_[x['chi'],x['g'],x['q'][:,2:6],x['alpha'],x['beta'],q[:,8],q[:,9]-e.r])
  atomic(a.output/'progress.json',dict(t=t,step=step,dt=dt,wall_seconds=time.monotonic()-begin,latest=row))
  with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return row
 def guard(row):
  if row['max_relative_null']>5e-14:raise FloatingPointError('Nullness guard')
  if row['pole']>.01:raise FloatingPointError('Scri numerator guard')
  if row['max_driver_boundary_defect']>1e-10:raise FloatingPointError('Driver compatibility guard')
  if row['max_speed']*dt/e.h>.4:raise FloatingPointError('CFL guard')
  if row['min_stage_nonadvective_gap']<=0:raise FloatingPointError('Speed-separation guard')
  if a.puncture:
   if not row.get('apparent_horizon_mass') or abs(3*row['apparent_horizon_mass']-1)>.05:raise FloatingPointError('Exploratory horizon guard')
 try:
  guard(record(True))
  for step in range(start_step+1,target_step+1):
   if time.monotonic()-begin>420:raise TimeoutError('Bounded walltime')
   q=e.step(q,dt);t=step*dt
   if step%cadence==0 or step==target_step:guard(record((step-start_step)%profile_cadence==0 or step==target_step))
  status='completed'
 except (FloatingPointError,ValueError,TimeoutError) as ex:
  status='bounded_stop' if isinstance(ex,TimeoutError) else 'failed';error=str(ex)
 np.savez_compressed(a.output/'profiles.npz',r=e.r,t=np.array(times),q=np.array(profiles),physical=np.array(physical),last=q,last_time=t,compensation=e.__dict__.get('compensation',np.zeros_like(q)))
 if hasattr(e,'failure_state'):np.savez_compressed(a.output/'failed-stage.npz',r=e.r,q=e.failure_state,last_accepted_time=t,attempted_step=step)
 atomic(a.output/'history.json',history)
 m.update(status=status,t=t,dt=dt,last_accepted_step=round(t/dt),attempted_step=step,error=error,wall_seconds=time.monotonic()-begin,stage_stats=e.stats,checkpoint_hashes={name:digest(a.output/name) for name in ['profiles.npz','history.json']})
 atomic(a.output/'manifest.json',m);print(json.dumps(dict(status=status,t=t,error=error,wall_seconds=m['wall_seconds'])),flush=True)
 return 0 if status=='completed' else 1
if __name__=='__main__':sys.exit(main())
