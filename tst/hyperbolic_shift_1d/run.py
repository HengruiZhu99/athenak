"""Immutable bounded experiment; preserves last accepted state and failures."""
import argparse,json,time,hashlib,shutil,sys,subprocess
from pathlib import Path
import numpy as np
from evolution import Evolution

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 p.add_argument('--n',type=int,default=65);p.add_argument('--t',type=float,default=1.)
 p.add_argument('--cfl',type=float,default=.025);p.add_argument('--amplitude',type=float,default=.01)
 p.add_argument('--ratio',type=float,default=2.);p.add_argument('--damping',type=float,default=1.)
 p.add_argument('--restoring',type=float,default=0.)
 p.add_argument('--inner-shift',type=float,default=0.,help='Coefficient of Omega^2 sqrt(chi) in the shift driver')
 p.add_argument('--lapse-floor',type=float,default=1e-10,help='Minimum physical lapse on positive-radius nodes')
 p.add_argument('--shift-source',choices=['reference','lapse-adapted','lapse-inner'],default='reference')
 p.add_argument('--connection-driver',choices=['evolved','metric'],default='evolved')
 p.add_argument('--shift-gauge',choices=['separated','moving-puncture'],default='separated')
 p.add_argument('--gamma-coefficient',type=float,default=1.)
 p.add_argument('--shift-inner-stencil',choices=['positive','odd'],default='positive')
 p.add_argument('--inner-z4-damping',type=float,default=0.,help='Interior lapse-adjusted kappa1 coefficient; zero preserves native damping')
 p.add_argument('--grid-stretch',type=float,default=16.)
 p.add_argument('--report-dt',type=float,default=.025,
                help='Maximum coordinate-time spacing of guard checks and saved diagnostics')
 p.add_argument('--puncture',action='store_true')
 p.add_argument('--ko',type=float,default=.5);p.add_argument('--mode',choices=['hyperbolic','elliptic'],default='hyperbolic')
 p.add_argument('--wall-seconds',type=float,default=180.);a=p.parse_args()
 if a.n<17 or a.t<=0 or a.cfl<=0 or a.ratio<=0 or a.report_dt<=0 or a.inner_shift<0 or a.lapse_floor<0 or not np.isfinite([a.t,a.cfl,a.ratio,a.report_dt,a.inner_shift,a.lapse_floor]).all():raise ValueError('Invalid inputs')
 a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'source';frozen.mkdir()
 root=Path(__file__).resolve().parents[2]
 sources=list(Path(__file__).parent.glob('*.py'))+list((root/'tst/elliptic_shift_1d').glob('*.py'))+list((root/'src/z4c/hyperboloidal').glob('*.hpp'))+[root/'tst/elliptic_shift_1d/spherical_api.cpp']
 hashes={}
 for f in sources:
  rel=f.relative_to(root);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
  hashes[str(rel)]=hashlib.sha256(f.read_bytes()).hexdigest()
 shutil.copy2(a.library,frozen/a.library.name)
 manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
   sources=hashes,library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
   status='running',scope='regular-center qualification; no puncture or 3D claim',command=sys.argv)
 mp=a.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
 if a.puncture:
  if a.mode!='hyperbolic':raise ValueError('Use the preserved elliptic puncture reference')
  from puncture import PunctureEvolution
  e=PunctureEvolution(a.n,a.library,a.ratio,a.damping,a.ko,restoring=a.restoring,inner_shift=a.inner_shift,lapse_floor=a.lapse_floor,shift_source=a.shift_source,connection_driver=a.connection_driver,shift_gauge=a.shift_gauge,gamma_coefficient=a.gamma_coefficient,grid_stretch=a.grid_stretch,shift_inner_stencil=a.shift_inner_stencil,inner_z4_damping=a.inner_z4_damping);q=e.initial()
  manifest['scope']='Fresh matched full-domain wormhole trial; no convergence claim'
 else:
  e=Evolution(a.n,a.library,a.ratio,a.damping,a.ko,mode=a.mode,restoring=a.restoring,inner_shift=a.inner_shift,lapse_floor=a.lapse_floor,shift_source=a.shift_source,connection_driver=a.connection_driver,shift_gauge=a.shift_gauge,gamma_coefficient=a.gamma_coefficient,inner_z4_damping=a.inner_z4_damping);q=np.zeros((a.n,10))
 r=e.r
 # Compact C-infinity pulse, even at the center and identically zero at scri.
 z=(r-.4)/.12;use=abs(z)<1
 if not a.puncture:q[use,6]=a.amplitude*np.exp(1-1/(1-z[use]**2))
 nominal_dt=a.cfl*e.h;count=a.t/nominal_dt;nearest=int(round(count))
 # Preserve one timestep bit pattern for aligned split/uninterrupted runs.
 # Recomputing t/steps can otherwise shift dt by one ulp at a later target.
 if nearest>0 and abs(count-nearest)<=8*np.finfo(float).eps*max(1,abs(count)):
  steps=nearest;dt=nominal_dt
 else:steps=int(np.ceil(count));dt=a.t/steps
 t=0.;start=time.monotonic();history=[];profiles=[];times=[]
 report_steps=max(1,min(steps//40,int(a.report_dt/dt)))
 def save_progress():
  # Replace atomically so a monitor never sees a partially written history.
  tmp=a.output/'history.tmp';tmp.write_text(json.dumps(history,indent=2)+'\n')
  tmp.replace(a.output/'history.json')
  tmp=a.output/'progress.tmp';tmp.write_text(json.dumps(dict(t=t,step=step,
    wall_seconds=time.monotonic()-start,dt=dt,steps_requested=steps))+'\n')
  tmp.replace(a.output/'progress.json')
 error=None;step=0
 try:
  history.append(e.report(q,0.));profiles.append(q.copy());times.append(0.)
  save_progress()
  if history[-1]['max_speed']*dt/e.h>.4:raise FloatingPointError('Initial CFL guard: choose an appropriate fixed timestep or grid before launch')
  if history[-1].get('min_stage_shift_lapse_gap_times_g',history[-1].get('shift_lapse_gap_times_g_min',1.))<=0:
   raise FloatingPointError('Initial lapse-shift speed coincidence guard')
  for step in range(1,steps+1):
   q=e.step(q,dt);t=step*dt
   if step%report_steps==0 or step==steps:
    history.append(e.report(q,t));profiles.append(q.copy());times.append(t)
    save_progress()
    if history[-1]['max_relative_null']>5e-14:raise FloatingPointError('Nullness guard')
    if history[-1]['pole']>1e-2:raise FloatingPointError('Scri numerator guard')
    if history[-1]['max_speed']*dt/e.h>.4:raise FloatingPointError('CFL guard')
    if history[-1].get('min_stage_shift_lapse_gap_times_g',history[-1].get('shift_lapse_gap_times_g_min',1.))<=0:
     raise FloatingPointError('Lapse-shift speed coincidence guard')
    if a.puncture and abs(history[-1].get('apparent_horizon_mass',0)/(1/3)-1)>.05:raise FloatingPointError('Exploratory 5 percent horizon-mass guard')
    if time.monotonic()-start>a.wall_seconds:raise TimeoutError('Bounded walltime')
  manifest['status']='completed'
 except (FloatingPointError,ValueError,TimeoutError) as ex:
  manifest['status']='failed' if not isinstance(ex,TimeoutError) else 'bounded_stop';error=str(ex)
 manifest.update(t=t,last_accepted_step=int(round(t/dt)),attempted_step=step,dt=dt,steps_requested=steps,
                 error=error,wall_seconds=time.monotonic()-start,stage_stats=e.stats)
 np.savez_compressed(a.output/'profiles.npz',r=r,t=np.array(times),q=np.array(profiles),last=q,last_time=t,
                     compensation=getattr(e,'compensation',np.zeros_like(q)))
 if hasattr(e,'failure_state'):
  np.savez_compressed(a.output/'failed-stage.npz',r=r,q=e.failure_state,last_accepted_time=t,attempted_step=step)
 if hasattr(e,'speed_crossing_state'):
  np.savez_compressed(a.output/'speed-crossing-state.npz',**e.speed_crossing_state)
 (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
 manifest['checkpoint_hashes']={name:hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in ['profiles.npz','history.json']}
 if (a.output/'speed-crossing-state.npz').exists():
  manifest['checkpoint_hashes']['speed-crossing-state.npz']=hashlib.sha256((a.output/'speed-crossing-state.npz').read_bytes()).hexdigest()
 mp.write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:manifest[k] for k in ['status','t','error','wall_seconds','stage_stats']}),flush=True)

if __name__=='__main__':main()
