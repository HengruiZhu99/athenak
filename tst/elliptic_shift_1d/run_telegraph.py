#!/usr/bin/env python3
"""Full-domain local vacuum telegraph experiment; no domain transfer or projection."""
import argparse,hashlib,json,platform,shutil,sys,time
from pathlib import Path
import numpy as np
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
from telegraph import TelegraphEngine,options,window
from time_lattice import FixedStepClock

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--n',type=int,default=257);p.add_argument('--t',type=float,default=3.)
 p.add_argument('--kappa',type=float,default=.02);p.add_argument('--tau',type=float,default=.02)
 p.add_argument('--inner-profile',choices=['collar','af-regular'],default='collar')
 p.add_argument('--mass',type=float,default=1/3);p.add_argument('--nu',type=float,default=1.);p.add_argument('--cfl',type=float,default=.05)
 p.add_argument('--quadrature-width',type=int,choices=[7,9,11],default=7)
 p.add_argument('--grid-stretch',type=float,default=0.)
 p.add_argument('--grid-map',choices=['sinh','inner-c7'],default='sinh')
 p.add_argument('--puncture-ko',type=float,default=.5);p.add_argument('--puncture-ko-radius',type=float,default=.8)
 p.add_argument('--inner',type=float,default=.01);p.add_argument('--outer',type=float,default=.8);p.add_argument('--transition',type=float,default=.08)
 p.add_argument('--report-dt',type=float,default=.05);p.add_argument('--ray-start',type=float,default=1.)
 p.add_argument('--ray-seeds',type=int,default=81);p.add_argument('--resume',type=Path)
 p.add_argument('--library',type=Path,required=True);p.add_argument('--fast-library',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.n<17 or args.ray_seeds<2 or not all(np.isfinite(x) for x in [args.t,args.kappa,args.tau,args.nu,args.cfl,args.report_dt,args.ray_start,args.mass]):raise ValueError('Invalid finite inputs')
 if args.t<0 or args.nu<0 or args.kappa<0 or min(args.tau,args.cfl,args.report_dt,args.mass)<=0 or args.ray_start<0:raise ValueError('Invalid positive inputs')
 if not np.isfinite(args.puncture_ko) or args.puncture_ko<0 or not 0<args.puncture_ko_radius<=1:raise ValueError('Invalid dissipation control')
 if not 0<=args.grid_stretch<=6:raise ValueError('Grid stretch must lie in [0,6]')
 args.output.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
 sources=[*sorted(here.glob('*.py')),here/'telegraph_integrator.cpp',here/'fast_integrator.cpp']
 binding=dict(source_sha256={x.name:sha(x) for x in sources},library_sha256=sha(args.library),fast_sha256=sha(args.fast_library))
 # All imported radial modules are frozen, including the independent analysis/test tools.
 frozen=args.output/'frozen-source';frozen.mkdir()
 for f in sources:shutil.copy2(f,frozen/f.name)
 for f in [args.library,args.fast_library]:shutil.copy2(f,frozen/f.name)
 config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
 manifest=dict(arguments=config,binding=binding,status='initializing',command=sys.argv,platform=platform.platform(),
  method='free-Z4c full AF chart, spherical conformal telegraph, RK4 stage rays; no projection',
  characteristic_scope='Coupled frozen lapse-trace-flux speeds; independent Z4/elliptic-shift nonlinear stability not assumed.',
  ray_scope='Alternating minus/plus seeds in .12:.4, held until ray_start. Each RK stage uses that stage geometry, linear radial speed interpolation and its derivative. Exits terminate the ray; no reinjection.',resource_authorization='User explicitly requested controlled local telegrapher evolutions.')
 mp=args.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
 opts=options(args.n,args.library,args.nu,args.cfl)
 for key in ['quadrature_width','puncture_ko','puncture_ko_radius','grid_stretch','grid_map']:setattr(opts,key,getattr(args,key))
 ev=PunctureChart(opts);support=window(ev.r,args.inner,args.outer,args.transition,args.inner_profile)
 ev.args.mass=args.mass
 initial=IsotropicWormhole(args.mass,ev.args.initial_quadrature_order)
 q,initial_q,initial_d,initial_dd,geometry=ev.initial_wormhole(initial);B=np.zeros(args.n)
 manifest['initial_data']=dict(mass=initial.mass,quadrature_order=initial.nodes.size,radial_map_scale=initial.k)
 np.savez_compressed(args.output/'initial-data.npz',r=ev.r,q=q,q_physical=initial_q,d=initial_d,dd=initial_dd)
 rays=np.zeros((2*args.ray_seeds,2));rays[:,0]=np.repeat(np.linspace(.12,.4,args.ray_seeds),2)
 engine=TelegraphEngine(ev,args.fast_library,args.kappa,args.tau,args.inner,args.outer,args.transition,len(rays),args.ray_start,args.inner_profile)
 t=0.;step=0;previous={};lattice=None
 if args.resume:
  with np.load(args.resume,allow_pickle=False) as z:
   old=json.loads(str(z['metadata']));assert old['binding']==binding,'Restart binding mismatch'
   for k in config:
    if k not in ['output','resume','t','library','fast_library']:assert old['arguments'][k]==config[k],f'Restart input mismatch: {k}'
   q=z['q'].copy();B=z['B'].copy();rays=z['rays'].copy();engine.compensation(z['compensation'])
   t=float(z['t']);step=int(z['step']);lattice=old['time_lattice'];previous=old.get('all_RK_stages',{})
  manifest['checkpoint_parent']=dict(path=str(args.resume.resolve()),sha256=sha(args.resume))
 dt=args.cfl*ev.h;clock=FixedStepClock(dt,t,step,args.t,lattice)
 # Activate the ray packet on an exact time-lattice point, not a partial RK step.
 if abs(args.ray_start/dt-round(args.ray_start/dt))>1e-8:raise ValueError('Ray start must lie on fixed time lattice')
 start=time.perf_counter();failure=None;history=[];next_report=clock.step;report_start=t;ri=0
 def stats():
  values=engine.diagnostics()
  for key,val in previous.items():values[key]=(min if key.startswith('min_') else max)(val,values.get(key,val))
  return values
 def checkpoint():
  manifest.update(time=clock.t,steps=clock.step,time_lattice=clock.metadata(),all_RK_stages=stats(),runtime_seconds=time.perf_counter()-start)
  temporary=args.output/'checkpoint.tmp.npz'
  np.savez(temporary,q=q,B=B,rays=rays,r=ev.r,t=clock.t,step=clock.step,compensation=engine.compensation(),metadata=json.dumps(manifest))
  temporary.replace(args.output/'checkpoint.npz');mp.write_text(json.dumps(manifest,indent=2)+'\n')
 manifest['status']='running';checkpoint()
 try:
  with (args.output/'history.jsonl').open('w') as log:
   while True:
    if clock.step>=next_report or clock.complete:
     # Include final accepted state in extrema. Reporting does not change the state.
     _,_,speeds=engine.rhs(q,B)
     row,b,diag=ev.report(q,clock.t);row['step']=clock.step;row['all_RK_stages']=stats()
     row['B_max']=float(max(abs(B)));row['gauge_speed_max']=float(np.max(abs(speeds)))
     for family,label in [(0,'minus'),(1,'plus')]:
      rr=rays[family::2];active=(rr[:,0]>ev.r[0])&(rr[:,0]<ev.r[-1])
      row[f'ray_{label}_active']=int(sum(active));row[f'ray_{label}_min_logJ']=float(min(rr[active,1])) if active.any() else None
      if np.any(np.diff(rr[active,0])<-1e-12):raise FloatingPointError('Stage ray ordering lost; characteristic integration unresolved')
     history.append(row);log.write(json.dumps(row)+'\n');log.flush()
     np.savez_compressed(args.output/f'profile-{len(history):04d}.npz',r=ev.r,grid_jacobian=ev.grid_jacobian,q=q,q_physical=ev.geometry_state,B=B,beta=b,diagnostic=diag,rays=rays,gauge_speeds=speeds,t=clock.t,step=clock.step)
     checkpoint();ri+=1;next_report=max(clock.step+1,clock.report_step(report_start+ri*args.report_dt))
     if clock.complete:break
    available=clock.full_steps_available;count=max(1,min(4096,available,next_report-clock.step));h=dt if available else clock.final_delta
    # Ensure the activation step is an epoch boundary despite floating addition.
    activation=round(args.ray_start/dt)
    if clock.step<activation:count=min(count,activation-clock.step)
    done,error=engine.advance(q,B,rays,count,h,clock.t);clock.advance(done,h)
    if error:raise FloatingPointError(error)
 except Exception as exc:
  failure=dict(type=type(exc).__name__,message=str(exc),time=clock.t,step=clock.step)
 finally:
  manifest.update(status='failed' if failure else 'finished',failure=failure);checkpoint();engine.close()
  print(json.dumps({k:manifest[k] for k in ['status','time','steps','runtime_seconds','failure']}),flush=True)
 return int(failure is not None)
if __name__=='__main__':sys.exit(main())
