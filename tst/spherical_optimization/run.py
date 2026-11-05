#!/usr/bin/env python3
"""One immutable local experiment. Failures retain their last accepted state."""
import argparse,hashlib,json,subprocess,time,traceback,tarfile,platform
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',type=int,choices=(0,1),default=1);ap.add_argument('--n',type=int,default=64)
    ap.add_argument('--radius',type=float,default=20.);ap.add_argument('--compact-end',type=float,default=0.);ap.add_argument('--slicing-ratio',type=float,default=2.)
    ap.add_argument('--separated',action='store_true');ap.add_argument('--grid-map',choices=('quadratic','two-end'),default='quadratic');ap.add_argument('--grading',type=float,default=0.)
    ap.add_argument('--penalty',type=float,default=.6510416666660156);ap.add_argument('--ko',type=float,default=.5)
    ap.add_argument('--chi-reference-power',type=int,choices=(0,2,4),default=0)
    ap.add_argument('--compatible-lapse',type=float,default=0.,help='Experimental fraction of composed D1 squared in lapse residual D2, localized to the shared gauge join')
    ap.add_argument('--auxiliary-damping',type=float,default=1.)
    ap.add_argument('--outflow-boundary',action='store_true');ap.add_argument('--outflow-width',type=int,choices=(3,5,7),default=5)
    ap.add_argument('--lapse-normal-lift',choices=('hermite','second-neutral'),default='hermite')
    ap.add_argument('--boundary-normal-system',choices=('coupled','shift'),default='coupled')
    ap.add_argument('--perturbation',choices=('none','gauge','constraint'),default='none')
    ap.add_argument('--perturb-amplitude',type=float,default=1e-5);ap.add_argument('--perturb-center',type=float,default=.9);ap.add_argument('--perturb-width',type=float,default=.025)
    ap.add_argument('--probes',action='store_true',help='Save actual total fields and radial jets at fixed probe nodes each accepted step')
    ap.add_argument('--derivative-boundary',action='store_true');ap.add_argument('--reference-tail',choices=('numerical','scri','outer'),default='numerical');ap.add_argument('--reference-step',type=float,default=5e-4);ap.add_argument('--reference-jets',action='store_true');ap.add_argument('--normal-policy',choices=('sampled','analytic'),default='sampled');ap.add_argument('--algebraic-chart',action='store_true');ap.add_argument('--floors',action='store_true');ap.add_argument('--duration',type=float,default=.1)
    ap.add_argument('--cfl',type=float,default=.2);ap.add_argument('--fixed-dt',type=float)
    ap.add_argument('--experimental-timestep',action='store_true')
    args=ap.parse_args()
    if not np.isfinite(args.compatible_lapse) or not 0<=args.compatible_lapse<=1 or (args.compatible_lapse and (not args.chi_reference_power or not args.reference_jets)):raise ValueError('compatible lapse fraction needs [0,1], reference-chi chart and reference jets')
    if args.derivative_boundary and not args.algebraic_chart:raise ValueError('derivative boundary requires algebraic chart')
    if args.lapse_normal_lift!='hermite' and not args.outflow_boundary:raise ValueError('experimental lapse lift requires outgoing boundary closure')
    if args.boundary_normal_system=='shift' and (not args.outflow_boundary or args.lapse_normal_lift!='hermite'):raise ValueError('shift normal system requires outgoing boundary and unchanged lapse stencil')
    if args.outflow_boundary and (not args.algebraic_chart or not args.chi_reference_power or args.derivative_boundary or args.normal_policy!='analytic'):
        raise ValueError('outflow boundary requires reference-chi algebraic chart, analytic normal, and excludes derivative-boundary mode')
    if args.perturbation!='none' and (not args.algebraic_chart or not 0<args.perturb_center-args.perturb_width<args.perturb_center+args.perturb_width<1 or not np.isfinite(args.perturb_amplitude)):
        raise ValueError('perturbation needs algebraic chart, finite amplitude and support strictly inside domain')
    if args.output.exists():raise FileExistsError('refuse to overwrite experiment')
    if args.cfl<=0 or args.duration<=0 or args.n<16 or (args.fixed_dt is not None and args.fixed_dt<=0):raise ValueError('invalid experiment')
    if (args.cfl>.4 or args.fixed_dt) and not args.experimental_timestep:raise ValueError('explicit experimental timestep opt-in required')
    args.output.mkdir(parents=True)
    source_paths=sorted([*ROOT.glob('src/z4c/hyperboloidal/*.hpp'),*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.cpp'),Path(__file__).parent/'CMakeLists.txt',ROOT/'tst/elliptic_shift_1d/numerics.py'])
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    identity=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
      commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      sources=hashes,binary_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest())
    identity['platform']=platform.platform()
    build=Path(args.library).resolve().parent
    identity['build_metadata']={str(p.relative_to(build)):p.read_text() for p in (build/'CMakeCache.txt',build/'CMakeFiles/current_spherical.dir/flags.make') if p.exists()}
    (args.output/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    with tarfile.open(args.output/'source.tar.gz','w:gz') as archive:
        for p in source_paths:archive.add(p,arcname=str(p.relative_to(ROOT)))
    setup_start=time.perf_counter();k=Kernel.from_config(args.library,vars(args))
    e=make_evolution(k,vars(args));q=e.initial.copy()
    if args.perturbation!='none':
        z=(e.r-args.perturb_center)/args.perturb_width;bump=np.zeros_like(z);inside=abs(z)<1
        bump[inside]=np.exp(1-1/(1-z[inside]**2));pulse=args.perturb_amplitude*bump
        # Smooth compact support, no boundary projection or constraint solve.
        # The reference derivatives/sources remain those of the unperturbed data.
        slots=(6,7) if args.perturbation=='gauge' else (4,5)
        q[:,slots[0]]+=pulse;q[:,slots[1]]+=pulse
    full=lambda state:k.expand_chart(state)[0] if args.algebraic_chart else state
    if args.reference_jets:np.savez_compressed(args.output/'reference-jets.npz',d=e.reference_d,dd=e.reference_dd,refinement_difference=e.reference_jet_difference)
    if e.reference_tail_correction is not None:np.savez_compressed(args.output/'reference-tail-correction.npz',difference=e.reference_tail_correction)
    np.savez_compressed(args.output/'initial.npz',r=e.r,q=full(q),omega=e.omega,source=e.source,configuration=k.config,compact_configuration=k.compact_config)
    history=[];t=0.;steps=0;dt_min=float('inf');dt_max=0.;cmax=0.;status='failed';failure=None
    probe_index=np.unique([int(np.argmin(abs(e.r-r))) for r in (.84,.88,.92,.96,.985,1.)]);probe_times=[];probe_q=[];probe_d=[]
    def record_probes():
        if args.probes:
            probe_times.append(3*t);probe_q.append(e.last['q'][probe_index].copy());probe_d.append(e.last['d'][probe_index].copy())
    setup_seconds=time.perf_counter()-setup_start;start=time.perf_counter()
    target=args.duration/3;next_output=0;interval=target/10
    try:
        history.append(e.report(q,t));speed0=e.last['speed_over_dr'];initial_speed=e.last['speed']
        record_probes()
        fixed_limit=min(args.cfl/speed0,.5*e.dr[-1]/(12*args.penalty),.5/30)
        limiters={'characteristic':args.cfl/speed0,'scri_policy':.5*e.dr[-1]/(12*args.penalty),'telegrapher_policy':.5/30}
        active=min(limiters,key=limiters.get)
        if args.fixed_dt:fixed_limit=args.fixed_dt;active='explicit_fixed_dt'
        next_output=interval
        while t<target-1e-14:
            dt=min(fixed_limit,target-t)
            q=e.step(q,dt);t+=dt;steps+=1;dt_min=min(dt_min,dt);dt_max=max(dt_max,dt)
            record_probes()
            cmax=max(cmax,dt*e.step_speed_over_dr)
            if t>=next_output-1e-14 or t>=target-1e-14:
                history.append(e.report(q,t));next_output+=interval
                np.savez_compressed(args.output/f'profile-{len(history)-1:02d}.npz',r=e.r,q=full(q),d=e.last['d'],dd=e.last['dd'],diag=e.last['diag'],t_M=3*t)
        status='completed_unqualified'
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if hasattr(e,'attempted_state'):np.savez_compressed(args.output/'failed-stage.npz',r=e.r,q=e.attempted_state,t_last_accepted_M=3*t)
    elapsed=time.perf_counter()-start
    if args.probes:np.savez_compressed(args.output/'probes.npz',r=e.r[probe_index],indices=probe_index,t_M=np.array(probe_times),q=np.array(probe_q),d=np.array(probe_d))
    np.savez_compressed(args.output/'last-accepted.npz',r=e.r,q=full(q),independent_state=q,t_M=3*t)
    (args.output/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
    result=dict(status=status,failure=failure,time_M=3*t,steps=steps,setup_seconds=setup_seconds,stepping_seconds=elapsed,
      rhs_calls=e.calls,rhs_seconds=e.cumulative_rhs_seconds,initial_speed=locals().get('initial_speed'),
      worst_evaluated_speed=e.max_speed,worst_evaluated_speed_over_dr=e.worst_speed_over_dr,C_char_max_accepted=cmax,dt_min=dt_min if steps else None,dt_max=dt_max,
      dt_over_M=3*dt_max,min_dr_over_M=3*e.h,nominal_dt_over_min_dr=dt_max/e.h,
      steps_per_M=steps/(3*t) if t else None,active_limiter=locals().get('active'),limits=locals().get('limiters'),
      floor_calls=e.floor_calls,floor_max_r=e.floor_max_r,
      qualifications=['spherical CPU only','completion is not stability or convergence','differential D boundary compatibility not yet qualified'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
    k.close()
if __name__=='__main__':main()
