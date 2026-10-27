#!/usr/bin/env python3
"""Replay a floored guard failure on copies with the exact frozen kernels."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--case',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.case=a.case.resolve();a.output.mkdir(parents=True,exist_ok=False)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads((a.case/'manifest.json').read_text());v=m['arguments'];source=a.case/'frozen-source'
for name,digest in m['binding']['source_sha256'].items():assert sha(source/name)==digest
for arg,key in [('library','library_sha256'),('fast_library','fast_sha256')]:assert sha(source/Path(v[arg]).name)==m['binding'][key]
checkpoint=a.case/'checkpoint.npz';original=sha(checkpoint)
sys.path.insert(0,str(source))
from telegraph import TelegraphEngine,options
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
opts=options(v['n'],source/Path(v['library']).name,v['nu'],v['cfl'])
for key in ['width','ko_order','quadrature_width','puncture_ko','puncture_ko_radius','grid_stretch','grid_map','mapped_ko']:setattr(opts,key,v[key])
ev=PunctureChart(opts);ev.args.mass=v['mass'];ev.initial_wormhole(IsotropicWormhole(v['mass'],ev.args.initial_quadrature_order))
with np.load(checkpoint,allow_pickle=False) as z:
 q=z['q'].copy();B=z['B'].copy();rays=z['rays'].copy();comp=z['compensation'].copy();t=float(z['t'])
def engine():
 e=TelegraphEngine(ev,source/Path(v['fast_library']).name,v['kappa'],v['tau'],v['inner'],v['outer'],v['transition'],len(rays),v['ray_start'],v['inner_profile'],v['ray_interpolation'],v['ray_exit'],v['lapse_floor'])
 e.compensation(comp);return e
e=engine();dt=m['time_lattice']['dt'];r=ev.r;O=(1-r*r)/2;ref=(1+r*r)/2*r*r/(r*r+O**4)
qq=q.copy();bb=B.copy();records=[];arrays={'r':r,'q_checkpoint':q,'B_checkpoint':B}
for stage in range(1,5):
 before=qq[:,6].copy();e.apply_lapse_floor(qq)
 dq,db,speeds=e.rhs(qq,bb)
 courant=dt*np.abs(speeds)/ev.grid_jacobian[:,None]/ev.h;i,f=np.unravel_index(np.argmax(courant),courant.shape)
 lapse=ref[i]*np.exp(qq[i,6])/O[i] if 0<i<len(r)-1 else None
 record=dict(stage=stage,node=int(i),r=float(r[i]),family=['minus','plus'][f],speed=float(speeds[i,f]),jacobian=float(ev.grid_jacobian[i]),courant=float(courant[i,f]),relaxation_dt=dt/v['tau'],physical_lapse_at_max_speed=float(lapse) if lapse is not None else None,max_log_floor_correction=float(max(qq[:,6]-before)),max_metric_deviation=float(max(abs(qq[:,1]))))
 record.update(shift_at_max_speed=float(-(speeds[i,0]+speeds[i,1])/2),intrinsic_speed_at_max=float((speeds[i,1]-speeds[i,0])/2),conformal_chi_at_max_speed=float(r[i]**4/(r[i]**4+O[i]**4)*np.exp(qq[i,0])))
 arrays[f'q_stage{stage}']=qq.copy();arrays[f'B_stage{stage}']=bb.copy();arrays[f'speeds_stage{stage}']=speeds.copy();records.append(record)
 if record['courant']>.4 or record['relaxation_dt']>.4:break
 if stage<4:
  factor=dt/2 if stage<3 else dt;qq=q+factor*dq;bb=B+factor*db
e.close();e=engine();done,error=e.advance(q.copy(),B.copy(),rays.copy(),1,dt,t);e.close()
assert done==0 and error==m['failure']['message']
assert sha(checkpoint)==original
result=dict(case=str(a.case),checkpoint_sha256=original,time=t,dt=dt,lapse_floor=v['lapse_floor'],stages=records,native_replay=dict(completed_steps=done,error=error),frozen_binding=m['binding'],scope='Offline copies only; no parameter changes, production restart, or checkpoint modification. Manual stage replay applies the same native lapse floor and mapped-grid Courant criterion; it stops at the first guard violation.')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.output/'stages.npz',**arrays)
print(json.dumps({k:result[k] for k in ['time','dt','stages','native_replay']},indent=2))
