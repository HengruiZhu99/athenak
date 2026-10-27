#!/usr/bin/env python3
"""Read-only RK-stage replay of a saved failure state with its frozen kernels."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--floor-duration',type=float,default=0);a=p.parse_args()
a.case=a.case.resolve();a.output.mkdir(parents=True,exist_ok=False)
m=json.loads((a.case/'manifest.json').read_text());source=a.case/'frozen-source'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for name,digest in m['binding']['source_sha256'].items():assert sha(source/name)==digest
for arg,key in [('library','library_sha256'),('fast_library','fast_sha256')]:assert sha(source/Path(m['arguments'][arg]).name)==m['binding'][key]
sys.path.insert(0,str(source))
from telegraph import TelegraphEngine,options
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
v=m['arguments'];opts=options(v['n'],source/Path(v['library']).name,v['nu'],v['cfl'])
for key in ['width','ko_order','quadrature_width','puncture_ko','puncture_ko_radius','grid_stretch','grid_map','mapped_ko']:setattr(opts,key,v[key])
ev=PunctureChart(opts);ev.args.mass=v['mass'];ev.initial_wormhole(IsotropicWormhole(v['mass'],ev.args.initial_quadrature_order))
with np.load(a.case/'checkpoint.npz',allow_pickle=False) as z:
 q=z['q'].copy();B=z['B'].copy();rays=z['rays'].copy();comp=z['compensation'].copy();t=float(z['t']);r=z['r'].copy()
np.testing.assert_array_equal(r,ev.r)
e=TelegraphEngine(ev,source/Path(v['fast_library']).name,v['kappa'],v['tau'],v['inner'],v['outer'],v['transition'],len(rays),v['ray_start'],v['inner_profile'],v['ray_interpolation'],v['ray_exit']);e.compensation(comp)
dt=m['time_lattice']['dt'];O=(1-r*r)/2;pos=r>0
logrefa=np.log((1+r[pos]**2)/2*r[pos]**2/(r[pos]**2+O[pos]**4));logrefc=np.log(r[pos]**4/(r[pos]**4+O[pos]**4))
records=[];arrays={'r':r,'q_checkpoint':q,'B_checkpoint':B};qq=q.copy();bb=B.copy()
for stage in range(1,5):
 la=logrefa+qq[pos,6];lc=logrefc+qq[pos,0];ia=1+int(np.argmin(la));ic=1+int(np.argmin(lc))
 record=dict(stage=stage,min_log_alpha=float(min(la)),alpha_min_node=ia,alpha_min_r=float(r[ia]),min_log_chi=float(min(lc)),chi_min_node=ic,chi_min_r=float(r[ic]),max_metric_deviation=float(max(abs(qq[:,1]))),q_finite=bool(np.isfinite(qq).all()))
 arrays[f'q_stage{stage}']=qq.copy();arrays[f'B_stage{stage}']=bb.copy()
 try:
  dq,db,_=e.rhs(qq,bb);base_dq,_=e.base.rhs(qq);record['node2_base_log_lapse_rate']=float(base_dq[2,6]);record['node2_telegraph_log_lapse_rate']=float(dq[2,6]-base_dq[2,6]);arrays[f'dq_stage{stage}']=dq;arrays[f'dB_stage{stage}']=db
  idx=int(np.argmax(abs(dq[:,6])));record.update(max_abs_log_lapse_rate=float(abs(dq[idx,6])),max_rate_node=idx,max_rate_r=float(r[idx]))
 except FloatingPointError as error:record['error']=str(error);records.append(record);break
 records.append(record)
 if stage<4:
  factor=dt/2 if stage<3 else dt;qq=q+factor*dq;bb=B+factor*db
qcopy=q.copy();bcopy=B.copy();rcopy=rays.copy();done,error=e.advance(qcopy,bcopy,rcopy,1,dt,t)
result=dict(case=str(a.case),checkpoint_sha256=sha(a.case/'checkpoint.npz'),time=t,dt=dt,stages=records,native_one_step_replay=dict(completed_steps=done,error=error),scope='Offline copies only; no trajectory restarted, no floor applied, and original checkpoint unchanged.')
if a.floor_duration>0:
 from compensated_update import accumulate
 probes=[];steps=round(a.floor_duration/dt);assert abs(steps*dt-a.floor_duration)<1e-12
 eligible=(r>0)&(r<.01);refalpha=(1+r*r)/2*r*r/(r*r+O**4)
 for epsilon in [1e-10,1e-8,1e-6]:
  state=q.copy();flux=B.copy();cq=comp[:q.size].copy().reshape(q.shape);cb=comp[q.size:q.size+len(B)].copy();counts=[0];nodes=set();max_adjustment=[0.]
  threshold=np.full_like(r,-np.inf);threshold[eligible]=np.log(epsilon*O[eligible]/refalpha[eligible])
  def floor(x,c=None):
   hit=eligible&(x[:,6]<threshold);counts[0]+=int(sum(hit));nodes.update(map(int,np.flatnonzero(hit)))
   if hit.any():max_adjustment[0]=max(max_adjustment[0],float(max(threshold[hit]-x[hit,6])));x[hit,6]=threshold[hit]
   if c is not None:c[hit,6]=0
   return x
  def rate(x,b):
   floor(x);dq,db,speeds=e.rhs(x,b)
   if dt*np.max(abs(speeds))/ev.h>.4:raise FloatingPointError('Coupled-speed timestep guard')
   return dq,db
  floor(state,cq);completed=0;error=None
  try:
   for it in range(steps):
    k1,b1=rate(state,flux);k2,b2=rate(state+dt*k1/2,flux+dt*b1/2);k3,b3=rate(state+dt*k2/2,flux+dt*b2/2);k4,b4=rate(state+dt*k3,flux+dt*b3)
    state,cq=accumulate(state,cq,dt*(k1+2*k2+2*k3+k4)/6);flux,cb=accumulate(flux,cb,dt*(b1+2*b2+2*b3+b4)/6);floor(state,cq);completed=it+1
  except FloatingPointError as exc:error=str(exc)
  rec=dict(physical_lapse_floor=epsilon,duration_completed=completed*dt,steps=completed,error=error,clamped_stage_node_count=counts[0],clamped_nodes=sorted(nodes),outermost_clamped_radius=float(max(r[list(nodes)])) if nodes else None,max_log_lapse_adjustment=max_adjustment[0])
  try:
   history,beta,diag=ev.report(state,t+completed*dt);outside=r>=history['apparent_horizon_r'];w=r*r*ev.grid_jacobian;norms=np.sqrt(np.sum(w[outside,None]*diag[outside,:4]**2,axis=0)/sum(w[outside]))
   rec.update(final_relative_AH_mass_error=abs(history['apparent_horizon_mass']-v['mass'])/v['mass'],final_exterior_RMS={k:float(value) for k,value in zip(['H','M','Theta','Z'],norms)})
  except Exception as exc:rec['diagnostic_error']=str(exc)
  probes.append(rec);arrays[f'floor{epsilon:g}_q']=state;arrays[f'floor{epsilon:g}_B']=flux;print(json.dumps(rec),flush=True)
 result['scope']='Original failed trajectory and checkpoint unchanged. Offline unfloored RK replay plus the explicitly labeled short floor probes below; no production continuation.';result['floor_probes']=probes;result['floor_scope']='Isolated short controls with a physical-lapse floor on 0<r<0.01 at every RK stage and accepted state, compensated increments with compensation reset at clamped lapse nodes. AF endpoint unchanged. Same frozen RHS kernels and timestep. Passive rays omitted because they do not feed back. This finite-grid projection changes the method and is not a validated production repair or an asymptotically consistent AF-end prescription.'
e.close();assert sha(a.case/'checkpoint.npz')==result['checkpoint_sha256']
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.output/'stages.npz',**arrays)
print(json.dumps(result,indent=2))
