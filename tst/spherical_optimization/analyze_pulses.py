#!/usr/bin/env python3
"""Paired pulse response and local characteristic-direction diagnostics.

The initial pulse is mixed, and the background/time-dependent operator scatters
it. Returning signals are not automatically attributable to the boundary and
these Euclidean projector norms are not a continuum energy reflection factor.
"""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel

def pack(q,d):
 out=np.zeros(q.shape[:-1]+(26,));out[...,0]=d[...,0];out[...,1]=d[...,1];out[...,4]=d[...,2]
 out[...,6]=q[...,3];out[...,9]=q[...,4];out[...,11:14]=q[...,[5,6,7]]
 out[...,16]=d[...,8];out[...,17]=d[...,9];out[...,20]=q[...,10];out[...,23]=q[...,11]
 return out

def projectors(P):
 eigen=np.linalg.eigvals(P);groups=[]
 if max(abs(eigen.imag))>1e-8:raise ValueError('complex characteristics')
 for value in sorted(eigen.real):
  if not groups or abs(value-np.mean(groups[-1]))>1e-7*max(1.,max(abs(eigen))):groups.append([value])
  else:groups[-1].append(value)
 vectors=[];speeds=[];residual=0
 for group in groups:
  value=np.mean(group);_,sv,Vh=np.linalg.svd(P-value*np.eye(len(P)));m=len(group)
  if sum(sv<1e-8*max(1.,sv[0]))<m:raise ValueError('defective characteristic cluster')
  V=Vh[-m:].T;vectors.append(V);speeds.extend([-value]*m)
  residual=max(residual,float(np.linalg.norm(P@V-value*V)))
 basis=np.concatenate(vectors,axis=1);assert basis.shape==P.shape
 inverse=np.linalg.inv(basis);speeds=np.array(speeds)
 return basis,inverse,speeds,float(np.linalg.cond(basis)),residual

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--perturbed',type=Path,required=True);ap.add_argument('--double-amplitude',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stride',type=int,default=8);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.baseline/'manifest.json').read_text())['config'];pcfg=json.loads((a.perturbed/'manifest.json').read_text())['config']
 assert cfg['perturbation']=='none' and pcfg['perturbation']!='none'
 assert all(cfg.get(key)==pcfg.get(key) for key in ('n','duration','compact_end','radius','slicing_ratio','chi_reference_power','fixed_dt','penalty'))
 assert cfg.get('auxiliary_damping',1.)==pcfg.get('auxiliary_damping',1.)
 for key,default in [('outflow_boundary',False),('outflow_width',5),('lapse_normal_lift','hermite'),('boundary_normal_system','coupled')]:assert cfg.get(key,default)==pcfg.get(key,default)
 b=np.load(a.baseline/'probes.npz');p=np.load(a.perturbed/'probes.npz');assert np.array_equal(b['r'],p['r']) and np.array_equal(b['t_M'],p['t_M'])
 delta_q=p['q']-b['q'];delta_d=p['d']-b['d'];delta=pack(delta_q,delta_d);times=b['t_M'];r=b['r'];k=Kernel.from_config(a.library,cfg)
 samples=np.unique(np.r_[np.arange(0,len(times),a.stride),len(times)-1]);incoming=np.zeros((len(samples),len(r)));outgoing=incoming.copy();stationary=incoming.copy();speed_min=incoming.copy();speed_max=incoming.copy();condition=0;residual=0
 for it,index in enumerate(samples):
  for j,x in enumerate(r):
   if x==1:continue
   B,I,speeds,c,res=projectors(k.symbol(x,b['q'][index,j]));coef=I@delta[index,j];condition=max(condition,c);residual=max(residual,res)
   # RHS is U_t=P U_r, so coordinate transport speeds are -eig(P).
   for target,mask in [(incoming,speeds<-1e-8),(outgoing,speeds>1e-8),(stationary,abs(speeds)<=1e-8)]:
    target[it,j]=np.linalg.norm(B[:,mask]@coef[mask])
   speed_min[it,j]=min(speeds);speed_max[it,j]=max(speeds)
 linearity=None
 if a.double_amplitude:
  d=np.load(a.double_amplitude/'probes.npz');dcfg=json.loads((a.double_amplitude/'manifest.json').read_text())['config']
  assert dcfg['perturb_amplitude']==2*pcfg['perturb_amplitude'] and np.array_equal(d['t_M'],times)
  difference=(d['q']-b['q'])-2*delta_q
  linearity=dict(max_absolute_residual=float(np.max(abs(difference))),relative_L2_residual=float(np.linalg.norm(difference)/max(np.linalg.norm(2*delta_q),1e-300)))
 rows=[]
 for j,x in enumerate(r):
  rows.append(dict(r=float(x),max_abs_lapse_response=float(max(abs(delta_q[:,j,8]))),max_abs_shift_response=float(max(abs(delta_q[:,j,9]))),max_incoming_projected_norm=float(max(incoming[:,j])) if x<1 else None,max_outgoing_projected_norm=float(max(outgoing[:,j])) if x<1 else None,late_incoming_projected_norm=float(max(incoming[times[samples]>1,j])) if x<1 else None,min_coordinate_speed=float(min(speed_min[:,j])) if x<1 else None,max_coordinate_speed=float(max(speed_max[:,j])) if x<1 else None))
 out=dict(baseline=str(a.baseline),perturbed=str(a.perturbed),amplitude=pcfg['perturb_amplitude'],kind=pcfg['perturbation'],rows=rows,max_eigenspace_condition=condition,max_projector_residual=residual,amplitude_linearity=linearity,qualification='Mixed smooth compact-support pulse on evolving background; saved total fields/derivatives and local incoming/outgoing principal-space Euclidean norms. Initial inward component, bulk scattering and nonnormality prevent interpreting this as a pure boundary-reflection coefficient. No endpoint symbol or 3D/boosted stability claim.')
 (a.output/'response.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(a.output/'response.npz',r=r,t_M=times,delta_q=delta_q,delta_d=delta_d,sampled_t_M=times[samples],incoming=incoming,outgoing=outgoing,stationary=stationary,speed_min=speed_min,speed_max=speed_max)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for j,x in enumerate(r):
  axes[0,0].plot(times,delta_q[:,j,8],label=f'r={x:.3f}');axes[0,1].plot(times,delta_q[:,j,9],label=f'r={x:.3f}')
  if x<1:
   axes[1,0].semilogy(times[samples],np.maximum(incoming[:,j],1e-15),label=f'r={x:.3f}');axes[1,1].semilogy(times[samples],np.maximum(outgoing[:,j],1e-15),label=f'r={x:.3f}')
 for ax,label in zip(axes.flat,['Lapse response','Shift response','Incoming projected norm','Outgoing projected norm']):ax.set(xlabel='t/M',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=7)
 fig.savefig(a.output/'response.png',dpi=150);plt.close(fig);print(json.dumps(out));k.close()
if __name__=='__main__':main()
