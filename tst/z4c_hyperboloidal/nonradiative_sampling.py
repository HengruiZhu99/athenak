#!/usr/bin/env python3
"""Replay exact analytic values through archived native surface/donor rows.

The A normal-compatibility solve is not an interpolation operator for arbitrary
prescribed coordinate gauges. This experiment measures the actual donor rows
without that solve, and is labelled as derivative reconstruction, not A replay.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import nonradiative_analytic as fixture
from scri_reader import load,reconstruct_sampled_jets
from scri_geometry import endpoint_jets,spacetime_metric,spacetime_derivatives,ricci
from scri_frame import FrameTransport,generator_geometry
from scri_angular import spin_basis,weighted_projection

def fields(t,x,family,parameter,mass,height=1):
 # Split exact endpoint from interior before evaluating the gauge-wave log.
 out=np.zeros((len(x),31));mask=abs(np.linalg.norm(x,axis=1)-1)<2e-14
 for subset in (mask,~mask):
  if not np.any(subset):continue
  xx=x[subset];g=fixture.compact(t,xx,family,parameter=parameter,mass=mass,height=height)[0]
  gamma=g[:,1:,1:]
  minimum=float(np.min(np.linalg.eigvalsh(gamma)))
  if not np.isfinite(gamma).all() or minimum<=0:raise ValueError(f'nonspacelike analytic slice; minimum spatial eigenvalue {minimum}')
  chi=np.linalg.det(gamma)**(-1/3)
  gt=gamma*chi[:,None,None];alpha=np.sqrt(-1/np.linalg.inv(g)[:,0,0]);beta=np.linalg.solve(gamma,g[:,0,1:,None])[...,0]
  v=np.zeros((len(xx),31));v[:,0]=chi
  for k,(i,j) in enumerate(((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):v[:,k+1]=gt[:,i,j]-float(i==j)
  v[:,18]=alpha-(1+np.sum(xx*xx,axis=1))/2;v[:,22:25]=beta+xx;out[subset]=v
 return out

def replay(parts,t,family,parameter,mass,height=1):
 pieces=[];coords=[];weights=[];ids=[]
 for state,epoch in parts:
  synthetic={}
  for support in ('bulk','edge'):
   x=epoch[support+'_geometry'][:,3:6]
   synthetic[support+'_state']=fields(t,x,family,parameter,mass,height)
  synthetic['sampled_residual_jets']=reconstruct_sampled_jets(synthetic,epoch)
  synthetic['normal_lift_jets']=np.zeros((len(epoch['surface_ids']),3,13))
  x=epoch['surface_geometry'][:,:3]
  pieces.append(spacetime_metric(endpoint_jets(synthetic,epoch),x));coords.append(x)
  weights.append(epoch['surface_weights']);ids.append(epoch['surface_ids'][:,[0,2,3]])
 ids=np.concatenate(ids);order=np.lexsort(ids.T[::-1])
 return np.concatenate(pieces)[order],np.concatenate(coords)[order],np.concatenate(weights)[order]

def run(layout,output,family='schwarzschild',parameter=0,mass=1/3,at=.2,dt=.002,lmax=8,height=1):
 output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 paths=sorted(layout.glob('*.00000000.r????????.scri'));assert paths
 parts=[load(p) for p in paths];times=at+dt*np.arange(-3,4);series=[]
 for t in times:
  g,x,w=replay(parts,t,family,parameter,mass,height);series.append(g)
 metric,dg,ddg=spacetime_derivatives(np.array(series),times,3)
 exact=fixture.compact(at,x,family,parameter=parameter,mass=mass,height=height)
 R,G,inv=ricci(metric,dg,ddg);flow,source,nt,H=generator_geometry(x,inv,G)
 T=H-metric*(source*nt)[:,None,None]
 transport=FrameTransport(x,w,lmax);direction=x.copy();velocity=np.zeros_like(x);accel=np.zeros_like(x)
 if family=='rotation':
  omega=parameter/mass;c=np.cos(omega*at);s=np.sin(omega*at)
  direction[:,0]=c*x[:,0]-s*x[:,1];direction[:,1]=s*x[:,0]+c*x[:,1]
  velocity[:,0]=-omega*direction[:,1];velocity[:,1]=omega*direction[:,0];accel[:,:2]=-omega**2*direction[:,:2]
 value=transport.project@np.column_stack((np.zeros(len(x)),direction,np.full(len(x),at)))
 first=transport.project@np.column_stack((np.zeros(len(x)),velocity,np.ones(len(x))))
 second=transport.project@np.column_stack((np.zeros(len(x)),accel,np.zeros(len(x))))
 obs=transport.observe(value,metric,R,G,flow,source,nt,second,first)
 spin,labels=spin_basis(obs['direction'],lmax);project=weighted_projection(spin,obs['angular_weights'])
 modes=project@obs['news'];pieces=np.array([project@obs[key] for key in ('ricci_piece','hessian_piece','gradient_piece')])
 summary={'height':height,'family':family,'parameter':parameter,'mass':mass,'at':at,'dt':dt,'nodes':len(x),'lmax':lmax,
  'analytic_frame':True,'A_normal_solve_applied':False,'native_A_evolution':False,
  'metric_jet_errors':[float(np.max(abs(a-b))) for a,b in zip((metric,dg,ddg),exact)],
  'all_mode_max':float(np.max(abs(modes))),'news_rms':float(np.sqrt(np.sum(w*abs(obs['news'])**2)/sum(w))),
  'T_max':float(np.max(abs(T))),'null_max':float(np.max(abs(np.einsum('qi,qi->q',flow*nt[:,None],-x)))),
  'frame':obs['diagnostics'],'seconds':time.monotonic()-start}
 np.savez(output/'nodes.npz',x=x,weights=w,T=T,news=obs['news'],modes=modes,pieces=pieces,ell_m=labels)
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('layout',type=Path);p.add_argument('output',type=Path);p.add_argument('--family',default='schwarzschild');p.add_argument('--parameter',type=float,default=0);p.add_argument('--at',type=float,default=.2);p.add_argument('--dt',type=float,default=.002);a=p.parse_args();run(a.layout,a.output,a.family,a.parameter,at=a.at,dt=a.dt)
