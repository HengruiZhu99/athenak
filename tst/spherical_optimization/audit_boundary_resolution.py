#!/usr/bin/env python3
"""Same-state boundary sensitivity; these are diagnostics, not new evolutions."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution,weights

def trace_defect(e,state):
 rate=e.rhs(state);base={key:e.last[key].copy() if isinstance(e.last[key],np.ndarray) else e.last[key] for key in e.last};values=[]
 for scale in (1e-5,3e-6,1e-6):
  eps=scale/max(1.,np.max(abs(rate)));e.rhs(state+eps*rate);plus=e.last['dependent_D'];e.rhs(state-eps*rate);minus=e.last['dependent_D'];values.append(float((plus-minus)/(2*eps)-base['auxiliary_trace_pde']))
 return dict(clock=float(base['clock']),pole=float(max(abs(base['pole']))),defects=values,dependent_D=float(base['dependent_D'])),base

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--runs',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);reports=[];inputs={};arrays={}
 for folder in a.runs:
  cfg=json.loads((folder/'manifest.json').read_text())['config'];saved=np.load(folder/'last-accepted.npz');state=saved['independent_state'];inputs[str(folder/'last-accepted.npz')]=hashlib.sha256((folder/'last-accepted.npz').read_bytes()).hexdigest();k=Kernel.from_config(a.library,cfg);report=dict(n=cfg['n'],run=str(folder),reference_steps=[],outflow_widths=[],endpoint_stencils=[])
  for factor in (1.,.5,.25):
   step=cfg['reference_step']*factor;e=make_evolution(k,dict(cfg,reference_step=step));result,_=trace_defect(e,state);report['reference_steps'].append(dict(reference_step=step,**result))
  for width in (3,5,7):
   e=make_evolution(k,dict(cfg,outflow_width=width));result,_=trace_defect(e,state);report['outflow_widths'].append(dict(width=width,**result))
  e=make_evolution(k,cfg);_,base=trace_defect(e,state);q=base['q'];r=e.r;scaled=e.scaled_values(q);values=scaled-e.scaled_initial
  R,_,_,_=k.parts(r,q,base['d'],base['dd'],e.source);normal=k.analytic_normal(q[-1],base['d'][-1],base['dd'][-1]);report['endpoint']=dict(q=q[-1].tolist(),d=base['d'][-1].tolist(),dd=base['dd'][-1].tolist(),clock_regular=float(R[-1,8]),clock_pole_normal=float(normal[8]))
  # Independent endpoint polynomial jets on the same nodal data, followed by
  # the same shared shift-normal compatibility equation. No rates are advanced.
  for width in (5,7,9,11):
   d=(weights(r[-width:],1.,1)@values[-width:]+e.scaled_d[-1])[None];dd=(weights(r[-width:],1.,2)@values[-width:]+e.scaled_dd[-1])[None]
   Q,D,DD=k.reference_chi(np.array([1.]),scaled[-1:],cfg['chi_reference_power'],d,dd);Q,D,DD=k.expand_chart(Q,D,DD);Q,D,DD=Q[0],D[0],DD[0]
   def boundary(derivative):return k.scri(Q,derivative,DD,k.analytic_normal(Q,derivative,DD),e.dr[-1],e.penalty)
   _,dep=boundary(D);pert=D.copy();pert[9]+=1;coef=(boundary(pert)[1]-dep)[0];D[9]-=(dep[0]-Q[11])/coef;_,dep=boundary(D)
   assert abs(dep[0]-Q[11])<1e-8
   if width==7:assert abs(dep[1]-base['clock'])<1e-8
   _,pole,_,_=k.parts(np.array([1.]),Q[None],D[None],DD[None],e.source[-1:]);report['endpoint_stencils'].append(dict(width=width,clock=float(dep[1]),pole=float(abs(pole).max()),dependent_D_residual=float(dep[0]-Q[11]),lapse_first=float(D[8]),lapse_second=float(DD[8])))
  tag=f'n{cfg["n"]}';mask=(r>.94)&(r<1);arrays[tag+'-r']=r[mask];residual=q-e.initial_full;arrays[tag+'-residual_over_distance']=(residual[mask]-residual[-1])/(1-r[mask,None]);arrays[tag+'-q']=q[mask]
  reports.append(report);k.close();print(json.dumps(report),flush=True)
 np.savez_compressed(a.output/'boundary_profiles.npz',**arrays)
 (a.output/'audit.json').write_text(json.dumps(dict(reports=reports,inputs=inputs,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Same-state sensitivity diagnostics. Altered reference preparation, extrapolation width or endpoint diagnostic stencil does not establish a stable evolved policy. No boundary clock or constraint is projected to zero.'),indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');axes=axes.ravel()
 for report in reports:
  n=report['n'];tag=f'n{n}';x=1-arrays[tag+'-r'];v=arrays[tag+'-residual_over_distance']
  for ax,f in zip(axes,[0,5,8,9]):ax.semilogx(x,v[:,f],'o-',ms=3,label=f'N{n}')
 for ax,name in zip(axes,['chi','Khat','lapse','shift']):ax.set(xlabel='Distance to scri (1-r)',ylabel=f'Endpoint-subtracted {name} residual / (1-r)');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.output/'boundary-profiles.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
