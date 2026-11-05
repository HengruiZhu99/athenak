#!/usr/bin/env python3
"""Independent field self-convergence on exact reference-map radii.

Local degree-six and degree-four interpolation disagreement is recorded, not
silently treated as evolution error. Temporal controls share a grid, so their
field differences are interpolated directly. No asymptotic-order assertion.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernel import Kernel,FIELDS
from evolution import weights

def interpolate(r,q,target,width):
 out=[]
 for x in target:
  start=min(max(np.searchsorted(r,x)-width//2,0),len(r)-width)
  idx=slice(start,start+width);out.append(weights(r[idx],x)@q[idx])
 return np.array(out)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference-interpolation',action='store_true');ap.add_argument('--library',required=True);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--cases',nargs='+',required=True);ap.add_argument('--kind',choices=('temporal','spatial'),required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 R=np.geomspace(25,100,301);cfg0=json.loads((a.evidence/a.cases[0]/'manifest.json').read_text())['config'];k=Kernel.from_config(a.library,cfg0)
 def radius(r):
  q,_,om=k.initial(np.array([r]));return 3*r/(om[0,0]*np.sqrt(q[0,0]))
 target=np.array([brentq(lambda r:radius(r)-value,k.config[2],.99999,xtol=2e-14) for value in R]);k.close()
 runs=[];data=[]
 for name in a.cases:
  folder=a.evidence/name;result=json.loads((folder/'result.json').read_text());cfg=json.loads((folder/'manifest.json').read_text())['config']
  assert result['status']=='completed_unqualified' and abs(result['time_M']-cfg0['duration'])<1e-12
  assert all(cfg.get(key)==cfg0.get(key) for key in ('radius','slicing_ratio','mode','separated','compact_end'))
  assert cfg.get('chi_reference_power',0)==cfg0.get('chi_reference_power',0)
  assert cfg.get('auxiliary_damping',1.)==cfg0.get('auxiliary_damping',1.)
  for key,default in [('outflow_boundary',False),('outflow_width',5),('lapse_normal_lift','hermite'),('boundary_normal_system','coupled'),('compatible_lapse',0.)]:assert cfg.get(key,default)==cfg0.get(key,default)
  state=np.load(folder/'last-accepted.npz');r=state['r'];q=state['q'];total_disagreement=np.sqrt(np.mean((interpolate(r,q,target,7)-interpolate(r,q,target,5))**2,axis=0))
  if a.reference_interpolation:q=q-np.load(folder/'initial.npz')['q']
  high=interpolate(r,q,target,7);low=interpolate(r,q,target,5)
  data.append((r,q,high));runs.append(dict(case=name,n=len(r),dt_M=result['dt_over_M'],seconds=result['stepping_seconds'],total_field_interpolation_disagreement_rms=total_disagreement.tolist(),interpolation_disagreement_rms=np.sqrt(np.mean((high-low)**2,axis=0)).tolist()))
 pairs=[];differences=[]
 for i in range(len(data)-1):
  ra,qa,sa=data[i];rb,qb,sb=data[i+1]
  if a.kind=='temporal':
   assert np.array_equal(ra,rb);delta=interpolate(ra,qa-qb,target,7);low=interpolate(ra,qa-qb,target,5)
  else:delta=sa-sb;low=interpolate(ra,qa,target,5)-interpolate(rb,qb,target,5)
  differences.append(delta);rms=np.sqrt(np.mean(delta**2,axis=0))
  pairs.append(dict(cases=a.cases[i:i+2],field_rms=rms.tolist(),max_field_rms=float(max(rms)),interpolation_disagreement_rms=np.sqrt(np.mean((delta-low)**2,axis=0)).tolist()))
 orders=[]
 for i in range(len(pairs)-1):
  x=np.array(pairs[i]['field_rms']);y=np.array(pairs[i+1]['field_rms']);p=np.log2(np.divide(x,y,out=np.ones_like(x),where=(x>1e-14)&(y>1e-14)));orders.append([float(v) if x[j]>1e-14 and y[j]>1e-14 else None for j,v in enumerate(p)])
 out=dict(kind=a.kind,reference_interpolation=a.reference_interpolation,domain=f"Reference R/M 25–100, t={cfg0['duration']:g}M; exact shared initial-map inversion; logarithmic reference sampling",fields=list(FIELDS),runs=runs,pairs=pairs,adjacent_log2_norm_ratios=orders,qualification='Self-differences and interpolation sensitivity only; boundary compatibility and long-duration qualification remain separate.')
 (a.output/'convergence.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(a.output/'differences.npz',reference_R_M=R,compact_r=target,differences=np.array(differences))
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for j,f in enumerate((0,3,8,9,11)):
  axes[0].semilogy(np.arange(len(pairs)),[max(x['field_rms'][f],1e-16) for x in pairs],'o-',label=FIELDS[f])
  axes[1].loglog(R,np.maximum(abs(differences[-1][:,f]),1e-16),label=FIELDS[f])
 axes[0].set(xticks=np.arange(len(pairs)),xticklabels=[f"{x['n']}→{y['n']}" if a.kind=='spatial' else f"dt/{2**i}→dt/{2**(i+1)}" for i,(x,y) in enumerate(zip(runs[:-1],runs[1:]))],ylabel='Common-exterior field difference RMS',title=f'{a.kind.capitalize()} self-differences')
 axes[1].set(xlabel='Reference areal radius / M',ylabel='Absolute field difference',title='Finest adjacent pair')
 for ax in axes:ax.grid(alpha=.25,which='both');ax.legend(fontsize=8)
 fig.savefig(a.output/'convergence.png',dpi=160);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
