#!/usr/bin/env python3
"""Measured characteristic compression versus frozen-model relaxation scales."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegraph_analysis_io import trajectory,physical_lapse_gradient
from telegraph import window

p=argparse.ArgumentParser()
p.add_argument('--cases',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
records=[]
for case_index,folder in enumerate(a.cases):
 m=json.loads((folder/'manifest.json').read_text());assert m['status']=='finished'
 args=m['arguments'];assert m['initial_data']['mass']==1/3
 data,ancestry=trajectory(folder);rows=[]
 label=f"{args.get('inner_profile','collar')}, k={args['kappa']:g}, N={args['n']}"
 for t,d in sorted(data.items()):
  r=d['r'];O=(1-r*r)/2;A=(1+r*r)/2;q=d['q']
  chi=r**4/(r**4+O**4)*np.exp(q[:,0]);alpha=A*r*r/(r*r+O**4)*np.exp(q[:,6])
  w,_=window(r,args['inner'],args['outer'],args['transition'],args.get('inner_profile','collar'))
  F=alpha**2+4*args['nu']*O**2*alpha
  ct2=chi*args['kappa']*w*w/args['tau'];cg2=chi*F
  gamma=np.divide(ct2,2*args['tau']*(cg2+ct2),out=np.zeros_like(r),where=(cg2+ct2)>0)
  mask=r<.3;physical=alpha[mask]/O[mask]
  gradient=physical_lapse_gradient(d)[mask];j=np.argmax(abs(gradient))
  compression=-np.gradient(d['gauge_speeds'][:,0],r,edge_order=2)
  # Snapshot derivatives are diagnostics; rays themselves used RK-stage speeds.
  row=dict(t=t,r_front=float(r[j]),front_gradient=float(abs(gradient[j])),
           compression_at_front=float(compression[j]),frozen_high_k_damping_at_front=float(gamma[j]),
           min_ray_logJ=float(d['rays'][:,1].min()),min_inward_spacing=float(np.diff(d['rays'][::2,0]).min()),
           H_max=float(abs(d['diagnostic'][:,0]).max()))
  rows.append(row)
  if t==max(data):
   # A sparse ray fan is readable and keeps both families separately labeled.
   times=sorted(data);rays=np.array([data[x]['rays'] for x in times]);active=np.array(times)>=args['ray_start']
   for k in range(0,rays.shape[1],20):
    axes[0,0].plot(np.array(times)[active],rays[active,k,0],color=f'C{case_index}',ls='-' if case_index%2==0 else '--',alpha=.65,lw=.8,label=label if k==0 else None)
 for ax,key in [(axes[0,1],'min_ray_logJ'),(axes[1,0],'compression_at_front'),(axes[1,1],'frozen_high_k_damping_at_front')]:
  ax.plot([x['t'] for x in rows],[x[key] for x in rows],label=label)
 records.append(dict(path=str(folder.resolve()),arguments=args,ancestry=ancestry,rows=rows))
for ax,title,ylabel in zip(axes.flat,['Ingoing characteristic paths','Minimum ray-map Jacobian','Ingoing compression at lapse front','Frozen high-frequency damping at front'],['r',r'$\min\log J$',r'$-\partial_r c_-$',r'$\Gamma_\infty$']):
 ax.set(title=title,xlabel='t',ylabel=ylabel);ax.grid(alpha=.2)
axes[0,1].legend(fontsize=7);axes[0,0].legend(fontsize=7)
fig.savefig(a.output/'characteristics.png',dpi=160);fig.savefig(a.output/'characteristics.pdf');plt.close(fig)
result=dict(records=records,scope='Rays are integrated with stage geometry. Lapse gradients use analytic AF factors and seven-point chart derivatives; speed gradients use second-order differences. All require refinement. Frozen damping is a local diagnostic, not a nonlinear threshold. Boundary exit coincidences are not crossings; finite-grid noncrossing is not a continuum proof.',
 formula=dict(c='-beta +/- sqrt(chi*(F+kappa*w^2/tau))',gamma_infinity='cT^2/[2*tau*(cg^2+cT^2)]'))
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(path=x['path'],final=x['rows'][-1]) for x in records],indent=2))
