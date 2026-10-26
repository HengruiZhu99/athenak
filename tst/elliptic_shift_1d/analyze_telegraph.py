#!/usr/bin/env python3
"""Accepted-state convergence and stage-ray comparison; no evolution."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
records=[];groups={};profiles={}
configuration_keys=['inner','outer','transition','cfl','inner_profile','quadrature_width','puncture_ko','puncture_ko_radius','grid_stretch','grid_map']
from telegraph_analysis_io import trajectory,physical_lapse_gradient
for folder in a.cases:
 m=json.loads((folder/'manifest.json').read_text());assert m['status']=='finished',str(folder)
 assert m.get('initial_data',{}).get('mass')==1/3,'Reject the misconfigured M=.1 pilots'
 args=m['arguments']
 for k,v in dict(inner_profile='collar',quadrature_width=7,puncture_ko=.5,puncture_ko_radius=.8,grid_stretch=0.,grid_map='sinh').items():args.setdefault(k,v)
 for existing in records:
  if (existing['kappa'],existing['tau'],existing['nu'])==(args['kappa'],args['tau'],args['nu']):
   assert existing['configuration']==[args[x] for x in configuration_keys],'Analyze separate support/CFL/solver controls in separate invocations'
 key=(args['kappa'],args['tau'],args['nu'],*[args[x] for x in configuration_keys]);n=args['n'];groups.setdefault(key,[]).append(n)
 data,ancestry=trajectory(folder);snapshot=None;front_history=[]
 for t in sorted(data):
  d=data[t];r=d['r'];mask=(r>=0)&(r<.3);alpha=d['diagnostic'][:,7];O=(1-r*r)/2
  grad=physical_lapse_gradient(d)[mask];j=int(np.argmax(abs(d['diagnostic'][:,0])))
  front_history.append(dict(t=t,gradient=float(max(abs(grad))),gradient_location=float(r[mask][np.argmax(abs(grad))]),H_peak=float(abs(d['diagnostic'][j,0])),H_peak_location=float(r[j]),min_logJ=float(min(d['rays'][:,1]))))
  if abs(t-2.5)<1e-9:
   d=data[round(t,9)];r=d['r'];mask=(r>.03)&(r<.3);alpha=d['diagnostic'][:,7];O=(1-r*r)/2
   grad=np.gradient(alpha[mask]/O[mask],r[mask],edge_order=2);j=int(np.argmax(abs(d['diagnostic'][:,0])))
   snapshot=dict(front_gradient=float(max(abs(grad))),front_H=float(max(abs(d['diagnostic'][mask,0]))),H_peak_location=float(r[j]),H_peak=float(abs(d['diagnostic'][j,0])),ray_min_logJ=float(min(d['rays'][::2,1])))
 profiles[(key,n)]=data
 records.append(dict(path=str(folder.resolve()),ancestry=ancestry,manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),n=n,configuration=[args[x] for x in configuration_keys],kappa=key[0],tau=key[1],nu=key[2],time=m['time'],stages=m['all_RK_stages'],snapshot_t2p5=snapshot,front_history=front_history))
comparisons=[]
for key,ns in groups.items():
 ns=sorted(ns)
 if len(ns)<3:continue
 for na,nb,nc in zip(ns,ns[1:],ns[2:]):
  A,B,C=[profiles[(key,n)] for n in [na,nb,nc]];times=sorted(set(A)&set(B)&set(C));rows=[]
  for t in times:
   r=A[t]['r'];rat1=(nb-1)//(na-1);rat2=(nc-1)//(nb-1);assert (nb-1)==rat1*(na-1) and (nc-1)==rat2*(nb-1)
   assert np.array_equal(r,B[t]['r'][::rat1]) and np.array_equal(r,C[t]['r'][::rat1*rat2])
   fields=[x[t]['q_physical'][:,:7] for x in [A,B,C]]
   e1=fields[0]-fields[1][::rat1];e2=fields[1][::rat1]-fields[2][::rat1*rat2]
   row=dict(t=t)
   for name,mask in [('bulk',r<.8),('endpoint',r>.9)]:
    measure=r**2*A[t]['grid_jacobian']
    norm=lambda e:float(np.sqrt(np.sum(measure[mask,None]*e[mask]**2)/(7*np.sum(measure[mask]))))
    row[name+'_coarse']=norm(e1);row[name+'_fine']=norm(e2)
   rows.append(row)
  out=dict(kappa=key[0],tau=key[1],nu=key[2],grids=[na,nb,nc],rows=rows)
  for name in ['bulk','endpoint']:
   e1=max(x[name+'_coarse'] for x in rows);e2=max(x[name+'_fine'] for x in rows)
   out[name+'_peak_order']=float(np.log(e1/e2)/np.log(rat1)) if e2 else None
  rs=[next(x for x in records if x['n']==n and (x['kappa'],x['tau'],x['nu'])==key[:3]) for n in [na,nb,nc]]
  out['constraint_peak_orders']={}
  for name in ['H_rms','M_rms','Theta_rms','Z_rms','H_bulk','M_bulk']:
   values=[x['stages']['max_accepted_'+name] for x in rs]
   out['constraint_peak_orders'][name]=dict(values=values,orders=[float(np.log(x/y)/np.log(rat1)) if x>0 and y>0 else None for x,y in zip(values,values[1:])])
  comparisons.append(out)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for rec in records:
 key=next(k for k in groups if k[:3]==(rec['kappa'],rec['tau'],rec['nu']));n=rec['n'];data=profiles[(key,n)];d=data[2.5];r=d['r'];mask=(r>.07)&(r<.3);label=f"k={key[0]:g}, N={n}"
 axes[0,0].plot(r[mask],d['diagnostic'][mask,7]/((1-r[mask]**2)/2),label=label)
 axes[0,1].plot(r[mask],d['diagnostic'][mask,0],label=label)
 times=sorted(data);axes[1,1].plot(times,[min(data[t]['rays'][::2,1]) for t in times],label=label)
for key,ns in groups.items():
 rs=sorted([x for x in records if (x['kappa'],x['tau'],x['nu'])==key[:3]],key=lambda x:x['n'])
 axes[1,0].loglog([x['n'] for x in rs],[x['stages']['max_accepted_H_rms'] for x in rs],'o-',label=f'k={key[0]:g}, tau={key[1]:g}')
axes[0,0].set(title='Physical lapse at t=2.5',xlabel='r',ylabel=r'$\widetilde\alpha$')
axes[0,1].set(title='Front Hamiltonian at t=2.5',xlabel='r',ylabel=r'$\mathcal{H}$');axes[0,1].set_yscale('symlog',linthresh=1e-5)
axes[1,0].set(title='Full-domain accepted-state maximum',xlabel='N',ylabel=r'$\max_t\|\mathcal{H}\|_r$')
axes[1,1].set(title='Ingoing rays integrated at RK stages',xlabel='t',ylabel=r'$\min\log J$')
for ax in axes.flat:ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=8)
fig.savefig(a.output/'comparison.png',dpi=160);fig.savefig(a.output/'comparison.pdf');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for rec in records:
 h=rec['front_history'];t=[x['t'] for x in h];label=f"k={rec['kappa']:g}, N={rec['n']}"
 for ax,name in zip(axes.flat,['gradient','gradient_location','H_peak','H_peak_location']):ax.plot(t,[x[name] for x in h],label=label)
for ax,title,ylabel in zip(axes.flat,['Physical lapse gradient, 0 <= r < .3','Position of maximum gradient','Full-domain Hamiltonian maximum','Position of maximum Hamiltonian'],[r'$\max|\partial_r\widetilde\alpha|$','r',r'$\max|\mathcal{H}|$','r']):
 ax.set(title=title,xlabel='t',ylabel=ylabel);ax.grid(alpha=.2)
axes[1,0].set_yscale('log');axes[0,0].legend(fontsize=8)
fig.savefig(a.output/'front-history.png',dpi=160);fig.savefig(a.output/'front-history.pdf');plt.close(fig)
result=dict(records=records,comparisons=comparisons,scope='Completed full-domain M=1/3 cases only. Snapshots at exact common times, no temporal interpolation. Accepted-state extrema include the entire run. Pilot duration cannot establish long-term robustness. Baseline and telegraph gauges are not compared as equal continuum solutions.')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(records=[{k:v for k,v in x.items() if k not in ['stages','front_history']} for x in records],comparisons=[{k:v for k,v in x.items() if k!='rows'} for x in comparisons]),indent=2))
