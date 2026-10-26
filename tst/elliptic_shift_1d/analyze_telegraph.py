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
for folder in a.cases:
 m=json.loads((folder/'manifest.json').read_text());assert m['status']=='finished',str(folder)
 assert m.get('initial_data',{}).get('mass')==1/3,'Reject the misconfigured M=.1 pilots'
 args=m['arguments'];
 for existing in records:
  if (existing['kappa'],existing['tau'],existing['nu'])==(args['kappa'],args['tau'],args['nu']):
   assert existing['configuration']==[args[x] for x in ['inner','outer','transition','cfl']],'Analyze separate support/CFL controls in separate invocations'
 key=(args['kappa'],args['tau'],args['nu'],args['inner'],args['outer'],args['transition'],args['cfl']);n=args['n'];groups.setdefault(key,[]).append(n)
 data={}
 for path in sorted(folder.glob('profile-*.npz')):
  with np.load(path,allow_pickle=False) as z:
   t=float(z['t']);data[round(t,9)]={k:z[k].copy() for k in ['r','q_physical','diagnostic','rays','B']}
  if abs(t-2.5)<1e-9:
   d=data[round(t,9)];r=d['r'];mask=(r>.03)&(r<.3);alpha=d['diagnostic'][:,7];O=(1-r*r)/2
   grad=np.gradient(alpha[mask]/O[mask],r[mask],edge_order=2);j=int(np.argmax(abs(d['diagnostic'][:,0])))
   snapshot=dict(front_gradient=float(max(abs(grad))),front_H=float(max(abs(d['diagnostic'][mask,0]))),H_peak_location=float(r[j]),H_peak=float(abs(d['diagnostic'][j,0])),ray_min_logJ=float(min(d['rays'][::2,1])))
 profiles[(key,n)]=data
 records.append(dict(path=str(folder.resolve()),manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),n=n,configuration=[args[x] for x in ['inner','outer','transition','cfl']],kappa=key[0],tau=key[1],nu=key[2],time=m['time'],stages=m['all_RK_stages'],snapshot_t2p5=snapshot))
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
    norm=lambda e:float(np.sqrt(np.sum(r[mask,None]**2*e[mask]**2)/(7*np.sum(r[mask]**2))))
    row[name+'_coarse']=norm(e1);row[name+'_fine']=norm(e2)
   rows.append(row)
  out=dict(kappa=key[0],tau=key[1],nu=key[2],grids=[na,nb,nc],rows=rows)
  for name in ['bulk','endpoint']:
   e1=max(x[name+'_coarse'] for x in rows);e2=max(x[name+'_fine'] for x in rows)
   out[name+'_peak_order']=float(np.log(e1/e2)/np.log(rat1)) if e2 else None
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
result=dict(records=records,comparisons=comparisons,scope='Completed full-domain M=1/3 cases only. Snapshots at exact common times, no temporal interpolation. Accepted-state extrema include the entire run. Pilot duration cannot establish long-term robustness. Baseline and telegraph gauges are not compared as equal continuum solutions.')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(records=[{k:v for k,v in x.items() if k not in ['stages']} for x in records],comparisons=[{k:v for k,v in x.items() if k!='rows'} for x in comparisons]),indent=2))
