#!/usr/bin/env python3
"""Compare accepted puncture diagnostics to the preserved unequal-resolution 1D run."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--history',action='append',required=True,help='label=path; repeat a label for ordered restart segments')
p.add_argument('--n',type=int,default=16);p.add_argument('--dt',type=float,default=.0005);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
reference=Path(__file__).resolve().parent.parent/'original-1d-reference/history.csv'
one=np.atleast_1d(np.genfromtxt(reference,delimiter=',',names=True));assert np.all(np.diff(one['t_over_M'])>0)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={str(reference):sha(reference)};groups={}
for item in a.history:
 label,path=item.split('=',1);path=Path(path);inputs[str(path)]=sha(path)
 data=np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True));assert np.all(np.diff(data['t_over_M'])>0)
 if label in groups:
  previous=groups[label];data=data[data['t_over_M']>previous['t_over_M'][-1]+1e-12];groups[label]=np.concatenate([previous,data])
 else:groups[label]=data
fields={'horizon_radius':'apparent_horizon_r','horizon_irreducible_mass':'apparent_horizon_mass','H_exterior_L2':'H_exterior_rms','Mr_exterior_L2':'M_exterior_rms'}
rows=[];summary={};fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');end=max(x['t_over_M'][-1] for x in groups.values());mask=one['t_over_M']<=end
axes[0,0].plot(one['t_over_M'][mask],one['apparent_horizon_r'][mask],label='1D N=513',color='black')
axes[0,1].semilogy(one['t_over_M'][mask],abs(3*one['apparent_horizon_mass'][mask]-1),color='black')
for axis,key in [(axes[1,0],'H_exterior_rms'),(axes[1,1],'M_exterior_rms')]:axis.semilogy(one['t_over_M'][mask],one[key][mask],color='black')
for label,data in groups.items():
 x=data['t_over_M'];valid=data['horizon_found']==1;assert np.allclose(x,3*data['t'],rtol=1e-14)
 assert x.min()>=one['t_over_M'][0] and x.max()<=one['t_over_M'][-1]
 for key in fields:assert np.isfinite(data[key][valid]).all() and np.isnan(data[key][~valid]).all()
 matched={k:np.interp(x,one['t_over_M'],one[v]) for k,v in fields.items()}
 error=(data['horizon_radius']/matched['horizon_radius']-1)*100
 for i,time in enumerate(x):
  row=dict(case=label,t_over_M=float(time),accepted=bool(valid[i]))
  for key in fields:row[key+'_3d']=float(data[key][i]) if valid[i] else None;row[key+'_1d_time_interpolated']=float(matched[key][i])
  row['radius_difference_percent']=float(error[i]) if valid[i] else None;rows.append(row)
 summary[label]=dict(accepted=int(valid.sum()),rejected=int((~valid).sum()),end_t_over_M=float(x[-1]),
  last_accepted_t_over_M=float(x[valid][-1]) if valid.any() else None,max_accepted_radius_difference_percent=float(np.max(abs(error[valid]))) if valid.any() else None)
 axes[0,0].plot(x,data['horizon_radius'],'o-',ms=3,label=label)
 axes[0,1].semilogy(x,abs(3*data['horizon_irreducible_mass']-1),'o-',ms=3)
 axes[0,2].plot(x,error,'o-',ms=3)
 axes[1,0].semilogy(x,data['H_exterior_L2'],'o-',ms=3)
 axes[1,1].semilogy(x,data['Mr_exterior_L2'],'o-',ms=3)
 axes[1,2].plot(x,np.maximum(data['null_surface'],data['null_ghost'])/1e-15,'o-',ms=3)
for axis,title,ylabel in zip(axes.flat,['Coordinate horizon radius','Irreducible mass error','Radius difference from 1D','Exterior Hamiltonian','Exterior radial momentum','Scri / ghost null closure'],['r','|M_irr/M - 1|','Percent','Coordinate-volume RMS H','Coordinate-volume RMS M_r','Maximum relative residual / 1e-15']):
 axis.set(title=title,xlabel='t / M',ylabel=ylabel,xlim=(0,end));axis.grid(alpha=.2)
axes[0,0].legend(fontsize=8);axes[0,2].axhline(0,color='black',lw=.7)
fig.suptitle(f'Unequal-resolution puncture comparison: 3D n={a.n}, 1D N=513\nRejected horizons remain unavailable; no interpolation across missing 3D samples')
fig.savefig(a.out/'comparison.png',dpi=180);fig.savefig(a.out/'comparison.pdf')
with (a.out/'matched-times.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(inputs=inputs,cases=summary,resolution=dict(n_3d=a.n,dx_3d=.36/(2*a.n),dr_1d=1/512,spacing_ratio=(.36/(2*a.n))*512,dt_3d=a.dt,dt_1d=4.8828125e-5),
 scope='Linear time interpolation of retained 1D history only; no 3D gap filling. Own accepted horizon cutoffs and coordinate-volume quadratures differ. This is a behavioral comparison, not an accuracy or convergence measurement.')
(a.out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result['cases'],indent=2))
