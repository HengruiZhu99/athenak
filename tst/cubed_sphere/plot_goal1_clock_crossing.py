#!/usr/bin/env python3
"""Plot measured native interface waveforms and complete saved endpoint errors."""
import argparse,json,csv,hashlib,ast
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('evidence',type=Path);p.add_argument('case');a=p.parse_args();report=json.loads((a.evidence/'measured-interface-waveforms.json').read_text());case=a.work/a.case
with (case/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2);x=np.fromfile(f,np.float64).reshape(int(n),int(c))
with (a.work/('analysis-'+a.case)/'target.bin').open('rb')as f:nn,cc=np.fromfile(f,np.uint64,2);target=np.fromfile(f,np.float64).reshape(-1,22,13)
assert n==nn and c==300 and cc==286;fields=x[:,14:].reshape(-1,22,13);amp=.0005;length=.2;config=[0,1,2,3,4,5,6,18,19,20,21];scale=np.full(22,amp/length);scale[config]=amp;d=(fields-target)/scale[None,:,None];point={k:np.sqrt(np.sum(z*z,axis=tuple(range(1,z.ndim))))for k,z in [('field',d[:,:,0]),('first',length*d[:,:,1:4]),('Hessian',length*length*d[:,config,4:])]};r=np.sqrt(np.einsum('ij,ij->i',x[:,:3],x[:,:3]));utility=Path(__file__).with_name('review_goal1_clock_pilots.py');tree=ast.parse(utility.read_text());ns={'np':np};exec(compile(ast.Module(body=[v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='reference_weights'],type_ignores=[]),str(utility),'exec'),ns);w=ns['reference_weights'](x)['fixed_reference_Penrose'];edges=np.linspace(0,.7,29);bins=np.minimum(np.searchsorted(edges,r,side='right')-1,len(edges)-2);profile=[]
for i in range(len(edges)-1):
 mask=bins==i
 if not mask.any():continue
 profile.append({'radius':float(np.sum(w[mask]*r[mask])/sum(w[mask])),'count':int(mask.sum()),'reference_volume':float(sum(w[mask])),**{k+'_RMS':float(np.sqrt(np.sum(w[mask]*v[mask]**2)/sum(w[mask])))for k,v in point.items()},**{k+'_Linf':float(v[mask].max())for k,v in point.items()}})
with (a.evidence/'native-clock-final-radial-profile.csv').open('w')as f:
 writer=csv.DictWriter(f,fieldnames=list(profile[0]));writer.writeheader();writer.writerows(profile)
xyedges=np.linspace(-.7,.7,61);ix=np.clip(np.searchsorted(xyedges,x[:,0],side='right')-1,0,59);iy=np.clip(np.searchsorted(xyedges,x[:,1],side='right')-1,0,59);xymax=np.full((60,60),np.nan);maximum=np.zeros(3600);np.maximum.at(maximum,iy*60+ix,point['first']);xymax=maximum.reshape(60,60);xymax[xymax==0]=np.nan;np.savez_compressed(a.evidence/'native-clock-final-error-map.npz',xy_edges=xyedges,first_max_over_z=xymax)
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=['#235b8e','#b35136','#548049','#9467bd']
for result,color in zip(report['results'],colors):
 rows=result['samples'];t=[s['time']for s in rows];label=result['name'].replace('_',' ')
 for panel,var in [(ax[0,0],'alpha'),(ax[0,1],'radial_beta')]:
  panel.plot(t,[s['oracle_'+var+'_perturbation']/amp for s in rows],'-',color=color,lw=1,label=label);panel.plot(t,[s['native_'+var+'_perturbation']/amp for s in rows],'.',color=color,ms=4)
ax[0,0].set(title='Lapse at four fixed interface nodes',xlabel='Coordinate time',ylabel='Perturbation / .0005');ax[0,1].set(title='Radial shift: oracle lines, native dots',xlabel='Coordinate time',ylabel='Perturbation / .0005');ax[0,0].legend(fontsize=8)
for name,color in [('field','#235b8e'),('first','#b35136'),('Hessian','#548049')]:ax[1,0].semilogy([v['radius']for v in profile],[v[name+'_RMS']for v in profile],color=color,label=name+' RMS')
ax[1,0].axvspan(.15,np.sqrt(3)*.15,color='0.8',alpha=.4,label='cube-shell radius range');ax[1,0].axhline(.01,color='0.5',ls=':',lw=1);ax[1,0].set(title='Final native radial error profile',xlabel='Compact radius',ylabel='Normalized reference-volume RMS');ax[1,0].legend(fontsize=8)
im=ax[1,1].pcolormesh(xyedges,xyedges,xymax,norm=LogNorm(vmin=1e-5,vmax=max(.01,np.nanmax(xymax))),cmap='viridis',shading='flat');ax[1,1].set_aspect('equal');ax[1,1].set(title='First-derivative error: max over z in each bin',xlabel='x',ylabel='y');fig.colorbar(im,ax=ax[1,1],label='Normalized point error; no interpolation')
for panel in ax.flat:panel.grid(alpha=.18)
phase=max(abs(x['fitted_native_time_advance'])for x in report['results']);fig.suptitle(f'Native second-order Z4c clock pilot: interface waveforms through t=.3\nMaximum fitted phase error {phase:.3g}; full-packet tails and causal boundary gates remain open',fontsize=12);fig.savefig(a.evidence/'native-clock-interface-profile.png',dpi=160);fig.savefig(a.evidence/'native-clock-interface-profile.svg');plt.close(fig)
receipt={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'case':a.case,'snapshot_sha256':hashlib.sha256((case/'cubed_sphere_native_last_jets.bin').read_bytes()).hexdigest(),'waveforms_sha256':hashlib.sha256((a.evidence/'measured-interface-waveforms.json').read_bytes()).hexdigest(),'scope':'Read-only saved native data plots; no scientific acceptance, all-node errors retained in binary/region reviews. Projection map takes maxima, not selected low-error slices.'};(a.evidence/'crossing-plot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
