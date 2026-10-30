#!/usr/bin/env python3
"""Plot verified retained 3M–20M data against N513; no evolutions or gap filling."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=Path(__file__).resolve().parent;RUN=OUT.parent;BASE=RUN.parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
table=lambda p:np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
ref=BASE/'original-1d-reference/history.csv';prov=BASE/'original-1d-reference/provenance.json'
audit=read(BASE/'aurora/stability-puncture-v1/verification.json')['original_1d_reference']
assert sha(ref)==audit['history_sha256'] and sha(prov)==audit['provenance_sha256']
hist=RUN/'segment01/puncture.history.csv';ver=read(RUN/'segment01/verification.json')
record=next(a for a in ver['artifacts_verified'] if a['path'].endswith('/puncture.history.csv'))
assert sha(hist)==record['sha256'] and hist.stat().st_size==record['bytes'] and ver['status']=='verified_completed_20M'
a=table(hist);b=table(ref);x=a['t_over_M'];valid=(a['horizon_found']==1)&(a['horizon_complete_surface']==1)
assert np.all(np.diff(x)>0) and x[0]==3 and x[-1]==20 and np.allclose(x,3*a['t'])
assert np.all(np.diff(b['t_over_M'])>0) and x[0]>=b['t_over_M'][0] and x[-1]<=b['t_over_M'][-1]
fields={'horizon_radius':'apparent_horizon_r','horizon_areal_radius':'apparent_horizon_areal_radius','horizon_irreducible_mass':'apparent_horizon_mass','H_exterior_L2':'H_exterior_rms','Mr_exterior_L2':'M_exterior_rms','Theta_exterior_L2':'Theta_exterior_rms'}
match={k:np.interp(x,b['t_over_M'],b[v]) for k,v in fields.items()}
for k in fields:assert np.isfinite(a[k][valid]).all() and np.isnan(a[k][~valid]).all()
assert np.all(a['horizon_complete_surface'][~valid]==1)
trial=np.where(valid,a['horizon_radius'],a['horizon_trial_radius']);assert np.isfinite(trial).all()
diff=(a['horizon_radius']/match['horizon_radius']-1)*100
trialdiff=(trial/match['horizon_radius']-1)*100
masserr=np.abs(a['horizon_areal_radius']/(2/3)-1)
rows=[]
for i,t in enumerate(x):
 r={'t_over_M':float(t),'accepted_horizon':bool(valid[i]),'provisional_radius':float(trial[i]),'provisional_radius_difference_percent':float(trialdiff[i])}
 for k in fields:r[k+'_3d']=float(a[k][i]) if np.isfinite(a[k][i]) else None;r[k+'_1d_time_interpolated']=float(match[k][i])
 r.update(radius_difference_percent=float(diff[i]) if valid[i] else None,areal_radius_error_3d=float(masserr[i]) if valid[i] else None)
 rows.append(r)
with (OUT/'matched-times.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
summary=dict(input_sha256={str(p.relative_to(BASE)):sha(p) for p in [ref,prov,hist,RUN/'segment01/verification.json']},
 interval_t_over_M=[3,20],one_d_n=513,three_d_n_per_block=16,three_d_core_cells_per_axis=32,one_d_dr=1/512,three_d_core_dx=.01125,spacing_ratio=5.76,one_d_dt=4.8828125e-5,three_d_dt=.0005,dt_ratio=10.24,
 horizon_accepted_samples=int(valid.sum()),horizon_rejected_samples=int((~valid).sum()),rejected_horizon_times=x[~valid].tolist(),
 max_accepted_coordinate_radius_difference_percent=float(np.max(abs(diff[valid]))),max_including_provisional_radius_difference_percent=float(np.max(abs(trialdiff))),
 final=rows[-1],final_ratios={k:float(a[k][-1]/match[k][-1]) for k in ['H_exterior_L2','Mr_exterior_L2','Theta_exterior_L2']},
 final_1d_relative_areal_radius_error=abs(float(match['horizon_areal_radius'][-1])/(2/3)-1),
 method='Linear time interpolation of original N513 history at 3D times, no extrapolation; main overlays use retained samples. Only complete accepted 3D horizons supply area and exterior norms. Open radius markers show provisional candidates; no absent mass/norm is filled.',
 scope='Unequal-resolution comparison, not convergence. 3D starts at verified 3M and switches to direct chi/lapse RK+KO then. 1D is exactly spherical. Core spacing, timestep, KO/variable charts, floors, patching, horizon method and coordinate-volume quadrature differ. Each exterior norm uses its own horizon cutoff. Radial covector momentum is compared, not the 3D invariant momentum norm.')
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
blue,orange,teal='#0072B2','#D55E00','#009E73'
mask=(b['t_over_M']>=2.9)&(b['t_over_M']<=20.1);tx=b['t_over_M'][mask]
subtitle='3D core Δx = 0.01125; 1D Δr = 0.001953125 (5.76× finer). 3D Δt is 10.24× larger.'
handles=[Line2D([],[],color=orange,lw=2,label='Preserved 1D N513'),Line2D([],[],color=blue,marker='o',ms=3,label='3D n16: accepted horizon diagnostics'),Line2D([],[],color=blue,marker='o',mfc='white',ls='',label='Provisional radius: horizon tolerance missed')]
fig,ax=plt.subplots(1,3,figsize=(14.5,5.3));fig.subplots_adjust(left=.07,right=.985,bottom=.23,top=.70,wspace=.32)
ax[0].plot(tx,b['apparent_horizon_r'][mask],color=orange,lw=2)
ax[0].plot(x,a['horizon_radius'],'o-',color=blue,ms=3,lw=1.5)
ax[0].plot(x[~valid],trial[~valid],'o',color=blue,mfc='white',ms=4)
ax[0].set(title='Coordinate horizon radius',ylabel='Compact coordinate r_AH')
oneerr=np.abs(b['apparent_horizon_areal_radius'][mask]/(2/3)-1)
ax[1].semilogy(tx,np.where(oneerr>0,oneerr,np.nan),color=orange,lw=2)
ax[1].semilogy(x,np.where(masserr>0,masserr,np.nan),'o-',color=blue,ms=3,lw=1.5)
ax[1].set(title='Horizon area-radius / mass error',ylabel='|R_AH / (2M) − 1| = |M_irr / M − 1|')
ax[2].plot(x,diff,'o-',color=blue,ms=3,lw=1.5);ax[2].plot(x[~valid],trialdiff[~valid],'o',color=blue,mfc='white',ms=4);ax[2].axhline(0,color=orange,lw=1)
ax[2].set(title='Coordinate-radius difference from 1D',ylabel='(r_3D / r_1D − 1) × 100%')
for v in ax:v.set(xlim=(3,20),xlabel='t / M');v.grid(alpha=.2)
fig.suptitle('Puncture evolution to 20M: horizon comparison',fontsize=16,y=.98)
fig.text(.5,.915,subtitle,ha='center',fontsize=10.5)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.875),ncol=3,frameon=False,fontsize=9)
fig.text(.07,.115,'Direct chi/lapse continuation begins at 3M. Open circles are provisional radii; missing area diagnostics are left as gaps.',fontsize=10)
fig.text(.07,.065,'Unequal resolution and different discretizations: this is not a convergence test. Radius agreement does not prove identical outermost-root selection.',fontsize=9.5)
fig.savefig(OUT/'horizons.png',dpi=180);fig.savefig(OUT/'horizons.pdf');plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(12,8.2));fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.80,hspace=.35,wspace=.28)
for v,k,refkey,title in [(ax[0,0],'H_exterior_L2','H_exterior_rms','Exterior Hamiltonian constraint'),(ax[0,1],'Mr_exterior_L2','M_exterior_rms','Exterior radial momentum constraint'),(ax[1,0],'Theta_exterior_L2','Theta_exterior_rms','Exterior Θ constraint')]:
 v.semilogy(tx,b[refkey][mask],color=orange,lw=2);v.semilogy(x,a[k],'o-',color=blue,ms=3,lw=1.5);v.set(title=title,ylabel='Normalized coordinate-volume RMS')
v=ax[1,1];v.plot(tx,b['max_relative_null'][mask]/1e-15,color=orange,lw=2);v.plot(x,a['null_surface']/1e-15,color=blue,label='3D surface');v.plot(x,a['null_ghost']/1e-15,':',color=teal,lw=2,label='3D ghosts (dotted)');v.set(title='Outer null closure: recorded maxima',ylabel='Relative residual / 10⁻¹⁵');v.legend(loc='center right',fontsize=9,framealpha=.9)
for v in ax.flat:v.set(xlim=(3,20),xlabel='t / M');v.grid(alpha=.2)
fig.suptitle('Puncture evolution to 20M: exterior constraints and scri closure',fontsize=15,y=.985)
fig.text(.5,.94,subtitle,ha='center',fontsize=10.5)
fig.legend(handles=handles[:2],loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
fig.text(.10,.075,'Each run uses its own horizon cutoff and coordinate-volume quadrature. Missing 3D norms are not interpolated.',fontsize=9.5)
fig.text(.10,.035,'3D norms are unavailable at 38 / 114 samples where the horizon tolerance was missed; all samples after 9M are accepted.',fontsize=9.5)
fig.savefig(OUT/'constraints-and-closure.png',dpi=180);fig.savefig(OUT/'constraints-and-closure.pdf');plt.close(fig)
print(json.dumps({k:summary[k] for k in ['horizon_accepted_samples','horizon_rejected_samples','max_accepted_coordinate_radius_difference_percent','max_including_provisional_radius_difference_percent','final','final_ratios','final_1d_relative_areal_radius_error']},indent=2))
