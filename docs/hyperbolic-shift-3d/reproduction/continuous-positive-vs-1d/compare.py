#!/usr/bin/env python3
"""Compare retained unequal-resolution histories; no evolution or missing-data filling."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
BASE=OUT.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def table(p):return np.genfromtxt(p,delimiter=',',names=True)

old=read(BASE/'aurora/stability-puncture-v1/verification.json')['original_1d_reference']
reference=BASE/'original-1d-reference/history.csv'
provenance=BASE/'original-1d-reference/provenance.json'
assert sha(reference)==old['history_sha256'] and sha(provenance)==old['provenance_sha256']
one=table(reference);assert np.all(np.diff(one['t'])>0)
inputs={str(reference.relative_to(BASE)):sha(reference),str(provenance.relative_to(BASE)):sha(provenance)}
segments=[]
for name in ['stability-continuous-positive-puncture-v1','stability-continuous-positive-puncture-to3m-v1']:
    d=BASE/'aurora'/name;v=read(d/'verification.json');assert v['verification_passed']
    p=d/'output/puncture.history.csv'
    a=next(x for x in v['artifacts'] if x['path'].endswith('/puncture.history.csv'))
    assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256']
    inputs[str(p.relative_to(BASE))]=sha(p)
    segments.append(table(p))
assert read(BASE/'aurora/stability-continuous-positive-puncture-v1/pilot-metadata-audit.json')['qualified_pilot']
three=np.concatenate([segments[0],segments[1][segments[1]['t']>segments[0]['t'][-1]+1e-12]])
assert np.all(np.diff(three['t'])>0) and three['t_over_M'][-1]==3
assert np.allclose(three['t_over_M'],3*three['t'],rtol=1e-14)
valid=three['horizon_found']==1
fields={'horizon_radius':'apparent_horizon_r','horizon_irreducible_mass':'apparent_horizon_mass',
        'horizon_areal_radius':'apparent_horizon_areal_radius','H_exterior_L2':'H_exterior_rms',
        'Mr_exterior_L2':'M_exterior_rms'}
for key in fields:
    assert np.isfinite(three[key][valid]).all() and np.isnan(three[key][~valid]).all()
x=three['t_over_M'];assert x.min()>=one['t_over_M'].min() and x.max()<=one['t_over_M'].max()
matched={key:np.interp(x,one['t_over_M'],one[value]) for key,value in fields.items()}
radius_error=(three['horizon_radius']/matched['horizon_radius']-1)*100
provisional_radius=np.where(valid,three['horizon_radius'],three['horizon_trial_radius'])
assert np.isfinite(provisional_radius).all() and np.all(three['horizon_complete_surface'][~valid]==1)
provisional_error=(provisional_radius/matched['horizon_radius']-1)*100
mass1=np.abs(3*one['apparent_horizon_mass']-1)
mass3=np.abs(3*three['horizon_irreducible_mass']-1)
rows=[]
for i,t in enumerate(x):
    row={'t_over_M':float(t),'horizon_3d_accepted':bool(valid[i]),'horizon_radius_including_provisional':float(provisional_radius[i]),'radius_difference_including_provisional_percent':float(provisional_error[i]),'trial_expansion_rms':float(three['horizon_trial_expansion_rms'][i])}
    for key in fields:
        row[key+'_3d']=float(three[key][i]) if np.isfinite(three[key][i]) else None
        row[key+'_1d_time_interpolated']=float(matched[key][i])
    row.update(radius_difference_percent=float(radius_error[i]) if valid[i] else None,
               relative_mass_error_3d=float(mass3[i]) if valid[i] else None,
               H_norm_ratio=float(three['H_exterior_L2'][i]/matched['H_exterior_L2'][i]) if valid[i] else None,
               Mr_norm_ratio=float(three['Mr_exterior_L2'][i]/matched['Mr_exterior_L2'][i]) if valid[i] else None)
    rows.append(row)
with (OUT/'matched-times.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
selected=[rows[int(np.argmin(abs(x-t)))] for t in [0,.45,.9,1.95,3]]
summary=dict(input_sha256=inputs,method='Linear time interpolation of retained 1D history at 3D diagnostic times, no extrapolation; failed 3D horizons remain unavailable.',
    resolution=dict(one_d_n=513,one_d_dr=1/512,three_d_n_per_block=16,three_d_core_cells_per_axis=32,three_d_core_dx=.36/32,spacing_ratio=5.76,one_d_dt=4.8828125e-5,three_d_dt=.0005,dt_ratio=10.24),
    horizon_accepted_samples=int(valid.sum()),horizon_rejected_samples=int((~valid).sum()),last_accepted_horizon_t_over_M=float(x[valid][-1]),
    maximum_accepted_radius_difference_percent=float(np.max(abs(radius_error[valid]))),
    maximum_radius_difference_including_provisional_percent=float(np.max(abs(provisional_error))),final_provisional_radius_difference_percent=float(provisional_error[-1]),
    selected=selected,scope='Unequal-resolution behavioral comparison, not an equal-resolution error test or convergence claim. Exterior norms use each run own horizon cutoff and different coordinate-volume quadrature. 1D radial momentum is paired with 3D radial momentum, and mass with area-derived irreducible mass.')
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(13,8),sharex=True)
fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.13,wspace=.30,hspace=.36)
blue,orange,gray='#0072B2','#D55E00','#888888'
mask=one['t_over_M']<=3.08;tx=one['t_over_M'][mask]
ax=axs[0,0];ax.plot(tx,one['apparent_horizon_r'][mask],color=orange);ax.plot(x,provisional_radius,'-',color=blue);ax.plot(x[valid],provisional_radius[valid],'o',ms=3,color=blue)
ax.plot(x[~valid],provisional_radius[~valid],'o',mfc='white',mec=blue,ms=5)
ax.set(title='Coordinate horizon radius',ylabel='Compact coordinate r')
ax=axs[0,1];ax.semilogy(tx,mass1[mask],color=orange);ax.semilogy(x,mass3,'o-',ms=3,color=blue);ax.set(title='Horizon irreducible-mass error',ylabel='|M_irr / M − 1|')
ax=axs[0,2];ax.plot(x,provisional_error,'-',color=blue);ax.plot(x[valid],provisional_error[valid],'o',ms=3,color=blue);ax.plot(x[~valid],provisional_error[~valid],'o',mfc='white',mec=blue,ms=5);ax.axhline(0,color=orange,lw=1);ax.set(title='Radius difference from 1D',ylabel='(r_3D / r_1D − 1) × 100%')
for ax,a,b,title,ylabel in [(axs[1,0],'H_exterior_rms','H_exterior_L2','Exterior Hamiltonian constraint','Coordinate-volume RMS H'),(axs[1,1],'M_exterior_rms','Mr_exterior_L2','Exterior radial momentum constraint','Coordinate-volume RMS M_r')]:
    ax.semilogy(tx,one[a][mask],color=orange);ax.semilogy(x,three[b],'o-',ms=3,color=blue);ax.set(title=title,ylabel=ylabel)
ax=axs[1,2];ax.plot(tx,one['max_relative_null'][mask]/1e-15,color=orange);ax.plot(x,three['null_surface']/1e-15,color=blue);ax.plot(x,three['null_ghost']/1e-15,'--',color=blue);ax.set(title='Null closure (recorded maxima)',ylabel='Relative residual / 10⁻¹⁵')
for ax in axs.flat:ax.set_xlim(0,3);ax.grid(alpha=.2)
for ax in axs[1]:ax.set_xlabel('t / M')
fig.suptitle('Puncture evolution through 3M: current 3D run versus preserved 1D baseline',y=.98,fontsize=15)
fig.text(.5,.935,'Unequal resolution: 3D core Δx = 0.01125, 1D Δr = 0.001953 (5.76× finer); 3D Δt is 10.24× larger',ha='center',fontsize=11)
fig.legend(handles=[Line2D([],[],color=orange,label='1D N=513'),Line2D([],[],color=blue,marker='o',ms=3,label='3D n=16: accepted measurements'),Line2D([],[],color=blue,marker='o',mfc='white',ls='',label='Provisional 3D radius (tolerance missed)'),Line2D([],[],color=blue,ls='--',label='3D ghost null closure')],loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
fig.text(.08,.065,'Open circles use the trial radius provisionally. Missing mass/constraint values were not recorded; they are not filled.',fontsize=10)
fig.text(.08,.035,'Exterior norms use each run’s own horizon cutoff. Unequal resolution prevents attributing the error gap to the formulation.',fontsize=10)
fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf')
print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','selected','method','scope']},indent=2))
