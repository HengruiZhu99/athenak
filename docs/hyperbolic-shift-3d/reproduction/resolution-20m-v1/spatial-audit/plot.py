#!/usr/bin/env python3
"""Compare native constraints in fixed coordinate-radius bands at exactly 3M."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
fields=['H2','Mr2','M2','Theta2','Zr2'];data={n:json.loads((HERE/f'n{n}.json').read_text()) for n in [16,24,32]}
colors={16:'#b45418',24:'#1769aa',32:'#278047'}
rows=[];checks={};values={};shares={}
for n,d in data.items():
 assert d['t_over_M']==3 and d['diagnostic_only'];c=d['radial_cumulative'];r=np.array([x['cutoff'] for x in c]);assert np.array_equal(r,np.array([x['cutoff'] for x in data[16]['radial_cumulative']]))
 vol=np.array([x['volume'] for x in c]);bandvol=vol[:-1]-vol[1:];assert np.all(bandvol>0)
 vals={};checks[n]={}
 for k in fields:
  a=np.array([x[k] for x in c]);b=a[:-1]-a[1:];assert np.all(b>=-1e-12);vals[k]=np.sqrt(np.maximum(0,b)/bandvol)
  normname=k.replace('2','_L2');total=np.sqrt(a[0]/vol[0]);assert np.isclose(total,d['norms']['total'][normname],rtol=1e-12,atol=1e-13)
  idx=np.flatnonzero(r==.0753)[0];ext=np.sqrt(a[idx]/vol[idx]);assert np.isclose(ext,d['norms']['exterior'][normname],rtol=1e-12,atol=1e-13)
  checks[n][k]={'total_reconstruction_abs_error':abs(total-d['norms']['total'][normname]),'exterior_reconstruction_abs_error':abs(ext-d['norms']['exterior'][normname])}
  at={x['cutoff']:x for x in c};shares.setdefault(n,{})[k]=(at[.6][k]-at[.8][k])/at[.0753][k]
 for i in range(len(bandvol)):rows.append(dict(n=n,r_lower=r[i],r_upper=r[i+1],coordinate_volume=bandvol[i],**{k.replace('2','_RMS'):float(vals[k][i]) for k in fields}))
 values[n]=vals
with (HERE/'radial-bands.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Original full-domain diagnostics provide an independent parity check.
for n,d in data.items():
 path={16:HERE.parent/'candidate-history/n16.history.csv',24:HERE.parent/'candidate-history/n24.history.csv',32:HERE.parent/'n32-at3m/puncture.history.csv'}[n]
 original=list(csv.DictReader(path.open()))[-1]
 for k in ['H_L2','Mr_L2','M_L2','Theta_L2','Zr_L2']:
  delta=abs(d['norms']['total'][k]-float(original[k]));assert delta<1e-11,(n,k,delta);checks[n][k]={'original_full_norm_abs_error':delta}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for j,(k,title) in enumerate([('H2','Hamiltonian'),('Mr2','Radial momentum')]):
 ax=axes[0,j]
 for n in data:ax.stairs(values[n][k],r,baseline=None,color=colors[n],lw=1.8,label=f'N{n}')
 ax.set(yscale='log',title=title,ylabel='RMS within each radial band',ylim=(1e-5,30));ax.legend(ncol=3)
 ax=axes[1,j]
 for pair,c in [((24,16),'#1769aa'),((32,24),'#278047')]:
  ax.stairs(values[pair[0]][k]/values[pair[1]][k],r,baseline=None,color=c,lw=1.8,label=f'N{pair[0]} / N{pair[1]}')
 ax.axhline(1,color='.3',ls='--',lw=1);ax.set(yscale='log',ylim=(.04,5),ylabel='Refined / coarser band RMS',xlabel='Compactified coordinate radius r');ax.legend()
for ax in axes.flat:
 ax.axvspan(0,.0753,color='.7',alpha=.2);ax.axvspan(.16,.18,color='#7851a9',alpha=.18);ax.axvspan(.6,.8,color='#e9bf36',alpha=.15)
 for boundary in [.45,.75]:ax.axvline(boundary,color='.5',ls=':',lw=.9)
 ax.set_xlim(0,1);ax.grid(alpha=.2)
fig.suptitle('Where refinement stops reducing constraints: matched states at 3M',fontsize=16,y=.98)
fig.subplots_adjust(top=.9,bottom=.15,hspace=.23,wspace=.25)
fig.text(.5,.035,'Gray: excluded interior (common cutoff r = 0.0753). Purple: core–shell blend r = 0.16–0.18.\nYellow: dominant outer-shell band r = 0.6–0.8. Dotted: shell interfaces r = 0.45, 0.75.\nRatios below 1 mean improvement; unequal grid ratios preclude a single factor-two order estimate.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(HERE/f'radial-convergence.{ext}',dpi=180)
(HERE/'verification-and-contributions.json').write_text(json.dumps(dict(parity=checks,outer_band_fraction_of_exterior_squared_norm_integral=shares),indent=2)+'\n')
print(json.dumps(shares,indent=2))
