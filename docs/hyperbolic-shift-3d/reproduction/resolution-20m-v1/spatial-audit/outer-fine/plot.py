#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
colors={16:'#b45418',24:'#1769aa',32:'#278047'}
fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,gridspec_kw={'height_ratios':[1,1,.65]})
summary={};visible_max={'H2':0.,'Mr2':0.}
for n,col in colors.items():
 d=json.loads((HERE/f'n{n}.json').read_text());c=d['radial_cumulative'];r=np.array([a['cutoff'] for a in c]);v=np.array([a['volume'] for a in c]);dv=v[:-1]-v[1:];valid=dv>1e-13;summary[n]={}
 for ax,k in zip(axes[:2],['H2','Mr2']):
  a=np.array([x[k] for x in c]);delta=a[:-1]-a[1:];assert np.all(delta>=-1e-12)
  rms=np.full(len(dv),np.nan);rms[valid]=np.sqrt(np.maximum(0,delta[valid])/dv[valid]);ax.stairs(rms,r,baseline=None,color=col,label=f'N{n}',lw=1.8)
  eligible=(r[:-1]>=.55)&(r[:-1]<.9)&valid;visible_max[k]=max(visible_max[k],float(np.nanmax(rms[eligible])));j=np.nanargmax(np.where(eligible,rms,np.nan));at={x['cutoff']:x for x in c}
  summary[n][k]={'peak_band':[r[j],r[j+1]],'peak_band_rms':rms[j],'fraction_exterior_integral_r0675_to075':(at[.675][k]-at[.75][k])/at[.0753][k]}
 # Match independently measured cumulative values at common cutoffs.
 broad=json.loads((HERE.parent/f'n{n}.json').read_text());common={x['cutoff']:x for x in broad['radial_cumulative']}
 for row in c:
  if row['cutoff'] in common:
   for k in ['volume','H2','Mr2','M2','Theta2','Zr2']:assert np.isclose(row[k],common[row['cutoff']][k],rtol=1e-12,atol=1e-13)
r=np.linspace(.55,.9,2000);y=np.clip((.8-r)/.24,0,1);w=y**8/(y**8+(1-y)**8);axes[2].plot(r,w,color='#7043a0',label='Gauge blend w(r)');axes[2].set(ylabel='Gauge blend',xlabel='Compactified coordinate radius r',ylim=(-.05,1.05));axes[2].legend()
for ax,k,title in zip(axes[:2],['H2','Mr2'],['Hamiltonian band RMS','Radial-momentum band RMS']):ax.set(ylabel=title,ylim=(0,visible_max[k]*1.15));ax.legend(ncol=3)
q=9**(1/8);t=q/(1+q);width=.24*(2*t-1);lo=.68-width/2;hi=.68+width/2
for ax in axes:
 ax.axvspan(lo,hi,color='#7043a0',alpha=.12);ax.axvline(.75,color='.3',ls='--',lw=1);ax.set_xlim(.55,.9);ax.grid(alpha=.2)
fig.suptitle('Outer-shell error concentration at 3M',fontsize=16,y=.98)
fig.subplots_adjust(top=.93,bottom=.12,hspace=.16)
fig.text(.5,.025,'Purple: gauge blend 10–90% transition. Dashed: shell interface r = 0.75.\nEmpty bands have no cell centers and are left blank. Alignment identifies suspects, not causation.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(HERE/f'outer-band-detail.{ext}',dpi=180)
summary['gauge_transition']={'center':.68,'width_10_to_90':width,'cells_10_to_90':{n:width/(.3/n) for n in colors}}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
