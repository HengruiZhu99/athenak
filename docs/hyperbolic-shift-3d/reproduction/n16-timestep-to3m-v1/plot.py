#!/usr/bin/env python3
"""Plot actual diagnostic samples only; keep failures and sampling times distinct."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
base=out.parent
read=lambda p:np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
one=read(base/'original-1d-reference/history.csv')
old=read(base/'aurora/stability-continuous-positive-puncture-v1/output/puncture.history.csv')
later=read(base/'aurora/stability-continuous-positive-puncture-to3m-v1/output/puncture.history.csv')
old=np.concatenate([old,later[later['t']>old['t'][-1]+1e-12]])
new=[]
for ratio,color in [('04','#D55E00'),('02','#0072B2'),('01','#009E73')]:
 d=out/f'n16-dtdx{ratio}-to3m-v1';v=json.loads((d/'verification.json').read_text());a=read(d/'output/puncture.history.csv')
 new.append((v,a,color))
stop=max(v['t_over_M'] for v,a,c in new)*1.12
fig,axs=plt.subplots(2,2,figsize=(10,7.4),sharex=True)
for ax in axs.flat:ax.grid(alpha=.2);ax.set_xlim(0,stop)
for arr,color,label in [(one,'#333333','1D N=513'),(old,'#999999','Previous 3D n16, dt/dx=0.0444')]:
 x=arr['t_over_M'];mask=x<=stop+.2
 if arr is one:
  radius=arr['apparent_horizon_r'];h=arr['H_exterior_rms'];mom=arr['M_exterior_rms'];null=arr['max_relative_null']
 else:
  radius=np.where(arr['horizon_found']==1,arr['horizon_radius'],arr['horizon_trial_radius']);h=arr['H_exterior_L2'];mom=arr['Mr_exterior_L2'];null=arr['null_surface']
 axs[0,0].plot(x[mask],h[mask],color=color,label=label)
 axs[0,1].plot(x[mask],mom[mask],color=color)
 axs[1,0].plot(x[mask],radius[mask],color=color)
 axs[1,1].plot(x[mask],null[mask]/1e-15,color=color)
for v,a,color in new:
 x=a['t_over_M'];label=f"Capped experiment dt/dx={v['dt_over_dx']:.1f}"
 axs[0,0].plot(x,a['H_exterior_L2'],'o-',ms=4,color=color,label=label)
 axs[0,1].plot(x,a['Mr_exterior_L2'],'o-',ms=4,color=color)
 radius=np.where(a['horizon_found']==1,a['horizon_radius'],a['horizon_trial_radius'])
 axs[1,0].plot(x,radius,'o-',ms=4,color=color)
 axs[1,1].plot(x,a['null_surface']/1e-15,'o-',ms=4,color=color)
 for ax in axs.flat:ax.axvline(v['t_over_M'],color=color,alpha=.45,ls=':')
axs[0,0].set(yscale='log',title='Exterior Hamiltonian RMS',ylabel='H RMS')
axs[0,1].set(yscale='log',title='Exterior radial momentum RMS',ylabel='M_r RMS')
axs[1,0].set(title='Coordinate horizon radius',ylabel='r_AH',xlabel='t / M')
axs[1,1].set(title='Scri null closure before rejection',ylabel='Recorded maximum residual / 10⁻¹⁵',xlabel='t / M')
fig.suptitle('n16 large-step attempts: capped boundary penalties',fontsize=14,y=.99)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False,fontsize=9)
fig.text(.08,.04,'Dotted lines mark last accepted times before failure. Points are actual diagnostic samples; no extrapolation.',fontsize=9)
fig.text(.08,.018,'The previous 3D run reached 3M. The 1D grid is 5.76× finer; this is not a convergence comparison.',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.875));fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.pdf')
