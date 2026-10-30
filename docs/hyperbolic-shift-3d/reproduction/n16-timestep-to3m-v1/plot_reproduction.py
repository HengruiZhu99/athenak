#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent;d=out/'n16-original-dt-reproduction-v1'
assert json.loads((d/'final-checkpoint-comparison.json').read_text())['bitwise_equal']
assert all(x['all_scientific_columns_equal'] for x in json.loads((d/'history-comparison.json').read_text()))
read=lambda p:np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
a=read(d/'output/puncture-initial.history.csv');b=read(d/'output/puncture.history.csv');a=np.concatenate([a,b[b['t']>a['t'][-1]+1e-12]])
o=read(out.parent/'original-1d-reference/history.csv');o=o[o['t_over_M']<=3.1]
x=a['t_over_M'];valid=a['horizon_found']==1;r=np.where(valid,a['horizon_radius'],a['horizon_trial_radius'])
fig,axes=plt.subplots(2,2,figsize=(10,7.5),sharex=True);blue='#0072B2';orange='#D55E00'
axes[0,0].semilogy(o['t_over_M'],abs(3*o['apparent_horizon_mass']-1),color=orange,label='1D N=513')
axes[0,0].semilogy(x,abs(3*a['horizon_irreducible_mass']-1),'o-',ms=3,color=blue,label='Optimized 3D n16 (identical to original)')
axes[0,0].set(title='Horizon mass error',ylabel='|M_AH / M − 1|')
axes[0,1].semilogy(o['t_over_M'],o['H_exterior_rms'],color=orange)
axes[0,1].semilogy(x,a['H_exterior_L2'],'o-',ms=3,color=blue)
axes[0,1].set(title='Exterior Hamiltonian RMS',ylabel='H RMS')
axes[1,0].plot(o['t_over_M'],o['apparent_horizon_r'],color=orange)
axes[1,0].plot(x,r,color=blue);axes[1,0].plot(x[valid],r[valid],'o',ms=3,color=blue);axes[1,0].plot(x[~valid],r[~valid],'o',ms=4,color=blue,mfc='white')
axes[1,0].set(title='Coordinate horizon radius',ylabel='r_AH',xlabel='t / M')
axes[1,1].plot(o['t_over_M'],o['max_relative_null']/1e-15,color=orange)
axes[1,1].plot(x,a['null_surface']/1e-15,color=blue);axes[1,1].plot(x,a['null_ghost']/1e-15,':',color=blue)
axes[1,1].set(title='Outer null closure',ylabel='Recorded residual / 10⁻¹⁵',xlabel='t / M')
for ax in axes.flat:ax.set_xlim(0,3);ax.grid(alpha=.2)
fig.suptitle('Original-step reproduction through 3M: dt/dx = 0.044444',fontsize=14,y=.99)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False,fontsize=9)
fig.text(.08,.045,'Open radius markers are complete horizon candidates that miss the residual tolerance. Missing norms are not filled.',fontsize=9)
fig.text(.08,.022,'1D is 5.76× finer than the 3D core. Dotted blue line: ghost null closure. This does not establish convergence.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.90));fig.savefig(out/'reproduction.png',dpi=170);fig.savefig(out/'reproduction.pdf')
