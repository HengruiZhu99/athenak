#!/usr/bin/env python3
"""V6/V7/1D descriptive comparison with missing horizons left unavailable."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
reference=root.parent.parent/'original-1d-reference/history.csv'
assert hashlib.sha256(reference.read_bytes()).hexdigest()=='95267982d749ea69c1bf397e6fc80407239f40e25d43240bf77b18828cfe3b6d'
one=np.genfromtxt(reference,delimiter=',',names=True);one=one[one['t_over_M']<=3]
series=[('V6','#777777',root.parent/'stability-donor-seven-puncture-v1/raw/output/puncture.history.csv'),('V7','#0072B2',root/'raw/output/puncture.history.csv')]
positive=lambda a:np.where(a>0,a,np.nan)
fig,ax=plt.subplots(2,2,figsize=(10.5,7.5),layout='constrained')
for label,color,path in series:
 d=np.genfromtxt(path,delimiter=',',names=True);t=d['t_over_M'];ok=d['horizon_found']==1
 ax[0,0].semilogy(t,np.where(ok,positive(np.abs(3*d['horizon_mass']-1)),np.nan),color=color,label=label+' n16')
 ax[0,1].plot(t,np.where(ok,d['horizon_radius'],np.nan),color=color,label=label+' n16')
 for field,ls,symbol in [('H_exterior_L2','-','H'),('Mr_exterior_L2',':','radial M')]:
  ax[1,0].semilogy(t,positive(d[field]),ls,color=color,label=label+' '+symbol+' (volume)')
 for field,ls,symbol in [('null_surface','-','surface'),('null_ghost',':','ghost')]:
  ax[1,1].semilogy(t,positive(d[field]),ls,color=color,label=label+' '+symbol)
color='#D55E00';t=one['t_over_M']
ax[0,0].semilogy(t,positive(np.abs(3*one['apparent_horizon_mass']-1)),color=color,label='1D N513')
ax[0,1].plot(t,one['apparent_horizon_r'],color=color,label='1D N513')
for field,ls,label in [('H_exterior_rms','-','1D H (radial)'),('M_exterior_rms',':','1D radial M (radial)')]:
 ax[1,0].semilogy(t,positive(one[field]),ls,color=color,label=label)
ax[1,1].semilogy(t,positive(one['max_relative_null']),color=color,label='1D endpoint')
for a,title,ylabel in zip(ax.flat,['Relative horizon mass error','Accepted horizon coordinate radius','Exterior constraints: unequal norm weights','Maximum relative null closure residual'],['|M_AH / M − 1|','Compactified radius','RMS','Relative residual']):
 a.set_title(title,fontsize=11);a.set_ylabel(ylabel);a.set_xlabel('t / M');a.grid(alpha=.2);a.legend(fontsize=8);a.set_xlim(0,3)
for a in ax[0]:a.axvspan(1.95,3,color='grey',alpha=.10)
fig.suptitle('Puncture: V6 fails at 2.8905 M; V7 fails at 2.9085 M')
fig.supxlabel('3D diagnostics stop at 2.85 M. Grey: no accepted 3D horizon. Constraint ratios are not convergence estimates.',fontsize=9)
fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)
