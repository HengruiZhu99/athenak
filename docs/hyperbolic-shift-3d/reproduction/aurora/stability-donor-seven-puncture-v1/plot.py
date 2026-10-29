#!/usr/bin/env python3
"""Descriptive comparison; exterior norm measures differ, no convergence claim."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
three=np.genfromtxt(root/'raw/output/puncture.history.csv',delimiter=',',names=True)
one=np.genfromtxt(root.parent.parent/'original-1d-reference/history.csv',delimiter=',',names=True)
one=one[one['t_over_M']<=2.9]
t=three['t_over_M'];found=three['horizon_found']==1
missing=t[~found];loss=missing[0] if len(missing) else None
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
ax[0,0].semilogy(t,np.where(found,np.abs(3*three['horizon_mass']-1),np.nan),label='3D n16')
ax[0,0].semilogy(one['t_over_M'],np.abs(3*one['apparent_horizon_mass']-1),label='1D N513')
ax[0,0].set_title('Relative horizon mass error');ax[0,0].set_ylabel('|M_AH / M − 1|')
ax[0,1].plot(t,np.where(found,three['horizon_radius'],np.nan),label='3D n16')
ax[0,1].plot(one['t_over_M'],one['apparent_horizon_r'],label='1D N513')
ax[0,1].set_title('Accepted horizon coordinate radius');ax[0,1].set_ylabel('Compactified radius')
for name,label in [('H_exterior_L2','3D H, volume weights'),('Mr_exterior_L2','3D radial M, volume weights')]:ax[1,0].semilogy(t,three[name],label=label)
for name,label in [('H_exterior_rms','1D H, radial weights'),('M_exterior_rms','1D radial M, radial weights')]:ax[1,0].semilogy(one['t_over_M'],one[name],'--',label=label)
ax[1,0].set_title('Exterior constraints (different norm weights)');ax[1,0].set_ylabel('RMS')
ax[1,1].semilogy(t,three['null_surface'],label='3D surface')
ax[1,1].semilogy(t,three['null_ghost'],label='3D ghosts')
ax[1,1].semilogy(one['t_over_M'],np.maximum(one['max_relative_null'],1e-17),label='1D endpoint')
ax[1,1].set_title('Maximum relative null closure residual');ax[1,1].set_ylabel('Relative residual')
for a in ax.flat:
 a.set_xlabel('t / M');a.legend(fontsize=8);a.grid(alpha=.2)
 a.axvline(2.8905,color='black',ls=':',lw=1)
 if loss is not None:a.axvspan(loss,2.8905,color='grey',alpha=.12)
 a.set_xlim(0,2.95)
fig.suptitle('V6 puncture: failed at 2.8905 M; grey region has no accepted 3D horizon')
fig.supxlabel('Diagnostic curves stop at 2.85 M. Unequal constraint weights preclude direct error-ratio/convergence claims.',fontsize=9)
fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf')
