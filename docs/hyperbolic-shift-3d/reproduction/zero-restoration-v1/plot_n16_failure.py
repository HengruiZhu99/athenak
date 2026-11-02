"""Compare preserved histories without admitting the failed NZ1 capture."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[4]
d=Path(__file__).resolve().parent/'n16-failure'
a=np.genfromtxt(r/'output/native-shared-transition/aurora/campaigns/n16-native-transition-v2-a-evolution-segment01/output/puncture.history.csv',delimiter=',',names=True)
z=np.genfromtxt(d/'puncture.history.csv',delimiter=',',names=True)
fig,axs=plt.subplots(2,3,figsize=(13,7.5))
for ax,key,title in zip(axs.flat[:4],['H_exterior_L2','M_exterior_L2','pole_max','clock_max'],['Candidate-exterior H','Candidate-exterior M','Scri pole residual','Scri clock residual']):
 for v,label in [(a,'A: original restoration'),(z,'Zero extra restoration')]:ax.semilogy(v['t_over_M'],np.maximum(v[key],1e-18),label=label)
 ax.set_title(title);ax.set_xlim(0,2.4);ax.axvline(2.3145,color='r',ls=':',lw=1);ax.grid(alpha=.25)
for key in ['shift_deviation_max','shift_deviation_gradient_max','D_max','D_gradient_max']:
 axs[1,1].semilogy(z['t_over_M'],np.maximum(z[key],1e-18),label=key)
axs[1,1].set_title('Zero-restoration gauge maxima');axs[1,1].legend(fontsize=7)
for key in ['H_fixed012_L2','M_fixed012_L2','H_fixed025_L2','M_fixed025_L2']:
 axs[1,2].semilogy(z['t_over_M'],z[key],label=key)
axs[1,2].set_title('Zero-restoration fixed coordinate cuts');axs[1,2].legend(fontsize=7)
for ax in axs.flat:ax.set_xlabel('t/M');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8)
fig.suptitle('N16: zero extra restoration fails at 2.3145M; last diagnostic at 2.25M\nHistories only; no admitted final state or matched-cutoff endpoint comparison',fontsize=12)
fig.tight_layout();fig.savefig(d/'histories.png',dpi=160)
