"""Reproduce comparison from compact archives only; white-background Matplotlib."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained',facecolor='white')
for name,label,color in [('compact','Compact support','#285d9c'),('gaussian','Gaussian','#c25629')]:
    p=P/name
    u=np.load(next(f for f in p.glob('*.user.hst.npz')if '.z4c.'not in f.name))
    z=np.load(next(p.glob('*.z4c.user.hst.npz')))
    axes[0].semilogy(z['time'],np.sqrt(z['Theta-norm']/z['Volume']),label=label,color=color)
    axes[1].semilogy(u['time'],u['alpha-res'],color=color,label=label+' lapse')
    axes[1].semilogy(u['time'],u['beta-res'],color=color,ls='--',label=label+' shift')
for ax in axes:
    ax.set_xlabel(r'$t/M$');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel(r'Exterior $\Theta$ RMS');axes[1].set_ylabel('Maximum gauge residual')
for suffix in ['png','pdf']:fig.savefig(P/('compact-gaussian.'+suffix),dpi=180)
