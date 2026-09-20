import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mode_operator import *
results=json.loads((ROOT/'validated-modes.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(10,6.1),layout='constrained',facecolor='white')
for n,rr in enumerate(results):
    v=np.fromfile(ROOT/f'arnoldi_s40_m32_eps0.001-mode{n}.bin').reshape(SHAPE)
    for c,(field,label) in enumerate([(17,r'$\delta\Theta$'),(18,r'$\delta\alpha$'),(1,r'$\delta\tilde{\gamma}_{xx}$')]):
        ax=axes[n,c];a=v[field,12,4:20,4:20];lim=abs(a).max()
        im=ax.imshow(a,extent=(-2,2,-2,2),origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
        ax.add_patch(plt.Circle((0,0),np.sqrt(1-.125**2),fill=False,color='black',ls='--',lw=.75))
        ax.set_xlabel(r'$x/M$');ax.set_ylabel(r'$y/M$')
        ax.set_title(label+rf' ($\gamma={rr["gamma"]:.5f}/M$)',fontsize=10)
        divider=make_axes_locatable(ax);cax=divider.append_axes('right',size='4%',pad=.05)
        cb=fig.colorbar(im,cax=cax);cb.formatter.set_powerlimits((0,0));cb.update_ticks()
fig.suptitle(r'Growing discrete vacuum modes, $G=2$: active-cell slice $z=0.125M$',fontsize=12)
fig.savefig(ROOT/'growing-mode-slices.png',dpi=180,bbox_inches="tight")
fig.savefig(ROOT/'growing-mode-slices.pdf')
plt.close(fig)
# Full-state Ritz convergence; this is not a claim that every Ritz candidate is an eigenvalue.
j=json.loads((ROOT/'arnoldi_s40_m32_eps0.001-results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained',facecolor='white')
for target,color in [(results[0]['gamma'],'#204d8c'),(results[1]['gamma'],'#bb5c23')]:
    kk=[];gg=[];err=[]
    for row in j['history']:
        mode=min(row['ritz'],key=lambda r:abs(r['gamma']-target))
        kk.append(row['k']);gg.append(mode['gamma']);err.append(mode['relative_residual'])
    axes[0].plot(kk,gg,color=color,label=rf'$\gamma={target:.5f}/M$')
    axes[1].semilogy(kk,err,color=color)
axes[0].set_ylim(.025,.055);axes[0].set_ylabel(r'$\log|\mu|/(3M)$');axes[0].legend(frameon=False)
axes[1].set_ylabel('Relative Arnoldi residual')
for ax in axes:ax.set_xlabel('Krylov dimension');ax.grid(alpha=.2)
fig.savefig(ROOT/'mode-convergence.png',dpi=180,bbox_inches="tight")
fig.savefig(ROOT/'mode-convergence.pdf')
