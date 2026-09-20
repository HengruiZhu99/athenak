import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mode_operator import ROOT
fig,axs=plt.subplots(2,3,figsize=(10,6),layout='constrained',facecolor='white')
for n in [0,1]:
 d=np.load(ROOT/f'baseline-mode{n}-physical-constraints.npz')
 for ax,a,label in zip(axs[n],[d['H'][8],d['M'][0,8],d['Q'][0,8]],[r'$\delta H$',r'$\delta M_x$',r'$\delta Q^x$']):
  v=abs(a).max();im=ax.imshow(a,origin='lower',extent=(-2,2,-2,2),cmap='RdBu_r',vmin=-v,vmax=v,interpolation='nearest')
  ax.add_patch(plt.Circle((0,0),np.sqrt(1-.125**2),fill=False,ls='--',color='k',lw=.8))
  ax.set_title(label);ax.set_xlabel(r'$x/M$');ax.set_ylabel(r'$y/M$')
  div=make_axes_locatable(ax);cax=div.append_axes('right',size='4%',pad=.05)
  cb=fig.colorbar(im,cax=cax);cb.formatter.set_powerlimits((0,0));cb.update_ticks()
fig.suptitle('Signed physical constraint perturbations: faster mode (top), slower mode (bottom)',fontsize=11)
fig.savefig(ROOT/'physical-constraint-mode-slices.png',dpi=180,bbox_inches="tight")
fig.savefig(ROOT/'physical-constraint-mode-slices.pdf')
