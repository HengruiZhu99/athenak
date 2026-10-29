import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
root=Path(__file__).resolve().parent
a=json.loads((root/'geometry-results.json').read_text());n=np.array([x['n'] for x in a])
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for k,label in [('gradient_l2','Gradient RMS'),('hessian_l2','Hessian RMS')]:
 y=np.array([x[k] for x in a]);ax[0].loglog(n,y,'o-',label=label)
ax[0].loglog(n,a[0]['hessian_l2']*(n[0]/n)**4,'--',color='gray',label='Fourth-order slope')
ax[0].set(xlabel='Cells per block direction',ylabel='Smooth-wave derivative error',title='Spatial test on the inner spherical shell')
ax[0].set_xticks(n,[str(int(x)) for x in n]);ax[0].xaxis.set_minor_locator(NullLocator());ax[0].legend(fontsize=9)
for k,label in [('core_cfl_dt_00025','Cartesian core'),('shell_cfl_dt_00025','Spherical shells')]:
 ax[1].plot(n,[x[k] for x in a],'o-',label=label)
ax[1].axhline(.4,color='gray',linestyle='--',label='Native CFL guard')
ax[1].set(xlabel='Cells per block direction',ylabel='Initial Minkowski characteristic CFL',title='Fixed dt = 0.00025',ylim=(0,.43));ax[1].legend(fontsize=9)
for a in ax:a.grid(True,alpha=.2)
fig.savefig(root/'spatial-and-cfl.png',dpi=170)
