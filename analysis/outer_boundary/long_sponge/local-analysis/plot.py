"""Plot the collected histories, preserving noise-floor data without fitting it."""
from pathlib import Path
import json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;meta=json.loads((p/'status.json').read_text())
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
fig,ax=plt.subplots(2,3,figsize=(11.2,6.0),layout='constrained');styles={'kappa01':('#c43c39',r'$\kappa=.01,\eta=2,\mu_\alpha=.1$'),'all_weak':('#d18228',r'$\kappa=.01,\eta=.2,\mu_\alpha=.01$'),'no_damping':('#3d8f63',r'$\kappa=\eta=\mu_\alpha=0$'),'gauge_weak':('#8a55a2',r'$\kappa=.1,\eta=.2,\mu_\alpha=.01$'),'fast_weakfield':('#2774ae',r'$\kappa=0,\eta=.02,\mu_\alpha=.01;\ \Delta t=3.2M$')}
for name,rec in meta['cases'].items():
 f=p/(name+'-histories.npz')
 if not f.exists():continue
 z=np.load(f);u=z['user'];h=z['z4c'];ul=list(z['user_labels']);hl=list(z['z4c_labels']);n=min(len(u),len(h));u=u[:n];h=h[:n];t=u[:,0];volume=h[:,hl.index('Volume')];tn='Theta-norm2'if'Theta-norm2'in hl else'Theta-norm';vals=[np.sqrt(h[:,hl.index(tn)]/volume),np.sqrt(h[:,hl.index('H-norm2')]/volume),u[:,ul.index('Theta-max')],u[:,ul.index('alpha-res')],u[:,ul.index('beta-res')],u[:,ul.index('detg-min')]]
 color,label=styles[name]
 for panel,v in zip(ax.ravel(),vals):
  panel.plot(t,np.where(v>0,v,np.nan),color=color,lw=1.25,label=label)
  if rec['status']=='failed':panel.plot(t[-1],v[-1],marker='x',color=color,ms=6)
for panel,label in zip(ax.ravel(),[r'$\sqrt{\int\Theta^2dV/\int dV}$',r'$\sqrt{\int H^2dV/\int dV}$',r'$\max|\Theta|$',r'$\max|\delta\alpha|$',r'$\max_i|\delta\beta^i|$',r'$\min\det\gamma_{ij}$ (active)']):
 panel.set_yscale('log');panel.set_xlabel(r'$t/M$');panel.set_ylabel(label);panel.grid(alpha=.16)
fig.legend(*ax[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
fig.savefig(p/'local-controls.png',dpi=180);fig.savefig(p/'local-controls.pdf');plt.close(fig)
