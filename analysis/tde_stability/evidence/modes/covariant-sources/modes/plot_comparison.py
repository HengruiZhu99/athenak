from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parents[1]/'mode-analysis'
baseline=json.loads((OLD/'validated-modes.json').read_text())
scaled=json.loads((OLD/'scaled-validated-modes.json').read_text())
covariant=json.loads((ROOT/'candidate-validated-modes.json').read_text())
responses=json.loads((ROOT/'sigma1-directional-responses.json').read_text())
assert len(responses)==2

fig,axes=plt.subplots(1,2,figsize=(10.8,3.5),layout='constrained',facecolor='white')
colors=['#1768ac','#c16622']
xx=np.arange(3)
for n in range(2):
    axes[0].plot(xx,[r[n]['gamma'] for r in [baseline,scaled,covariant]],
                 'o-',color=colors[n],label=f'Growing branch {n+1}')
axes[0].set_xticks(xx,['Original\n'+r'$\sigma=0.1\alpha$',
                     'Lapse-scaled only\n'+r'$\sigma=0.3$',
                     'Covariant sources\n'+r'$\sigma=0.3$'])
axes[0].set_ylabel(r'Validated discrete growth rate $\gamma$ [$M^{-1}$]')
axes[0].axhline(0,color='.4',lw=.8)
axes[0].set_ylim(-.001,.052)
axes[0].legend(frameon=False,fontsize=9)
axes[0].grid(axis='y',alpha=.2)
labels=['State\n(active)',r'$\delta H$',r'$\delta M_i$',r'$\delta Q^i$']
for n,r in enumerate(responses):
    gains=[r['active_state']['norm_gain']]+[r['physical_constraints'][f]['norm_gain'] for f in ['H','M_cov','Q_contrav']]
    axes[1].bar(np.arange(4)+(n-.5)*.33,gains,width=.31,color=colors[n],
                label=rf'$\sigma=0.3$ mode {n+1}')
axes[1].axhline(1,color='.3',ls='--',lw=.9)
axes[1].set_xticks(range(4),labels)
axes[1].set_ylabel(r'Norm ratio after $3M$ with $\sigma=1$')
axes[1].set_ylim(0,1.65)
axes[1].text(.02,.98,'Directional responses, not eigenvalues',transform=axes[1].transAxes,
             va='top',fontsize=9)
axes[1].grid(axis='y',alpha=.2)
for fmt in ['png','pdf']:
    fig.savefig(ROOT/f'covariant-mode-comparison.{fmt}',dpi=180)
plt.close(fig)
