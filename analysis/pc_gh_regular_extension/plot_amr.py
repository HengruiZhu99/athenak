"""Plot complete pulse errors and the generated-family contamination."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('summary',type=Path);ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args();rows=json.loads(args.summary.read_text())['runs']
fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
for ax,family in zip(axes.flat,['p','Q','L','B']):
    for rate,color in [(0.,'C1'),(1.,'C0')]:
        for smr,style in [(0,'--'),(1,'-')]:
            series=sorted((r for r in rows if r['family']==family and r['rate']==rate and r['smr']==smr),key=lambda r:r['n'])
            ax.loglog([r['n'] for r in series],[r['metrics']['error_l2_over_amplitude'] for r in series],
                      marker='o' if smr else 'x',ls=style,color=color,
                      label=f"lambda={rate:g}, {'SMR' if smr else 'uniform'}")
    ax.set(title=f'{family}-seeded: all 33 components',xlabel='Root resolution N',ylabel='Final L2 error / initial amplitude')
    ax.set_xticks([32,48,64],labels=['32','48','64']);ax.grid(alpha=.2)
ax=axes[1,1]
series=sorted((r for r in rows if 'decomposition' in r),key=lambda r:r['n'])
for family in ['p','Q','L','B']:
    ax.loglog([r['n'] for r in series],[r['decomposition']['groups'][family]['error_l2_over_amplitude'] for r in series],
              'o-',label=family)
ax.set(title='p-seeded SMR, lambda=1: family errors',xlabel='Root resolution N',ylabel='Final L2 error / initial amplitude')
ax.set_xticks([32,48,64],labels=['32','48','64']);ax.legend();ax.grid(alpha=.2)
ax=axes[1,2]
ax.plot([r['n'] for r in series],[r['metrics']['speed_fit'] for r in series],'o-',label='All-component energy centroid')
ax.plot([r['n'] for r in series],[r['decomposition']['groups']['p']['speed_fit'] for r in series],'s-',label='Seeded p-family centroid')
ax.axhline(-.5,color='k',ls=':',label='Continuum pulse speed')
ax.set(title='p-seeded: centroid contamination',xlabel='Root resolution N',ylabel='Fitted speed')
ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
for ax in axes.flat:
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.title.set_fontsize(11)
fig.suptitle('CUDA interface pulse controls: FD2 / RK4, t=4M\nTotal errors retain every component; lower panels expose the p-pulse transfer contamination',fontsize=12)
fig.savefig(args.output,dpi=170)
